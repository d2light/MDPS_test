"""
MDPS 복원 이미지 평가 스크립트
복원된 이미지의 품질을 다양한 지표로 평가합니다.

평가 지표:
    - PSNR (Peak Signal-to-Noise Ratio): 높을수록 좋음 (일반적으로 20-40 dB)
    - SSIM (Structural Similarity Index): 1에 가까울수록 좋음 (0~1)
    - LPIPS (Learned Perceptual Image Patch Similarity): 낮을수록 좋음 (0~1)
    - MSE (Mean Squared Error): 낮을수록 좋음
    - MAE (Mean Absolute Error): 낮을수록 좋음

사용법:
    # 복원 이미지와 원본(불량) 이미지 비교
    python evaluate_reconstruction.py --original data/input --reconstructed data/output --mode defect_vs_recon
    
    # 복원 이미지와 양품(GT) 이미지 비교 (양품 이미지가 있는 경우)
    python evaluate_reconstruction.py --original data/good --reconstructed data/output --mode good_vs_recon
    
    # Anomaly Detection 평가 (마스크 기반)
    python evaluate_reconstruction.py --original data/input --reconstructed data/output --masks data/masks --mode anomaly
"""

import os
import sys
import argparse
import json
from pathlib import Path
from glob import glob
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import pandas as pd

# LPIPS 사용 시도
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("경고: lpips 패키지가 없습니다. pip install lpips로 설치하면 LPIPS 지표를 사용할 수 있습니다.")


class ReconstructionEvaluator:
    """복원 이미지 평가기"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # LPIPS 모델 로드
        self.lpips_model = None
        if LPIPS_AVAILABLE:
            try:
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_model.eval()
            except Exception as e:
                print(f"LPIPS 모델 로드 실패: {e}")
        
        # 이미지 변환
        self.to_tensor = transforms.ToTensor()
    
    def load_image(self, image_path: str, size: int = None) -> np.ndarray:
        """이미지 로드 (0-1 범위의 float numpy array)"""
        img = Image.open(image_path).convert('RGB')
        if size:
            img = img.resize((size, size), Image.BILINEAR)
        return np.array(img).astype(np.float32) / 255.0
    
    def load_mask(self, mask_path: str, size: int = None) -> np.ndarray:
        """마스크 로드 (0-1 범위의 binary numpy array)"""
        mask = Image.open(mask_path).convert('L')
        if size:
            mask = mask.resize((size, size), Image.NEAREST)
        mask_array = np.array(mask).astype(np.float32) / 255.0
        return (mask_array > 0.5).astype(np.float32)
    
    def compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """PSNR 계산 (높을수록 좋음)"""
        return psnr(img1, img2, data_range=1.0)
    
    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """SSIM 계산 (1에 가까울수록 좋음)"""
        return ssim(img1, img2, data_range=1.0, channel_axis=2)
    
    def compute_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """MSE 계산 (낮을수록 좋음)"""
        return np.mean((img1 - img2) ** 2)
    
    def compute_mae(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """MAE 계산 (낮을수록 좋음)"""
        return np.mean(np.abs(img1 - img2))
    
    def compute_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """LPIPS 계산 (낮을수록 좋음, 인지적 유사도)"""
        if self.lpips_model is None:
            return None
        
        # numpy to tensor, normalize to [-1, 1]
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(self.device) * 2 - 1
        t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(self.device) * 2 - 1
        
        with torch.no_grad():
            lpips_value = self.lpips_model(t1, t2)
        
        return lpips_value.item()
    
    def compute_masked_metrics(self, img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> dict:
        """마스크 영역만의 메트릭 계산"""
        # 마스크를 3채널로 확장
        mask_3ch = np.stack([mask] * 3, axis=-1)
        
        # 마스크 영역 추출
        masked_img1 = img1 * mask_3ch
        masked_img2 = img2 * mask_3ch
        
        # 마스크 영역의 픽셀만 추출
        mask_pixels = mask.flatten() > 0
        if mask_pixels.sum() == 0:
            return {'mse': 0, 'mae': 0, 'psnr': float('inf')}
        
        img1_flat = img1.reshape(-1, 3)[mask_pixels]
        img2_flat = img2.reshape(-1, 3)[mask_pixels]
        
        mse = np.mean((img1_flat - img2_flat) ** 2)
        mae = np.mean(np.abs(img1_flat - img2_flat))
        
        # PSNR (마스크 영역)
        if mse > 0:
            psnr_val = 10 * np.log10(1.0 / mse)
        else:
            psnr_val = float('inf')
        
        return {
            'masked_mse': mse,
            'masked_mae': mae,
            'masked_psnr': psnr_val
        }
    
    def evaluate_pair(self, img1_path: str, img2_path: str, mask_path: str = None, size: int = 256) -> dict:
        """이미지 쌍 평가"""
        img1 = self.load_image(img1_path, size)
        img2 = self.load_image(img2_path, size)
        
        results = {
            'psnr': self.compute_psnr(img1, img2),
            'ssim': self.compute_ssim(img1, img2),
            'mse': self.compute_mse(img1, img2),
            'mae': self.compute_mae(img1, img2),
        }
        
        # LPIPS
        lpips_val = self.compute_lpips(img1, img2)
        if lpips_val is not None:
            results['lpips'] = lpips_val
        
        # 마스크 영역 메트릭
        if mask_path and os.path.exists(mask_path):
            mask = self.load_mask(mask_path, size)
            masked_metrics = self.compute_masked_metrics(img1, img2, mask)
            results.update(masked_metrics)
        
        return results
    
    def evaluate_directory(self, 
                          original_dir: str, 
                          reconstructed_dir: str, 
                          mask_dir: str = None,
                          size: int = 256) -> pd.DataFrame:
        """디렉토리 내 모든 이미지 쌍 평가"""
        
        # 이미지 파일 찾기
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        original_files = []
        for ext in extensions:
            original_files.extend(glob(os.path.join(original_dir, ext)))
        
        if not original_files:
            print(f"이미지를 찾을 수 없습니다: {original_dir}")
            return None
        
        results_list = []
        
        for orig_path in original_files:
            filename = Path(orig_path).stem
            
            # 복원 이미지 찾기 (여러 패턴 시도)
            recon_path = None
            for pattern in [f"{filename}_reconstructed.png", f"{filename}.png", f"{filename}_reconstructed.jpg"]:
                possible_path = os.path.join(reconstructed_dir, pattern)
                if os.path.exists(possible_path):
                    recon_path = possible_path
                    break
            
            if not recon_path:
                print(f"복원 이미지를 찾을 수 없음: {filename}")
                continue
            
            # 마스크 찾기
            mask_path = None
            if mask_dir:
                for pattern in [f"{filename}_mask.png", f"{filename}.png", f"{filename}_mask.jpg"]:
                    possible_mask = os.path.join(mask_dir, pattern)
                    if os.path.exists(possible_mask):
                        mask_path = possible_mask
                        break
            
            # 평가
            try:
                metrics = self.evaluate_pair(orig_path, recon_path, mask_path, size)
                metrics['filename'] = filename
                metrics['original'] = orig_path
                metrics['reconstructed'] = recon_path
                results_list.append(metrics)
                print(f"평가 완료: {filename}")
            except Exception as e:
                print(f"평가 실패 ({filename}): {e}")
                continue
        
        if not results_list:
            return None
        
        df = pd.DataFrame(results_list)
        return df


class AnomalyDetectionEvaluator:
    """Anomaly Detection 평가기"""
    
    def __init__(self):
        pass
    
    def load_anomaly_map(self, path: str, size: int = 256) -> np.ndarray:
        """Anomaly map 로드"""
        img = Image.open(path).convert('L')
        if size:
            img = img.resize((size, size), Image.BILINEAR)
        return np.array(img).astype(np.float32) / 255.0
    
    def load_mask(self, path: str, size: int = 256) -> np.ndarray:
        """Ground truth 마스크 로드"""
        mask = Image.open(path).convert('L')
        if size:
            mask = mask.resize((size, size), Image.NEAREST)
        return (np.array(mask) > 127).astype(np.float32)
    
    def compute_image_auroc(self, scores: list, labels: list) -> float:
        """Image-level AUROC"""
        return roc_auc_score(labels, scores)
    
    def compute_pixel_auroc(self, anomaly_maps: list, gt_masks: list) -> float:
        """Pixel-level AUROC"""
        # Flatten all pixels
        all_preds = np.concatenate([am.flatten() for am in anomaly_maps])
        all_gts = np.concatenate([gt.flatten() for gt in gt_masks])
        
        return roc_auc_score(all_gts, all_preds)
    
    def compute_pro_score(self, anomaly_maps: list, gt_masks: list, num_thresholds: int = 200) -> float:
        """Per-Region Overlap (PRO) Score"""
        from scipy.ndimage import label as connected_components
        
        # Normalize anomaly maps
        all_anomaly = np.concatenate([am.flatten() for am in anomaly_maps])
        min_val, max_val = all_anomaly.min(), all_anomaly.max()
        
        thresholds = np.linspace(max_val, min_val, num_thresholds)
        
        pro_scores = []
        fpr_list = []
        
        for threshold in thresholds:
            region_scores = []
            fp_sum = 0
            tn_sum = 0
            
            for anomaly_map, gt_mask in zip(anomaly_maps, gt_masks):
                # 이진화
                pred_mask = (anomaly_map >= threshold).astype(np.float32)
                
                # False positive rate 계산
                fp = np.sum((pred_mask == 1) & (gt_mask == 0))
                tn = np.sum((pred_mask == 0) & (gt_mask == 0))
                fp_sum += fp
                tn_sum += tn
                
                # Connected components로 영역 분리
                labeled_gt, num_regions = connected_components(gt_mask)
                
                for region_id in range(1, num_regions + 1):
                    region_mask = (labeled_gt == region_id)
                    region_overlap = np.sum(pred_mask[region_mask]) / np.sum(region_mask)
                    region_scores.append(region_overlap)
            
            if region_scores:
                pro_scores.append(np.mean(region_scores))
                fpr = fp_sum / (fp_sum + tn_sum) if (fp_sum + tn_sum) > 0 else 0
                fpr_list.append(fpr)
        
        # Integration limit 0.3
        fpr_array = np.array(fpr_list)
        pro_array = np.array(pro_scores)
        
        valid_idx = fpr_array <= 0.3
        if valid_idx.sum() > 0:
            pro_score = np.trapz(pro_array[valid_idx], fpr_array[valid_idx]) / 0.3
        else:
            pro_score = 0.0
        
        return pro_score
    
    def compute_ap(self, scores: list, labels: list) -> float:
        """Average Precision"""
        return average_precision_score(labels, scores)
    
    def evaluate_anomaly_detection(self, 
                                   anomaly_map_dir: str, 
                                   gt_mask_dir: str,
                                   labels: list = None,
                                   size: int = 256) -> dict:
        """Anomaly detection 평가"""
        
        # Anomaly map 파일 찾기
        anomaly_files = glob(os.path.join(anomaly_map_dir, "*_anomaly_map.png"))
        
        if not anomaly_files:
            print(f"Anomaly map을 찾을 수 없습니다: {anomaly_map_dir}")
            return None
        
        anomaly_maps = []
        gt_masks = []
        image_scores = []
        image_labels = []
        
        for am_path in anomaly_files:
            filename = Path(am_path).stem.replace("_anomaly_map", "")
            
            # GT mask 찾기
            gt_path = None
            for pattern in [f"{filename}_mask.png", f"{filename}.png"]:
                possible_gt = os.path.join(gt_mask_dir, pattern)
                if os.path.exists(possible_gt):
                    gt_path = possible_gt
                    break
            
            if not gt_path:
                print(f"GT 마스크를 찾을 수 없음: {filename}")
                continue
            
            try:
                am = self.load_anomaly_map(am_path, size)
                gt = self.load_mask(gt_path, size)
                
                anomaly_maps.append(am)
                gt_masks.append(gt)
                
                # Image-level score (max value)
                image_scores.append(am.max())
                image_labels.append(1 if gt.max() > 0 else 0)
                
            except Exception as e:
                print(f"로드 실패 ({filename}): {e}")
                continue
        
        if not anomaly_maps:
            return None
        
        results = {}
        
        # Pixel-level AUROC
        try:
            results['pixel_auroc'] = self.compute_pixel_auroc(anomaly_maps, gt_masks)
        except Exception as e:
            print(f"Pixel AUROC 계산 실패: {e}")
            results['pixel_auroc'] = None
        
        # Image-level AUROC (불량/양품 분류)
        if len(set(image_labels)) > 1:
            try:
                results['image_auroc'] = self.compute_image_auroc(image_scores, image_labels)
                results['image_ap'] = self.compute_ap(image_scores, image_labels)
            except Exception as e:
                print(f"Image AUROC 계산 실패: {e}")
        
        # PRO Score
        try:
            results['pro_score'] = self.compute_pro_score(anomaly_maps, gt_masks)
        except Exception as e:
            print(f"PRO Score 계산 실패: {e}")
            results['pro_score'] = None
        
        results['num_samples'] = len(anomaly_maps)
        
        return results


def save_results(results: dict, output_path: str, format: str = 'json'):
    """결과 저장"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if format == 'json':
        # numpy/pandas 타입 변환
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            return obj
        
        results_converted = {k: convert(v) for k, v in results.items()}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)
    
    elif format == 'csv' and 'dataframe' in results:
        results['dataframe'].to_csv(output_path, index=False)
    
    print(f"결과 저장: {output_path}")


def print_summary(results: dict, mode: str):
    """결과 요약 출력"""
    print("\n" + "="*60)
    print(f"평가 결과 요약 (Mode: {mode})")
    print("="*60)
    
    if 'dataframe' in results and results['dataframe'] is not None:
        df = results['dataframe']
        print(f"\n총 평가 이미지 수: {len(df)}")
        print("\n[전체 통계]")
        
        metrics = ['psnr', 'ssim', 'mse', 'mae', 'lpips']
        for metric in metrics:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                print(f"  {metric.upper():8s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # 마스크 영역 메트릭
        masked_metrics = ['masked_psnr', 'masked_mse', 'masked_mae']
        has_masked = any(m in df.columns for m in masked_metrics)
        if has_masked:
            print("\n[마스크 영역 통계]")
            for metric in masked_metrics:
                if metric in df.columns:
                    mean_val = df[metric].mean()
                    std_val = df[metric].std()
                    print(f"  {metric:12s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Anomaly Detection 결과
    if 'pixel_auroc' in results:
        print("\n[Anomaly Detection 성능]")
        if results.get('pixel_auroc'):
            print(f"  Pixel AUROC: {results['pixel_auroc']*100:.1f}%")
        if results.get('image_auroc'):
            print(f"  Image AUROC: {results['image_auroc']*100:.1f}%")
        if results.get('image_ap'):
            print(f"  Image AP:    {results['image_ap']*100:.1f}%")
        if results.get('pro_score'):
            print(f"  PRO Score:   {results['pro_score']*100:.1f}%")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='MDPS 복원 이미지 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
    # 복원 이미지 품질 평가 (원본 불량 vs 복원)
    python evaluate_reconstruction.py -o data/input -r data/output --mode quality
    
    # 복원 이미지와 양품 비교 (양품 이미지가 있는 경우)  
    python evaluate_reconstruction.py -o data/good -r data/output --mode quality
    
    # Anomaly Detection 성능 평가
    python evaluate_reconstruction.py -r data/output -g data/ground_truth --mode anomaly
        """
    )
    
    parser.add_argument('--original', '-o', default=None,
                       help='원본 이미지 디렉토리')
    parser.add_argument('--reconstructed', '-r', required=True,
                       help='복원 이미지 디렉토리')
    parser.add_argument('--masks', '-m', default=None,
                       help='마스크 디렉토리 (복원 품질 평가 시)')
    parser.add_argument('--ground-truth', '-g', default=None,
                       help='Ground truth 마스크 디렉토리 (anomaly detection 평가 시)')
    parser.add_argument('--mode', choices=['quality', 'anomaly', 'both'], default='quality',
                       help='평가 모드')
    parser.add_argument('--size', type=int, default=256,
                       help='이미지 크기')
    parser.add_argument('--output', default='evaluation_results',
                       help='결과 저장 경로 (확장자 없이)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='사용할 디바이스')
    
    args = parser.parse_args()
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 복원 품질 평가
    if args.mode in ['quality', 'both']:
        if not args.original:
            print("복원 품질 평가를 위해 --original 옵션이 필요합니다.")
        else:
            print("\n[복원 품질 평가 시작]")
            evaluator = ReconstructionEvaluator(device=args.device)
            df = evaluator.evaluate_directory(
                args.original, 
                args.reconstructed, 
                args.masks,
                args.size
            )
            
            if df is not None:
                results['dataframe'] = df
                results['summary'] = {
                    'num_samples': len(df),
                    'mean_psnr': df['psnr'].mean() if 'psnr' in df else None,
                    'mean_ssim': df['ssim'].mean() if 'ssim' in df else None,
                    'mean_lpips': df['lpips'].mean() if 'lpips' in df else None,
                }
    
    # Anomaly Detection 평가
    if args.mode in ['anomaly', 'both']:
        if not args.ground_truth:
            print("Anomaly detection 평가를 위해 --ground-truth 옵션이 필요합니다.")
        else:
            print("\n[Anomaly Detection 평가 시작]")
            ad_evaluator = AnomalyDetectionEvaluator()
            ad_results = ad_evaluator.evaluate_anomaly_detection(
                args.reconstructed,
                args.ground_truth,
                size=args.size
            )
            
            if ad_results:
                results.update(ad_results)
    
    # 결과 출력 및 저장
    if results:
        print_summary(results, args.mode)
        
        # JSON 저장
        json_path = f"{args.output}_{timestamp}.json"
        save_results(results, json_path, 'json')
        
        # CSV 저장 (품질 평가 결과)
        if 'dataframe' in results and results['dataframe'] is not None:
            csv_path = f"{args.output}_{timestamp}.csv"
            results['dataframe'].to_csv(csv_path, index=False)
            print(f"상세 결과 CSV 저장: {csv_path}")
    else:
        print("평가할 수 없습니다. 입력 경로를 확인하세요.")


if __name__ == "__main__":
    main()


