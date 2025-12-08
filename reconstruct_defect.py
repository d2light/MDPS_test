"""
MDPS 기반 불량 이미지 복원 스크립트
불량(결함) 이미지를 양품 이미지에 가깝게 복원합니다.

사용법:
    python reconstruct_defect.py --input <이미지 경로> --output <출력 경로>
    python reconstruct_defect.py --input data/input --output data/output --config config/default.yaml
"""

import os
import sys
import argparse
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.models.unet import UNetModel
from src.models.resnet import Resnet
from src.diffusion import sample, sample_mask, compute_alpha
from src.compare import distance


class MDPSReconstructor:
    """MDPS 기반 불량 이미지 복원기"""
    
    def __init__(self, config_path: str = None, checkpoint_path: str = None):
        """
        Args:
            config_path: 설정 파일 경로
            checkpoint_path: 모델 체크포인트 경로
        """
        # 설정 로드
        if config_path and os.path.exists(config_path):
            self.config = OmegaConf.load(config_path)
        else:
            self.config = self._get_default_config()
        
        # 디바이스 설정
        self.device = self._setup_device()
        self.config.model.device = self.device
        
        # 모델 초기화
        self.unet = None
        self.resnet = None
        self.checkpoint_loaded = False
        
        # 체크포인트 로드
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # 이미지 변환 설정
        self.image_transform = transforms.Compose([
            transforms.Resize((self.config.data.image_size, self.config.data.image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)  # [-1, 1] 범위로 정규화
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),  # [0, 1] 범위로 역정규화
            transforms.Lambda(lambda t: t.clamp(0, 1))
        ])
    
    def _get_default_config(self):
        """기본 설정 반환"""
        config = OmegaConf.create({
            'data': {
                'name': 'Custom',
                'image_size': 256,
                'batch_size': 1,
                'mask': True,
                'imput_channel': 3
            },
            'model': {
                'checkpoint_dir': 'checkpoints',
                'resnet': 'wide_resnet101_2',
                'eta': 7,
                'diffusion_steps': 1000,
                'test_steps': 200,
                'skip': 20,
                'w': 100,
                'w_mask': 50,
                'mask_steps': 200,
                'skip_mask': 20,
                'mask0_thresholds': 0.12,
                'mask_repeat': 1,
                'test_repeat': 1,
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'device': 'cuda',
                'num_workers': 4
            },
            'visualization': {
                'save_intermediate': False,
                'save_anomaly_map': True,
                'colormap': 'jet'
            }
        })
        return config
    
    def _setup_device(self):
        """디바이스 설정"""
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("CPU 사용 (GPU를 사용하면 더 빠릅니다)")
        return device
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        
        print(f"체크포인트 로딩: {checkpoint_path}")
        
        # UNet 모델 초기화 및 로드
        self.unet = UNetModel(
            self.config.data.image_size, 
            64, 
            dropout=0.0, 
            n_heads=4,
            in_channels=self.config.data.imput_channel
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 체크포인트에서 state_dict 추출
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # DataParallel로 저장된 체크포인트 처리 (module. 접두사 제거)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # DataParallel로 저장된 경우: 'module.' 제거
                new_key = k[7:]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        # 모델 로드 (strict=False로 일부 키가 없어도 로드 가능)
        missing_keys, unexpected_keys = self.unet.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"경고: {len(missing_keys)}개의 키가 모델에 없습니다 (일부는 정상일 수 있음)")
        if unexpected_keys:
            print(f"경고: {len(unexpected_keys)}개의 예상치 못한 키가 있습니다")
        
        self.unet.to(self.device)
        
        # DataParallel로 감싸기
        self.unet = torch.nn.DataParallel(self.unet)
        self.unet.eval()
        
        # ResNet 특징 추출기 초기화
        self.resnet = Resnet(self.config)
        self.resnet.to(self.device)
        self.resnet.eval()
        
        self.checkpoint_loaded = True
        print("모델 로드 완료!")
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """이미지 로드 및 전처리
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            전처리된 이미지 텐서 [1, C, H, W]
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image)
        
        # 그레이스케일 이미지 처리
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.expand(3, -1, -1)
        
        return image_tensor.unsqueeze(0)  # 배치 차원 추가
    
    def load_mask(self, mask_path: str) -> torch.Tensor:
        """마스크 이미지 로드 및 전처리
        
        Args:
            mask_path: 마스크 이미지 파일 경로
            
        Returns:
            마스크 텐서 [1, 1, H, W] (0: 정상, 1: 불량 영역)
        """
        mask_transform = transforms.Compose([
            transforms.Resize((self.config.data.image_size, self.config.data.image_size)),
            transforms.ToTensor(),
        ])
        
        mask = Image.open(mask_path).convert('L')  # 그레이스케일로 로드
        mask_tensor = mask_transform(mask)
        
        # 이진화 (0.5 이상이면 불량 영역)
        mask_tensor = (mask_tensor > 0.5).float()
        
        return mask_tensor.unsqueeze(0)  # 배치 차원 추가
    
    def reconstruct(self, image_tensor: torch.Tensor, use_mask: bool = True, external_mask: torch.Tensor = None) -> dict:
        """불량 이미지를 양품에 가깝게 복원
        
        Args:
            image_tensor: 입력 이미지 텐서 [B, C, H, W]
            use_mask: 마스크 기반 복원 사용 여부
            external_mask: 외부 마스크 텐서 [B, 1, H, W] (제공 시 자동 마스크 생성 건너뜀)
            
        Returns:
            복원 결과 딕셔너리 (reconstructed, anomaly_map, mask)
        """
        if not self.checkpoint_loaded:
            raise RuntimeError("체크포인트를 먼저 로드하세요: load_checkpoint()")
        
        data = image_tensor.to(self.device)
        results = {
            'original': data.cpu(),
            'reconstructed': None,
            'anomaly_map': None,
            'mask': None
        }
        
        with torch.no_grad():
            # 외부 마스크가 제공된 경우
            if external_mask is not None:
                print("외부 마스크 사용")
                mask = external_mask.to(self.device)
                results['mask'] = mask.cpu()
                
                # 마스크 기반 복원
                seq = range(0, self.config.model.test_steps, self.config.model.skip)
                final_anomaly_batch = []
                
                for i in range(self.config.model.test_repeat):
                    reconstructed = sample_mask(data, mask, seq, self.unet, self.config, w=self.config.model.w)
                    data_reconstructed = reconstructed[-1].to(self.device)
                    anomaly_map = distance(data_reconstructed, data, self.resnet, self.config) / 2
                    final_anomaly_batch.append(anomaly_map.unsqueeze(0))
                
                final_anomaly_batch = torch.cat(final_anomaly_batch, dim=0)
                final_anomaly_map = torch.mean(final_anomaly_batch, dim=0)
                
                results['reconstructed'] = data_reconstructed.cpu()
                results['anomaly_map'] = final_anomaly_map.cpu()
            
            elif use_mask and self.config.model.mask_steps > 0:
                # 1단계: 초기 anomaly map으로 마스크 생성
                mask_steps = torch.Tensor([self.config.model.mask_steps]).type(torch.int64).to(self.device)
                at = compute_alpha(mask_steps.long(), self.config)
                seq = range(0, self.config.model.mask_steps, self.config.model.skip_mask)
                
                anomaly_batch = []
                for i in range(self.config.model.mask_repeat):
                    noisy_image = at.sqrt() * data + (1 - at).sqrt() * torch.randn_like(data).to(self.device)
                    reconstructed = sample(data, noisy_image, seq, self.unet, self.config, w=self.config.model.w_mask)
                    data_reconstructed = reconstructed[-1].to(self.device)
                    anomaly_map = distance(data_reconstructed, data, self.resnet, self.config) / 2
                    anomaly_batch.append(anomaly_map.unsqueeze(0))
                
                anomaly_batch = torch.cat(anomaly_batch, dim=0)
                anomaly_map = torch.mean(anomaly_batch, dim=0)
                
                # 마스크 생성
                pixel_min = torch.min(anomaly_map)
                pixel_max = torch.max(anomaly_map)
                threshold = pixel_min + self.config.model.mask0_thresholds * (pixel_max - pixel_min)
                mask = torch.where(anomaly_map > threshold, torch.ones_like(anomaly_map), torch.zeros_like(anomaly_map))
                results['mask'] = mask.cpu()
                
                # 2단계: 마스크 기반 복원
                seq = range(0, self.config.model.test_steps, self.config.model.skip)
                final_anomaly_batch = []
                
                for i in range(self.config.model.test_repeat):
                    reconstructed = sample_mask(data, mask, seq, self.unet, self.config, w=self.config.model.w)
                    data_reconstructed = reconstructed[-1].to(self.device)
                    anomaly_map = distance(data_reconstructed, data, self.resnet, self.config) / 2
                    final_anomaly_batch.append(anomaly_map.unsqueeze(0))
                
                final_anomaly_batch = torch.cat(final_anomaly_batch, dim=0)
                final_anomaly_map = torch.mean(final_anomaly_batch, dim=0)
                
                results['reconstructed'] = data_reconstructed.cpu()
                results['anomaly_map'] = final_anomaly_map.cpu()
            
            else:
                # 마스크 없이 복원
                test_steps = torch.Tensor([self.config.model.test_steps]).type(torch.int64).to(self.device)
                at = compute_alpha(test_steps.long(), self.config)
                seq = range(0, self.config.model.test_steps, self.config.model.skip)
                
                anomaly_batch = []
                for i in range(self.config.model.test_repeat):
                    noisy_image = at.sqrt() * data + (1 - at).sqrt() * torch.randn_like(data).to(self.device)
                    reconstructed = sample(data, noisy_image, seq, self.unet, self.config, w=self.config.model.w)
                    data_reconstructed = reconstructed[-1].to(self.device)
                    anomaly_map = distance(data_reconstructed, data, self.resnet, self.config) / 2
                    anomaly_batch.append(anomaly_map.unsqueeze(0))
                
                anomaly_batch = torch.cat(anomaly_batch, dim=0)
                final_anomaly_map = torch.mean(anomaly_batch, dim=0)
                
                results['reconstructed'] = data_reconstructed.cpu()
                results['anomaly_map'] = final_anomaly_map.cpu()
        
        return results
    
    def reconstruct_from_file(self, image_path: str, mask_path: str = None, use_mask: bool = True) -> dict:
        """파일에서 이미지를 로드하여 복원
        
        Args:
            image_path: 입력 이미지 경로
            mask_path: 마스크 이미지 경로 (선택사항, 제공 시 해당 마스크 사용)
            use_mask: 마스크 기반 복원 사용 여부
            
        Returns:
            복원 결과 딕셔너리
        """
        image_tensor = self.load_image(image_path)
        
        external_mask = None
        if mask_path and os.path.exists(mask_path):
            external_mask = self.load_mask(mask_path)
            print(f"마스크 로드: {mask_path}")
        
        return self.reconstruct(image_tensor, use_mask, external_mask)
    
    def save_results(self, results: dict, output_dir: str, filename: str = "result"):
        """복원 결과 저장
        
        Args:
            results: 복원 결과 딕셔너리
            output_dir: 출력 디렉토리
            filename: 파일명 접두사
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 원본 이미지 저장
        original = self.inverse_transform(results['original'][0])
        original_pil = transforms.ToPILImage()(original)
        original_pil.save(os.path.join(output_dir, f"{filename}_original.png"))
        
        # 복원 이미지 저장
        if results['reconstructed'] is not None:
            reconstructed = self.inverse_transform(results['reconstructed'][0])
            reconstructed_pil = transforms.ToPILImage()(reconstructed)
            reconstructed_pil.save(os.path.join(output_dir, f"{filename}_reconstructed.png"))
        
        # Anomaly map 저장
        if results['anomaly_map'] is not None:
            anomaly_map = results['anomaly_map'][0, 0].numpy()
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(anomaly_map, cmap=self.config.visualization.colormap)
            plt.colorbar(label='Anomaly Score')
            plt.title('Anomaly Map')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"{filename}_anomaly_map.png"), 
                       bbox_inches='tight', dpi=150)
            plt.close()
        
        # 마스크 저장
        if results['mask'] is not None:
            mask = results['mask'][0, 0].numpy()
            plt.figure(figsize=(8, 8))
            plt.imshow(mask, cmap='gray')
            plt.title('Defect Mask')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"{filename}_mask.png"), 
                       bbox_inches='tight', dpi=150)
            plt.close()
        
        # 비교 이미지 생성
        self._save_comparison(results, output_dir, filename)
        
        print(f"결과 저장 완료: {output_dir}")
    
    def _save_comparison(self, results: dict, output_dir: str, filename: str):
        """원본/복원/anomaly map 비교 이미지 저장"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 원본 이미지
        original = self.inverse_transform(results['original'][0]).permute(1, 2, 0).numpy()
        axes[0].imshow(original)
        axes[0].set_title('Original (Defective)', fontsize=14)
        axes[0].axis('off')
        
        # 복원 이미지
        if results['reconstructed'] is not None:
            reconstructed = self.inverse_transform(results['reconstructed'][0]).permute(1, 2, 0).numpy()
            axes[1].imshow(reconstructed)
            axes[1].set_title('Reconstructed (Normal-like)', fontsize=14)
        axes[1].axis('off')
        
        # Anomaly map
        if results['anomaly_map'] is not None:
            anomaly_map = results['anomaly_map'][0, 0].numpy()
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            im = axes[2].imshow(anomaly_map, cmap=self.config.visualization.colormap)
            axes[2].set_title('Anomaly Map', fontsize=14)
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        axes[2].axis('off')
        
        plt.suptitle('MDPS Defect Reconstruction Result', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}_comparison.png"), 
                   bbox_inches='tight', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='MDPS 기반 불량 이미지 복원',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
    # 단일 이미지 복원
    python reconstruct_defect.py --input image.png --output output/ --checkpoint model.pth
    
    # 폴더 내 모든 이미지 복원
    python reconstruct_defect.py --input data/input/ --output data/output/ --checkpoint checkpoints/bottle/2000
        """
    )
    
    parser.add_argument('--input', '-i', required=True, 
                       help='입력 이미지 경로 또는 디렉토리')
    parser.add_argument('--mask', '-m', default=None,
                       help='마스크 이미지 경로 또는 디렉토리 (선택사항)')
    parser.add_argument('--output', '-o', default='data/output',
                       help='출력 디렉토리 (기본값: data/output)')
    parser.add_argument('--checkpoint', '-c', required=True,
                       help='모델 체크포인트 경로')
    parser.add_argument('--config', default='config/default.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--no-mask', action='store_true',
                       help='마스크 기반 복원 비활성화')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None,
                       help='사용할 디바이스')
    
    args = parser.parse_args()
    
    # 복원기 초기화
    reconstructor = MDPSReconstructor(
        config_path=args.config if os.path.exists(args.config) else None,
        checkpoint_path=args.checkpoint
    )
    
    # 디바이스 오버라이드
    if args.device:
        reconstructor.config.model.device = args.device
        reconstructor.device = args.device
    
    # 입력 처리
    input_path = Path(args.input)
    mask_path = Path(args.mask) if args.mask else None
    use_mask = not args.no_mask
    
    if input_path.is_file():
        # 단일 이미지 처리
        print(f"\n이미지 복원 중: {input_path}")
        
        # 마스크 경로 결정
        mask_file = None
        if mask_path:
            if mask_path.is_file():
                mask_file = str(mask_path)
            elif mask_path.is_dir():
                # 같은 이름의 마스크 파일 찾기
                possible_mask = mask_path / (input_path.stem + "_mask.png")
                if possible_mask.exists():
                    mask_file = str(possible_mask)
                else:
                    possible_mask = mask_path / (input_path.stem + ".png")
                    if possible_mask.exists():
                        mask_file = str(possible_mask)
        
        results = reconstructor.reconstruct_from_file(str(input_path), mask_path=mask_file, use_mask=use_mask)
        reconstructor.save_results(results, args.output, input_path.stem)
        
    elif input_path.is_dir():
        # 디렉토리 내 모든 이미지 처리
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(str(input_path / ext)))
        
        if not image_files:
            print(f"이미지를 찾을 수 없습니다: {input_path}")
            return
        
        print(f"\n{len(image_files)}개의 이미지 발견")
        
        for idx, image_file in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] 복원 중: {image_file}")
            try:
                # 마스크 경로 결정
                mask_file = None
                if mask_path and mask_path.is_dir():
                    img_stem = Path(image_file).stem
                    # 여러 마스크 파일명 패턴 시도
                    for pattern in [f"{img_stem}_mask.png", f"{img_stem}.png", f"{img_stem}_mask.jpg", f"{img_stem}.jpg"]:
                        possible_mask = mask_path / pattern
                        if possible_mask.exists():
                            mask_file = str(possible_mask)
                            break
                
                results = reconstructor.reconstruct_from_file(image_file, mask_path=mask_file, use_mask=use_mask)
                reconstructor.save_results(results, args.output, Path(image_file).stem)
            except Exception as e:
                print(f"오류 발생: {e}")
                continue
    else:
        print(f"유효하지 않은 경로: {input_path}")
        return
    
    print("\n모든 복원 완료!")


if __name__ == "__main__":
    main()

