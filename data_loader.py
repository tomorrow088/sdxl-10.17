"""
数据加载器模块 - 处理迷彩图像和掩码数据
"""
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from typing import Tuple, List, Optional, Dict
import random
import glob

from config import training_config, model_config

class CamouflageDataset(Dataset):
    """迷彩数据集类"""
    
    def __init__(self, 
                 train_dir: str,
                 mask_dir: str, 
                 image_size: Tuple[int, int] = (512, 512),
                 is_training: bool = True,
                 augment: bool = True):
        """
        初始化数据集
        
        Args:
            train_dir: 训练图像目录
            mask_dir: 掩码图像目录
            image_size: 图像尺寸
            is_training: 是否为训练模式
            augment: 是否进行数据增强
        """
        self.train_dir = train_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.is_training = is_training
        self.augment = augment
        
        # 调试路径信息
        print(f"训练图像目录: {train_dir}")
        print(f"掩码图像目录: {mask_dir}")
        print(f"训练目录是否存在: {os.path.exists(train_dir)}")
        print(f"掩码目录是否存在: {os.path.exists(mask_dir)}")
        
        # 获取所有图像文件
        self.train_images = sorted(glob.glob(os.path.join(train_dir, "*.jpg")))
        self.mask_images = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        
        # 如果没找到jpg文件，尝试其他扩展名
        if len(self.train_images) == 0:
            print("未找到.jpg文件，尝试其他格式...")
            for ext in ["*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
                alt_images = sorted(glob.glob(os.path.join(train_dir, ext)))
                if alt_images:
                    print(f"找到 {len(alt_images)} 个 {ext} 文件")
                    self.train_images = alt_images
                    break
        
        # 确保图像和掩码数量匹配
        print(f"找到 {len(self.train_images)} 张训练图像")
        print(f"找到 {len(self.mask_images)} 张掩码图像")
        
        # 如果数据太少，给出警告
        if len(self.train_images) < 4:
            print(f"?? 警告：训练图像数量太少({len(self.train_images)})，可能影响批量训练效果")
            print(f"   建议：确保 {train_dir} 中有至少4张图像，以支持batch_size=4的训练")
        
        # 创建文件名映射
        self.image_pairs = self._create_image_pairs()
        
        # 定义变换
        self.transform = self._get_transforms()
        self.mask_transform = self._get_mask_transforms()
    
    def _create_image_pairs(self) -> List[Tuple[str, str]]:
        """创建图像和掩码的配对"""
        pairs = []
        
        for train_img in self.train_images:
            # 提取文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(train_img))[0]
            
            # 查找对应的掩码文件
            mask_path = os.path.join(self.mask_dir, f"{base_name}.png")
            
            if os.path.exists(mask_path):
                pairs.append((train_img, mask_path))
            else:
                # 如果没有精确匹配，尝试随机选择一个掩码
                if self.mask_images:
                    random_mask = random.choice(self.mask_images)
                    pairs.append((train_img, random_mask))
        
        print(f"成功配对 {len(pairs)} 组图像")
        return pairs
    
    def _get_transforms(self):
        """获取图像变换"""
        transforms_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        
        if self.is_training and self.augment:
            # 仅保留颜色抖动在图像端；几何增强改为在__getitem__中对图像与掩码一致地应用
            transforms_list.insert(-2, transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ))
        
        return transforms.Compose(transforms_list)
    
    def _get_mask_transforms(self):
        """获取掩码变换"""
        return transforms.Compose([
            transforms.Resize(self.image_size, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据项"""
        train_path, mask_path = self.image_pairs[idx]
        
        # 加载图像
        image = Image.open(train_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # 几何增强：对图像与掩码一致地应用，避免位置错位/镜像问题
        if self.is_training and self.augment:
            # 水平翻转
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # 轻微旋转（图像双线性、掩码最近邻）
            angle = random.uniform(-5.0, 5.0)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, expand=False, fill=0)

        # 之后再做Resize/颜色/张量化（颜色抖动仅在图像端）
        image = self.transform(image)
        mask = self.mask_transform(mask)
        
        # 将掩码转换为二值掩码
        mask = (mask > 0.5).float()
        
        return {
            'image': image,
            'mask': mask,
            'image_path': train_path,
            'mask_path': mask_path
        }

class CamouflageDataLoader:
    """迷彩数据加载器管理器"""
    
    def __init__(self,
                 train_dir: str = None,
                 test_dir: str = None,
                 mask_train_dir: str = None,
                 mask_test_dir: str = None,
                 batch_size: int = None,
                 num_workers: int = 6,  # 增加workers数量以提高数据加载效率
                 use_separate_test: bool = True):
        """
        初始化数据加载器管理器
        
        Args:
            train_dir: 训练图像目录
            test_dir: 测试图像目录
            mask_train_dir: 训练掩码目录
            mask_test_dir: 测试掩码目录
            batch_size: 批次大小
            num_workers: 工作进程数
            use_separate_test: 是否使用独立的测试集
        """
        self.train_dir = train_dir or training_config.train_data_dir
        self.test_dir = test_dir or training_config.test_data_dir
        self.mask_train_dir = mask_train_dir or training_config.mask_train_dir
        self.mask_test_dir = mask_test_dir or training_config.mask_test_dir
        self.batch_size = batch_size or training_config.batch_size
        self.num_workers = num_workers
        self.use_separate_test = use_separate_test
        
        # 创建训练数据集
        self.train_dataset = CamouflageDataset(
            train_dir=self.train_dir,
            mask_dir=self.mask_train_dir,
            image_size=model_config.image_size,
            is_training=True,
            augment=True
        )
        
        # 从训练集分割出验证集（不使用独立测试集进行训练）
        print("从训练集分割验证集...")
        self._split_dataset()
    
    def _split_dataset(self):
        """分割数据集"""
        dataset_size = len(self.train_dataset)
        val_size = int(dataset_size * 0.2)  # 20%作为验证集
        train_size = dataset_size - val_size
        
        # 随机分割
        torch.manual_seed(42)
        indices = list(range(dataset_size))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # 创建子数据集
        full_train_dataset = self.train_dataset
        self.train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        self.test_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)
        
        print(f"训练集大小: {len(self.train_dataset)}")
        print(f"验证集大小: {len(self.test_dataset)}")
    
    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,  # 启用持久化workers
            prefetch_factor=4,        # 增加预取因子
            drop_last=False  # 改为False，保留所有数据，即使批次不完整
        )
    
    def get_test_loader(self) -> DataLoader:
        """获取测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,  # 启用持久化workers
            prefetch_factor=4,        # 增加预取因子
            drop_last=False
        )

def create_data_loaders(train_dir: str = None,
                       test_dir: str = None,
                       mask_train_dir: str = None,
                       mask_test_dir: str = None,
                       batch_size: int = None) -> Tuple[DataLoader, DataLoader]:
    """
    便捷函数：创建训练和测试数据加载器
    
    Returns:
        训练数据加载器和测试数据加载器的元组
    """
    data_manager = CamouflageDataLoader(
        train_dir=train_dir,
        test_dir=test_dir,
        mask_train_dir=mask_train_dir,
        mask_test_dir=mask_test_dir,
        batch_size=batch_size
    )
    
    return data_manager.get_train_loader(), data_manager.get_test_loader()

if __name__ == "__main__":
    # 测试数据加载器
    print("测试数据加载器...")
    
    train_loader, test_loader = create_data_loaders()
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 测试一个批次
    for batch in train_loader:
        print(f"图像形状: {batch['image'].shape}")
        print(f"掩码形状: {batch['mask'].shape}")
        print(f"图像数据范围: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        print(f"掩码数据范围: [{batch['mask'].min():.3f}, {batch['mask'].max():.3f}]")
        break
    
    print("数据加载器测试完成！")
