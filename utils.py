"""
工具模块 - 图像处理、评估指标和辅助功能
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import os
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance
try:
    from torchmetrics.image.lpips import LearnedPerceptualImagePatchSimilarity
except ImportError:
    try:
        from torchmetrics import LearnedPerceptualImagePatchSimilarity
    except ImportError:
        # 如果都导入失败，使用占位符
        LearnedPerceptualImagePatchSimilarity = None
import json

class ImageProcessor:
    """图像处理工具类"""
    
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """
        将张量转换为PIL图像
        
        Args:
            tensor: 图像张量 [C, H, W] 或 [B, C, H, W]
            
        Returns:
            PIL图像
        """
        if tensor.dim() == 4:
            tensor = tensor[0]  # 取第一张图像
        
        # 反归一化
        tensor = (tensor * 0.5) + 0.5
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为numpy数组
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        img_array = tensor.permute(1, 2, 0).numpy()
        img_array = (img_array * 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    @staticmethod
    def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
        """
        将PIL图像转换为张量
        
        Args:
            image: PIL图像
            normalize: 是否归一化到[-1, 1]
            
        Returns:
            图像张量
        """
        # 转换为numpy数组
        img_array = np.array(image)
        if len(img_array.shape) == 2:  # 灰度图
            img_array = np.expand_dims(img_array, axis=-1)
        
        # 转换为张量
        tensor = torch.from_numpy(img_array).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # [C, H, W]
        
        if normalize:
            tensor = (tensor - 0.5) / 0.5  # 归一化到[-1, 1]
        
        return tensor
    
    @staticmethod
    def apply_mask_to_image(image: Union[torch.Tensor, Image.Image],
                          mask: Union[torch.Tensor, Image.Image],
                          background_color: Tuple[int, int, int] = (0, 0, 0)) -> Union[torch.Tensor, Image.Image]:
        """
        将掩码应用到图像上
        
        Args:
            image: 输入图像
            mask: 掩码
            background_color: 背景颜色
            
        Returns:
            掩码处理后的图像
        """
        if isinstance(image, torch.Tensor):
            # 张量处理
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # 添加通道维度
            if mask.shape[0] == 1:
                mask = mask.repeat(3, 1, 1)  # 扩展到RGB
            
            # 应用掩码
            masked_image = image * mask
            
            # 添加背景色
            background = torch.tensor(background_color).float().view(3, 1, 1) / 255.0
            if image.min() < 0:  # 如果图像已归一化到[-1,1]
                background = (background - 0.5) / 0.5
            
            masked_image = masked_image + background * (1 - mask)
            
            return masked_image
        else:
            # PIL图像处理
            if isinstance(mask, torch.Tensor):
                mask = ImageProcessor.tensor_to_pil(mask)
            
            # 确保掩码是L模式
            if mask.mode != 'L':
                mask = mask.convert('L')
            
            # 创建alpha通道
            image_rgba = image.convert('RGBA')
            image_rgba.putalpha(mask)
            
            # 创建背景
            background = Image.new('RGB', image.size, background_color)
            result = Image.alpha_composite(background.convert('RGBA'), image_rgba)
            
            return result.convert('RGB')
    
    @staticmethod
    def blend_images(image1: torch.Tensor,
                    image2: torch.Tensor,
                    mask: torch.Tensor,
                    blend_mode: str = 'normal') -> torch.Tensor:
        """
        混合两张图像
        
        Args:
            image1: 第一张图像（背景）
            image2: 第二张图像（前景）
            mask: 混合掩码
            blend_mode: 混合模式 ('normal', 'multiply', 'screen', 'overlay')
            
        Returns:
            混合后的图像
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1:
            mask = mask.repeat(3, 1, 1)
        
        if blend_mode == 'normal':
            result = image1 * (1 - mask) + image2 * mask
        elif blend_mode == 'multiply':
            result = image1 * (1 - mask) + (image1 * image2) * mask
        elif blend_mode == 'screen':
            result = image1 * (1 - mask) + (1 - (1 - image1) * (1 - image2)) * mask
        elif blend_mode == 'overlay':
            overlay = torch.where(
                image1 < 0.5,
                2 * image1 * image2,
                1 - 2 * (1 - image1) * (1 - image2)
            )
            result = image1 * (1 - mask) + overlay * mask
        else:
            result = image1 * (1 - mask) + image2 * mask
        
        return result
    
    @staticmethod
    def resize_with_aspect_ratio(image: Image.Image,
                               target_size: Tuple[int, int],
                               fill_color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
        """
        保持宽高比的图像缩放
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)
            fill_color: 填充颜色
            
        Returns:
            缩放后的图像
        """
        original_size = image.size
        ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
        
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # 创建目标大小的画布
        result = Image.new('RGB', target_size, fill_color)
        
        # 居中粘贴
        paste_x = (target_size[0] - new_size[0]) // 2
        paste_y = (target_size[1] - new_size[1]) // 2
        result.paste(image, (paste_x, paste_y))
        
        return result

class MetricsCalculator:
    """评估指标计算器（仅评估阶段调用，训练阶段请勿调用以避免CPU瓶颈）"""
    
    def __init__(self, device: str = "cuda"):
        """
        初始化指标计算器
        
        Args:
            device: 计算设备
        """
        self.device = device
        
        # 初始化FID计算器
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        
        # 初始化LPIPS计算器
        if LearnedPerceptualImagePatchSimilarity is not None:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type='alex', normalize=True
            ).to(device)
        else:
            self.lpips = None
    
    def compute_fid(self, 
                   real_images: torch.Tensor,
                   generated_images: torch.Tensor) -> float:
        """
        计算FID分数
        
        Args:
            real_images: 真实图像 [B, C, H, W]
            generated_images: 生成图像 [B, C, H, W]
            
        Returns:
            FID分数
        """
        # 确保图像在[0, 1]范围内
        real_images = (real_images * 0.5) + 0.5
        generated_images = (generated_images * 0.5) + 0.5
        
        # 转换为uint8
        real_images = (real_images * 255).clamp(0, 255).to(torch.uint8)
        generated_images = (generated_images * 255).clamp(0, 255).to(torch.uint8)
        
        # 计算FID
        self.fid.update(real_images, real=True)
        self.fid.update(generated_images, real=False)
        fid_score = self.fid.compute().item()
        
        self.fid.reset()
        return fid_score
    
    def compute_lpips(self,
                     image1: torch.Tensor,
                     image2: torch.Tensor) -> float:
        """
        计算LPIPS分数
        
        Args:
            image1: 第一组图像
            image2: 第二组图像
            
        Returns:
            LPIPS分数
        """
        if self.lpips is not None:
            lpips_score = self.lpips(image1, image2).mean().item()
            return lpips_score
        else:
            return -1.0  # 返回错误值表示LPIPS不可用
    
    def compute_ssim(self,
                    image1: torch.Tensor,
                    image2: torch.Tensor) -> float:
        """
        计算SSIM分数
        
        Args:
            image1: 第一组图像 [B, C, H, W]
            image2: 第二组图像 [B, C, H, W]
            
        Returns:
            平均SSIM分数
        """
        # 转换为numpy并计算SSIM
        image1_np = ((image1 * 0.5 + 0.5) * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        image2_np = ((image2 * 0.5 + 0.5) * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        
        ssim_scores = []
        for i in range(image1_np.shape[0]):
            img1 = image1_np[i].transpose(1, 2, 0)  # CHW -> HWC
            img2 = image2_np[i].transpose(1, 2, 0)
            
            # 转换为灰度图像计算SSIM
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
            score = ssim(img1_gray, img2_gray)
            ssim_scores.append(score)
        
        return np.mean(ssim_scores)
    
    @torch.no_grad()
    def compute_all_metrics(self,
                          original_images: torch.Tensor,
                          generated_images: torch.Tensor) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            original_images: 原始图像
            generated_images: 生成图像
            
        Returns:
            指标字典
        """
        metrics = {}
        
        try:
            metrics['fid'] = self.compute_fid(original_images, generated_images)
        except Exception as e:
            print(f"FID计算失败: {e}")
            metrics['fid'] = -1.0
        
        if self.lpips is not None:
            try:
                metrics['lpips'] = self.compute_lpips(original_images, generated_images)
            except Exception as e:
                print(f"LPIPS计算失败: {e}")
                metrics['lpips'] = -1.0
        else:
            metrics['lpips'] = -1.0
        
        try:
            metrics['ssim'] = self.compute_ssim(original_images, generated_images)
        except Exception as e:
            print(f"SSIM计算失败: {e}")
            metrics['ssim'] = -1.0
        
        return metrics

def save_results_grid(original_images: List[torch.Tensor],
                     generated_images: List[torch.Tensor],
                     save_path: str,
                     nrow: int = 4):
    """
    保存结果对比网格图
    
    Args:
        original_images: 原始图像列表
        generated_images: 生成图像列表
        save_path: 保存路径
        nrow: 每行图像数量
    """
    from torchvision.utils import make_grid
    
    # 限制图像数量
    max_images = min(len(original_images), len(generated_images), 16)
    original_images = original_images[:max_images]
    generated_images = generated_images[:max_images]
    
    # 创建对比网格
    all_images = []
    for orig, gen in zip(original_images, generated_images):
        all_images.extend([orig, gen])
    
    # 创建网格
    grid = make_grid(all_images, nrow=nrow*2, padding=2, normalize=True)
    
    # 保存
    grid_pil = ImageProcessor.tensor_to_pil(grid)
    grid_pil.save(save_path)

def compute_metrics(original_images: torch.Tensor,
                   generated_images: torch.Tensor,
                   device: str = "cuda",
                   training: bool = False) -> Dict[str, float]:
    """
    便捷函数：计算评估指标（默认仅评估阶段调用）。
    training=True 时，直接返回空指标，避免训练阶段CPU/IO开销。
    """
    if training:
        return { 'fid': -1.0, 'lpips': -1.0, 'ssim': -1.0 }
    calculator = MetricsCalculator(device=device)
    return calculator.compute_all_metrics(original_images, generated_images)

def visualize_detection_results(images: List[torch.Tensor],
                              detection_results: Dict,
                              save_path: str):
    """
    可视化检测结果
    
    Args:
        images: 图像列表
        detection_results: 检测结果
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (img_tensor, detected) in enumerate(zip(images[:8], detection_results['detected'][:8])):
        if i >= 8:
            break
            
        # 转换图像
        img = ImageProcessor.tensor_to_pil(img_tensor)
        
        # 显示图像
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # 添加检测结果标题
        color = 'red' if detected else 'green'
        status = '检测到' if detected else '未检测到'
        axes[i].set_title(f'{status}', color=color, fontsize=12)
    
    # 隐藏多余的子图
    for i in range(len(images), 8):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_training_config(config_dict: Dict, save_path: str):
    """
    保存训练配置
    
    Args:
        config_dict: 配置字典
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

def load_training_config(config_path: str) -> Dict:
    """
    加载训练配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class ProgressLogger:
    """进度日志记录器"""
    
    def __init__(self, log_file: str):
        """
        初始化日志记录器
        
        Args:
            log_file: 日志文件路径
        """
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """
        记录日志
        
        Args:
            message: 日志消息
            level: 日志级别
        """
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {level}: {message}\n"
        
        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line)
        
        # 打印到控制台
        print(log_line.strip())

if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数...")
    
    # 测试图像处理
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建测试图像
    test_image = torch.randn(3, 256, 256)
    test_mask = torch.rand(1, 256, 256)
    
    # 测试转换
    pil_image = ImageProcessor.tensor_to_pil(test_image)
    print(f"PIL图像尺寸: {pil_image.size}")
    
    # 测试掩码应用
    masked_image = ImageProcessor.apply_mask_to_image(test_image, test_mask)
    print(f"掩码图像形状: {masked_image.shape}")
    
    # 测试指标计算
    calculator = MetricsCalculator(device=device)
    
    original_batch = torch.randn(4, 3, 256, 256).to(device)
    generated_batch = torch.randn(4, 3, 256, 256).to(device)
    
    metrics = calculator.compute_all_metrics(original_batch, generated_batch)
    print(f"评估指标: {metrics}")
    
    print("工具函数测试完成！")
