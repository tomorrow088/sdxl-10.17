"""
配置文件 - 迷彩对抗生成系统
"""
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class ModelConfig:
    """模型配置"""
    # SDXL 生成器配置
    sdxl_model_path: str = "models/stable-diffusion-xl-base-1.0"
    clip_model_path: str = "models/laion-CLIP-ViT-H-14-laion2B-s32B-b79K"
    
    # 检测器模型配置
    yolo_model: str = "yolov8x.pt"  # 将自动下载
    efficientnet_model: str = "efficientnet_b4"  # 恢复使用本地EfficientNet-B4
    clip_model: str = "ViT-L/14"
 
    
    # 图像尺寸 - 进一步减小避免显存问题
    image_size: Tuple[int, int] = (512, 512)  # 大幅减小图像尺寸
    mask_size: Tuple[int, int] = (512, 512)
    
    # 生成参数
    num_inference_steps: int = 8        # 推理步数（验证阶段加速）
    guidance_scale: float = 6     # CFG引导强度
    
@dataclass  
class TrainingConfig:
    """训练配置"""
    # 数据路径
    train_data_dir: str = "data/train"   # 训练目录
    test_data_dir: str = "data/test"     # 测试目录
    mask_train_dir: str = "data/mask_train"
    mask_test_dir: str = "data/mask_test"
    output_dir: str = "outputs"
    reference_patterns_dir: str = "data/reference_patterns"  # 假设你的参考图放在这个目录

    # 新增一个开关来控制是否使用参考迷彩系统
    adapter_use_reference: bool = True
    lora_rank: int = 8
    # 训练参数
    batch_size: int = 2 # 快速训练模式下可以使用更大的batch_size
    micro_batch_size: int = 1 # 微批尺寸（用于累积梯度）
    # 适配器注入强度与调度
    adapter_weight: float = 1.0
    adapter_cond_tau: float = 0.8  # 残差在前80%步数生效
    img2img_strength: float = 0.0  # >0 使用原图latent作为起点(0~1)
    learning_rate: float = 1e-4
    num_epochs: int = 50
    save_every: int = 50
    # 训练/验证的扩散步数（训练低步数可显著提速）
    train_num_steps: int = 10  # 进一步减少训练步数
    eval_num_steps: int = 20   # 减少验证步数
    # 日志与性能
    log_interval: int = 100
    use_torch_compile: bool = True  # 启用torch.compile加速
    # 训练中每多少步评估一次检测率（减少额外检测前向）
    eval_interval_steps: int = 50
    
    # 对抗训练参数
    generator_steps: int = 1  # 生成器训练步数
    discriminator_steps: int = 3  # 判别器训练步数
    adversarial_weight: float = 0.1  # 对抗损失权重（默认值，若使用调度则会被覆盖）
    content_weight: float = 1.0  # 内容保持损失权重

    # 对抗权重调度（可选）
    use_adversarial_schedule: bool = True         # 是否使用对抗权重调度
    adversarial_weight_start: float = 0.1        # 开始权重
    adversarial_weight_end: float = 0.3          # 结束权重
    adversarial_warmup_epochs: int = 10           # 预热轮数

    # 多检测器分支权重（以YOLO为主）
    yolo_branch_weight: float = 0.6
    clip_branch_weight: float = 0.2
    efficientnet_branch_weight: float = 0.2

    # 对抗损失中各可微分分支权重（用于compute_adversarial_loss融合）
    yolo_adv_weight: float = 0.5        # YOLO置信度损失权重
    clip_adv_weight: float = 1.0        # 强化CLIP语义损失
    efficientnet_adv_weight: float = 0.2  # 辅助分类损失

    # (重新启用) 掩码区域一致性与风格损失
    balanced_color_weight: float = 0.08         # 颜色一致性权重
    balanced_brightness_weight: float = 0.04    # 亮度一致性权重
    balanced_tv_weight: float = 0.002           # 总变差权重 (轻微平滑)
    balanced_hue_forbid_weight: float = 0.003   # 色相禁止权重
    
    # 扩散训练（LoRA真正在UNet上生效）
    use_diffusion_training: bool = True
    diffusion_timestep_sampling: str = "uniform"  # uniform / log

    # 优化器参数
    generator_lr: float = 1e-5       # 生成器学习率
    beta1: float = 0.9                # Adam优化器参数
    beta2: float = 0.999               # Adam优化器参数

    # 色相禁止区间与权重（仅抑制极端颜色，允许自然色调）
    forbid_hue_ranges: Tuple[Tuple[float, float], ...] = (
        (0.8, 0.95),  # 禁止品红色、紫色等非自然色
    )
    
    # 调试：是否启用梯度异常检测
    detect_anomaly: bool = False

    # 单一平衡权重配置
    balanced_content_weight: float = 1.0        # 内容保持权重
    balanced_adversarial_weight: float = 0.3    # 对抗权重
    balanced_color_weight: float = 0.08         # 颜色一致性权重
    balanced_brightness_weight: float = 0.04    # 亮度一致性权重
    balanced_tv_weight: float = 0.002           # 总变差权重
    balanced_diffusion_weight: float = 0.6      # 扩散权重
    balanced_hue_forbid_weight: float = 0.003   # 色相禁止权重

    def get_balanced_weights(self) -> dict:
        """获取平衡的损失权重"""
        return {
            'content': self.balanced_content_weight,
            'adv': self.balanced_adversarial_weight,
            'color': self.balanced_color_weight,
            'bright': self.balanced_brightness_weight,
            'tv': self.balanced_tv_weight,
            'diff': self.balanced_diffusion_weight,
            'hist': 1.0,  # (重新启用) 背景HSV直方图损失权重
            'hue': self.balanced_hue_forbid_weight,
        }
    
@dataclass
class EvaluationConfig:
    """评估配置"""
    # 检测阈值
    detection_confidence: float = 0.5
    detection_iou_threshold: float = 0.45
    
    # 评估指标
    calculate_fid: bool = True
    calculate_lpips: bool = True
    calculate_ssim: bool = True
    
    # 测试数据
    test_split: float = 0.2
    num_test_samples: int = 100

# 全局配置实例
model_config = ModelConfig()
training_config = TrainingConfig()
eval_config = EvaluationConfig()

# 负面提示词库 (已整合)
# 由 generate_adaptive_camouflage_prompt 动态调用
NEGATIVE_PROMPTS_CONFIG = {
    'base': [
        "organic shapes, smooth blending, curved lines, natural patterns, blotches, soft edges",
        "blurry, low quality, distorted, artifacts, watermark",
        "text, letters, signature, logo, cartoon, anime",
        "oversaturated, too dark, unrealistic colors",
        "artificial patterns, synthetic colors, unnatural appearance",
        "obvious camouflage, repetitive patterns, geometric shapes",
        "bright colors, saturated colors, contrasting colors",
        "low quality, blurry, distorted, artifacts"
    ],
    'color_strict': [

    ]
}
