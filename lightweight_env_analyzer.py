"""
轻量级环境分析器 - 优化用于实时训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class DifferentiableHistogram(nn.Module):
    """可微分的RGB颜色直方图"""
    def __init__(self, bins=16, sigma=0.01):
        super().__init__()
        self.bins = bins
        self.sigma = sigma

        # 创建固定的、均匀分布的颜色箱中心点
        # [bins, bins, bins, 3] -> [bins^3, 3]
        bin_centers = torch.linspace(0, 1, bins)
        grid = torch.stack(torch.meshgrid(bin_centers, bin_centers, bin_centers, indexing='ij'), dim=-1)
        self.register_buffer('bin_centers', grid.view(-1, 3))

    def forward(self, x):
        """
        x: [B, 3, H, W] in range [0, 1]
        """
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, -1, 3) # [B, H*W, 3]

        # 计算每个像素与所有颜色箱中心的距离
        # x: [B, N, 1, 3], centers: [1, 1, C, 3] -> dist: [B, N, C]
        # (N=H*W, C=bins^3)
        diff = x.unsqueeze(2) - self.bin_centers.unsqueeze(0).unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=-1) # L2距离的平方

        # 使用径向基函数（高斯核）进行软分配
        # 权重与距离成反比
        weights = torch.exp(-dist_sq / (2 * self.sigma**2))

        # 归一化权重，使每个像素的权重和为1
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 对所有像素的权重求和，得到直方图
        histogram = weights.sum(dim=1)

        # 再次归一化，使整个直方图的和为1
        histogram = histogram / (histogram.sum(dim=-1, keepdim=True) + 1e-8)

        return histogram # [B, bins^3]

class EnhancedEnvAnalyzer(nn.Module):
    """
    增强版环境分析器
    使用MobileNetV2作为主干网络，并增强颜色分析
    """
    def __init__(self,
                 env_types=7,
                 texture_types=8,
                 color_bins=4, # 4x4x4=64个颜色箱
                 guidance_dim=256):
        super().__init__()

        # 1. 主干网络：MobileNetV2
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = mobilenet.features
        # 冻结主干网络参数
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # MobileNetV2特征输出维度
        backbone_out_dim = 1280

        # 2. 颜色分析分支
        self.color_analyzer = DifferentiableHistogram(bins=color_bins)
        color_feature_dim = color_bins**3

        # 3. 顶层分析头
        self.env_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_out_dim, env_types)
        )
        self.texture_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_out_dim, texture_types)
        )

        # 4. 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(env_types + texture_types + color_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, guidance_dim),
            nn.Tanh()
        )

    def forward(self, x, mask=None):
        # 确保输入在[0,1]
        x = x.clamp(0, 1)

        if mask is not None:
            background_mask = 1.0 - mask.float().clamp(0, 1)
            # MobileNetV2期望输入是标准化的，但这里分析的是颜色和纹理
            # 保持在[0,1]范围，仅对背景区域进行分析
            x_bg = x * background_mask
        else:
            x_bg = x

        # 主干网络提取高级特征
        # MobileNetV2需要3通道输入
        features = self.backbone(x_bg)

        # 分析头
        env_logits = self.env_head(features)
        texture_logits = self.texture_head(features)

        # 颜色分析
        # 降低颜色分析的图像分辨率以加速
        x_color = F.interpolate(x_bg, size=(64, 64), mode='area')
        color_hist = self.color_analyzer(x_color)

        # 转换为概率
        env_probs = F.softmax(env_logits, dim=1)
        texture_probs = F.softmax(texture_logits, dim=1)

        # 融合
        combined = torch.cat([env_probs, texture_probs, color_hist], dim=1)
        guidance_vector = self.fusion(combined)

        return {
            'env_logits': env_logits,
            'texture_logits': texture_logits,
            'color_histogram': color_hist,
            'guidance_vector': guidance_vector
        }

class LightweightEnvAnalyzer(nn.Module):
    """
    轻量级环境分析器
    专注于快速提取关键特征用于适配器条件
    """

    def __init__(self,
                 in_channels=3,
                 env_types=7,
                 texture_types=8,
                 color_bins=32,
                 feature_dim=256):
        super().__init__()

        # 共享的特征提取器（轻量级）
        self.shared_encoder = nn.Sequential(
            # 第一层：快速下采样
            nn.Conv2d(in_channels, 32, 7, 2, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 第二层
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 第三层
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 特征层
            nn.Conv2d(128, feature_dim, 3, 1, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )

        # 环境分类头
        self.env_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, env_types)
        )

        # 纹理分类头
        self.texture_head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 1),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 16, texture_types)
        )

        # 颜色直方图头（可微分）
        self.color_head = nn.Sequential(
            nn.Conv2d(feature_dim, color_bins, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(env_types + texture_types + color_bins, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.Tanh()  # 归一化输出
        )

    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: [B, 3, H, W] 输入图像（范围[0,1]）
            mask: [B, 1, H, W] 可选掩码
        Returns:
            分析结果字典
        """
        # 应用掩码 - 分析背景而非掩码区域
        if mask is not None:
            # 反转掩码，分析背景区域
            background_mask = 1.0 - mask.float().clamp(0, 1)
            x = x * background_mask

        # 共享特征提取
        features = self.shared_encoder(x)

        # 三个分支的预测
        env_logits = self.env_head(features)
        texture_logits = self.texture_head(features)
        color_hist = self.color_head(features)

        # 获取概率分布
        env_probs = F.softmax(env_logits, dim=1)
        texture_probs = F.softmax(texture_logits, dim=1)

        # 融合所有信息
        combined = torch.cat([env_probs, texture_probs, color_hist], dim=1)
        guidance_vector = self.fusion(combined)

        return {
            'features': features,
            'env_logits': env_logits,
            'env_probs': env_probs,
            'texture_logits': texture_logits,
            'texture_probs': texture_probs,
            'color_histogram': color_hist,
            'guidance_vector': guidance_vector
        }

    def get_top_predictions(self, results):
        """获取最可能的预测结果"""
        env_type = torch.argmax(results['env_probs'], dim=1)
        texture_type = torch.argmax(results['texture_probs'], dim=1)

        return {
            'env_type': env_type,
            'texture_type': texture_type,
            'guidance': results['guidance_vector']
        }


class EnvAwareAdapter(nn.Module):
    """
    环境感知适配器
    将环境分析集成到适配器中
    """

    def __init__(self,
                 base_adapter,
                 use_env_analysis=True,
                 env_weight=0.1):
        super().__init__()

        self.base_adapter = base_adapter
        self.use_env_analysis = use_env_analysis
        self.env_weight = env_weight

        if use_env_analysis:
            # 轻量级分析器
            self.env_analyzer = EnhancedEnvAnalyzer(guidance_dim=256,
        color_bins=4 )
            
            # 条件注入层（对应SDXL的4个尺度）
            channels = [320, 640, 1280, 1280]
            self.condition_injectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(256, ch),
                    nn.ReLU(),
                    nn.Linear(ch, ch)
                ) for ch in channels
            ])
            
    def forward(self, x, text_embeds=None, time_embeds=None):
        """
        前向传播
        Args:
            x: [B, C, H, W] 输入（图像+掩码）
            text_embeds: 文本嵌入
            time_embeds: 时间嵌入
        """
        # 基础适配器特征
        base_features = self.base_adapter(x, text_embeds, time_embeds)
        
        if not self.use_env_analysis:
            return base_features
        
        # 分离图像和掩码
        image = x[:, :3, :, :]
        mask = x[:, 3:4, :, :] if x.shape[1] > 3 else None
        
        # 环境分析（梯度传播），分析背景（掩码外）
        # 注意：这里的image已经是[0,1]范围
        if mask is not None:
            # 传递掩码给环境分析器，让它正确处理
            env_results = self.env_analyzer(image, mask)
        else:
            env_results = self.env_analyzer(image, None)
        guidance = env_results['guidance_vector']
        
        # 注入环境条件
        enhanced_features = []
        for i, feat in enumerate(base_features):
            # 投影指导向量到对应通道
            cond = self.condition_injectors[i](guidance)
            cond = cond.unsqueeze(-1).unsqueeze(-1)
            
            # 加权融合
            enhanced_feat = feat + self.env_weight * cond
            enhanced_features.append(enhanced_feat)
        
        return enhanced_features


def create_env_aware_adapter(base_adapter_type='base', 
                           use_env_analysis=True,
                           env_weight=0.1):
    """
    创建环境感知适配器
    
    Args:
        base_adapter_type: 基础适配器类型
        use_env_analysis: 是否启用环境分析
        env_weight: 环境条件的权重
    """
    from sdxl_advanced_adapter import create_sdxl_adapter
    
    # 创建基础适配器
    base_adapter = create_sdxl_adapter(adapter_type=base_adapter_type)
    
    # 包装成环境感知适配器
    env_adapter = EnvAwareAdapter(
        base_adapter=base_adapter,
        use_env_analysis=use_env_analysis,
        env_weight=env_weight
    )
    
    return env_adapter


# 使用示例
if __name__ == "__main__":
    # 创建环境感知适配器
    adapter = create_env_aware_adapter(
        base_adapter_type='base',
        use_env_analysis=True,
        env_weight=0.15
    )
    
    # 测试
    x = torch.randn(2, 4, 512, 512)  # [B, RGB+mask, H, W]
    text_embeds = torch.randn(2, 2048)
    time_embeds = torch.randn(2, 6)
    
    with torch.no_grad():
        features = adapter(x, text_embeds, time_embeds)
        
    print("环境感知适配器输出：")
    for i, feat in enumerate(features):
        print(f"特征 {i}: {feat.shape}")
