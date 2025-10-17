"""
环境分析器模块 - 基于多尺度特征提取的环境和纹理分析
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
# from transformers import CLIPModel, CLIPProcessor  <- 暂时禁用CLIP，因为它与MobileNetV2功能重叠且增加复杂性

class DifferentiableHistogram(nn.Module):
    """可微分的RGB颜色直方图，用于更精确的颜色分析"""
    def __init__(self, bins=8, sigma=0.02):
        super().__init__()
        self.bins = bins
        self.sigma = sigma
        bin_centers = torch.linspace(0, 1, bins)
        grid = torch.stack(torch.meshgrid(bin_centers, bin_centers, bin_centers, indexing='ij'), dim=-1)
        self.register_buffer('bin_centers', grid.view(-1, 3))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, -1, 3)
        diff = x.unsqueeze(2) - self.bin_centers.unsqueeze(0).unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=-1)
        weights = torch.exp(-dist_sq / (2 * self.sigma**2))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        histogram = weights.sum(dim=1)
        histogram = histogram / (histogram.sum(dim=-1, keepdim=True) + 1e-8)
        return histogram


class UpgradedColorAnalyzer(nn.Module):
    """升级版颜色分析器，使用可微分直方图"""
    def __init__(self, color_bins=8):
        super().__init__()
        self.histogram_generator = DifferentiableHistogram(bins=color_bins)
        # 颜色特征维度就是直方图的维度
        self.color_feature_dim = color_bins**3

    def forward(self, x):
        """
        x: [B, 3, H, W] in range [0,1]
        """
        # 为了加速，降低计算直方图的图像分辨率
        x_small = F.interpolate(x, size=(64, 64), mode='area')
        color_histogram = self.histogram_generator(x_small)
        # 在此版本中，直方图本身就是颜色特征
        return color_histogram, color_histogram, torch.tensor(0) # 返回兼容的元组


class UpgradedTextureAnalyzer(nn.Module):
    """升级版纹理分析器（使用MobileNet特征）"""
    def __init__(self, in_features, num_texture_types=8):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_texture_types)
        )

    def forward(self, backbone_features):
        logits = self.classifier(backbone_features)
        probs = F.softmax(logits, dim=1)
        # 返回特征本身和分类概率
        return backbone_features, probs

class UpgradedEnvDistributionAnalyzer(nn.Module):
    """升级版环境分布分析器（使用MobileNet特征）"""
    def __init__(self, in_features, num_env_types=7):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_env_types)
        )

    def forward(self, backbone_features):
        logits = self.classifier(backbone_features)
        return F.softmax(logits, dim=1)

class UpgradedFeatureFusion(nn.Module):
    """升级版特征融合模块"""
    def __init__(self, env_dim=7, texture_dim=8, color_dim=8**3, output_dim=512):
        super().__init__()
        # 直接融合概率分布和直方图
        total_input_dim = env_dim + texture_dim + color_dim
        self.fusion_network = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
        self.guidance_generator = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, env_dist, texture_dist, color_hist):
        combined = torch.cat([env_dist, texture_dist, color_hist], dim=1)
        fused_features = self.fusion_network(combined)
        guidance = self.guidance_generator(fused_features)
        return fused_features, guidance


class IntegratedEnvironmentAnalyzer(nn.Module):
    """集成环境分析器 - 升级版"""
    def __init__(self, device='cuda', use_clip=False): # use_clip 默认为False
        super().__init__()
        # 1. 主干网络: MobileNetV2
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.feature_extractor = mobilenet.features
        # 冻结参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        backbone_out_dim = 1280

        # 2. 升级后的分析模块
        self.env_analyzer = UpgradedEnvDistributionAnalyzer(in_features=backbone_out_dim)
        self.texture_analyzer = UpgradedTextureAnalyzer(in_features=backbone_out_dim)
        self.color_analyzer = UpgradedColorAnalyzer(color_bins=8)

        # 3. 升级后的融合模块
        self.feature_fusion = UpgradedFeatureFusion(color_dim=8**3)

    def forward(self, x, mask=None):
        x_normalized = (x + 1.0) / 2.0

        if mask is not None:
            if mask.dim() == 3: mask = mask.unsqueeze(1)
            background_mask = 1.0 - mask.float().clamp(0, 1)
            x_region = x_normalized * background_mask
        else:
            x_region = x_normalized

        # 1. 特征提取
        backbone_features = self.feature_extractor(x_region)

        # 2. 各个分析分支
        env_distribution = self.env_analyzer(backbone_features)
        texture_features, texture_pattern = self.texture_analyzer(backbone_features)
        color_features, color_histogram, color_pattern = self.color_analyzer(x_region)

        # 3. 特征融合
        fused_features, guidance_vector = self.feature_fusion(
            env_distribution, texture_pattern, color_histogram
        )

        return {
            'environment_distribution': env_distribution,
            'texture_pattern': texture_pattern,
            'color_histogram': color_histogram,
            'guidance_vector': guidance_vector,
            # 添加一些兼容性的返回项
            'texture_features': texture_features,
            'color_features': color_features,
            'color_pattern': color_pattern,
            'fused_features': fused_features,
            'clip_analysis': None
        }

# 辅助函数
def create_environment_analyzer(device='cuda', use_clip=False):
    """创建环境分析器实例"""
    analyzer = IntegratedEnvironmentAnalyzer(device=device, use_clip=use_clip)
    return analyzer.to(device)
