"""
SDXL高级适配器模块 - 基于参考代码但适配SDXL架构
注意：SDXL和SD1.5的主要区别在于通道数和层数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from environment_analyzer import IntegratedEnvironmentAnalyzer, create_environment_analyzer
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple
from lightweight_env_analyzer import EnhancedEnvAnalyzer

class Downsample(nn.Module):
    """下采样层"""
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(
                self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):
    """ResNet块 - 避免原地操作"""
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()  # 不使用inplace
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class SDXLAdapter(nn.Module):
    """
    SDXL适配器 - 基于参考代码但适配SDXL架构
    SDXL UNet通道: [320, 640, 1280, 1280] (与SD1.5相同)
    但SDXL有更多的transformer层和条件机制
    """

    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=256, ksize=3, sk=False, use_conv=True,
                 text_dim=2048, time_dim=6, env_guidance_dim=256, color_guidance_dim=64):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels, self.nums_rb = channels, nums_rb
        self.body = nn.ModuleList()
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

        for i in range(len(channels)):
            for j in range(nums_rb):
                is_down = (i != 0) and (j == 0)
                in_c = channels[i - 1] if is_down else channels[i]
                self.body.append(ResnetBlock(in_c, channels[i], is_down, ksize, sk, use_conv))

        # --- 统一的条件投影层，现在全部位于基础适配器中 ---
        self.text_projectors = nn.ModuleList(
            [nn.Sequential(nn.Linear(text_dim, ch), nn.SiLU(), nn.Linear(ch, ch)) for ch in channels])
        self.time_projectors = nn.ModuleList(
            [nn.Sequential(nn.Linear(time_dim, ch), nn.SiLU(), nn.Linear(ch, ch)) for ch in channels])
        self.env_guidance_projectors = nn.ModuleList(
            [nn.Sequential(nn.Linear(env_guidance_dim, ch), nn.ReLU(), nn.Linear(ch, ch)) for ch in channels])
        self.color_injectors = nn.ModuleList(
            [nn.Sequential(nn.Linear(color_guidance_dim, ch), nn.ReLU(), nn.Linear(ch, ch)) for ch in channels])



    def forward(self, x, text_embeds=None, time_embeds=None, env_guidance=None, color_guidance=None):
        x = self.unshuffle(x)
        features = []
        x = self.conv_in(x)

        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)

            # --- 真正的“过程引导”：在每一步特征提取后，注入所有引导条件 ---
            current_dtype = x.dtype

            if text_embeds is not None:
                cond = self.text_projectors[i](text_embeds.to(current_dtype)).unsqueeze(-1).unsqueeze(-1)
                x = x + cond

            if time_embeds is not None:
                cond = self.time_projectors[i](time_embeds.to(current_dtype)).unsqueeze(-1).unsqueeze(-1)
                x = x + cond

            if env_guidance is not None:
                cond = self.env_guidance_projectors[i](env_guidance.to(current_dtype)).unsqueeze(-1).unsqueeze(-1)
                x = x + 0.15 * cond

            if color_guidance is not None:
                cond = self.color_injectors[i](color_guidance.to(current_dtype)).unsqueeze(-1).unsqueeze(-1)
                x = x + 0.25 * cond

            features.append(x)
        return features


class SDXLAdapterWithReference(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin_base=4, cin_ref=3, ksize=3, sk=False,
                 use_conv=True):
        super().__init__()
        # 两个适配器现在都是增强版，都能处理所有引导
        self.main_adapter = SDXLAdapter(channels, nums_rb, cin_base * 64, ksize, sk, use_conv)
        self.ref_adapter = SDXLAdapter(channels, nums_rb // 2, cin_ref * 64, ksize, sk, use_conv)
        self.fusion_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(ch * 2, ch, 1), nn.ReLU(), nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU()) for ch in
             channels])

    def forward(self, x, ref=None, text_embeds=None, time_embeds=None, env_guidance=None, color_guidance=None):
        # --- 将所有引导条件“透传”给 main_adapter ---
        main_features = self.main_adapter(x, text_embeds, time_embeds, env_guidance, color_guidance)

        if ref is not None:
            # --- 将所有引导条件也“透传”给 ref_adapter ---
            ref_features = self.ref_adapter(ref, text_embeds, time_embeds, env_guidance, color_guidance)
            fused = []
            for i, (main, r) in enumerate(zip(main_features, ref_features)):
                r_resized = F.interpolate(r, size=main.shape[-2:], mode='bilinear', align_corners=False)
                fused.append(self.fusion_layers[i](torch.cat([main, r_resized], dim=1)))
            return fused
        return main_features


class ResnetBlock_light(nn.Module):
    """轻量级ResNet块"""
    def __init__(self, in_c):
        super().__init__()
        self.block1 = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(in_c, in_c, 3, 1, 1)

    def forward(self, x):
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        return h + x


class SDXLAdapter_light(nn.Module):
    """
    轻量级SDXL适配器 - 用于快速推理
    减少参数量但保持多尺度结构
    """
    def __init__(self, 
                 channels=[320, 640, 1280, 1280],
                 nums_rb=3,
                 cin=256):
        super(SDXLAdapter_light, self).__init__()
        
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        
        for i in range(len(channels)):
            if i == 0:
                self.body.append(
                    self._make_extractor(cin, channels[i]//4, channels[i], nums_rb, down=False)
                )
            else:
                self.body.append(
                    self._make_extractor(channels[i-1], channels[i]//4, channels[i], nums_rb, down=True)
                )
        
        self.body = nn.ModuleList(self.body)

    def _make_extractor(self, in_c, inter_c, out_c, nums_rb, down=False):
        """创建特征提取器"""
        layers = []
        
        if down:
            layers.append(nn.AvgPool2d(2, 2))
        
        layers.append(nn.Conv2d(in_c, inter_c, 1, 1, 0))
        
        for _ in range(nums_rb):
            layers.append(ResnetBlock_light(inter_c))
        
        layers.append(nn.Conv2d(inter_c, out_c, 1, 1, 0))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.unshuffle(x)
        features = []
        
        for i in range(len(self.channels)):
            x = self.body[i](x)
            features.append(x)
        
        return features


class SDXLSmartAdapter(nn.Module):
    """
    智能SDXL适配器 - 集成环境分析的自适应生成
    """
    def __init__(self, 
                 channels=[320, 640, 1280, 1280],        # SDXL通道数
                 nums_rb=3,                                 # 每个分辨率的ResNet块数
                 cin=256,                                   # 输入通道数（PixelUnshuffle后，RGB+mask=4→4*8*8=256）
                 use_env_analyzer=True,                     # 是否使用环境分析器
                 device='cuda'):
        super().__init__()
        
        # 基础适配器
        self.base_adapter = SDXLAdapter(channels, nums_rb, cin)
        
        # 环境分析器
        self.use_env_analyzer = use_env_analyzer
        if use_env_analyzer:
            self.env_analyzer = EnhancedEnvAnalyzer()
            
            # 环境指导投影层
            self.env_guidance_projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, ch),  # 从guidance_vector投影到各层
                    nn.ReLU(),
                    nn.Linear(ch, ch)
                ) for ch in channels
            ])
            
            # 颜色直方图注入层
            self.color_injectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, ch),  # 从color_histogram投影，维度从64修正为512 (8*8*8)
                    nn.ReLU(),
                    nn.Linear(ch, ch)
                ) for ch in channels
            ])

    def forward(self, x, text_embeds=None, time_embeds=None, analyze_env=True):
        """
        智能前向传播
        Args:
            x: [B, C, H, W] 输入（图像+掩码）
            text_embeds: 文本嵌入
            time_embeds: 时间嵌入
            analyze_env: 是否进行环境分析
        """
        # 分离图像和掩码
        if x.shape[1] >= 4:
            image = x[:, :3, :, :]
            mask = x[:, 3:4, :, :]
        else:
            image = x
            mask = None

        # 环境分析（如果启用）
        env_guidance = None
        color_guidance = None

        if self.use_env_analyzer and analyze_env:
            # 允许梯度传播，让环境分析影响训练
            # 分析器期望的输入范围是[0,1]，此处的输入 'image' 已经是[0,1]
            image_for_analysis = image
            # 重要：传递掩码给环境分析器，让它分析背景而非掩码区域
            analysis = self.env_analyzer(image_for_analysis, mask)
            
            # 保留梯度，允许颜色指导影响训练
            env_guidance = analysis['guidance_vector']
            color_guidance = analysis['color_histogram']
        
        # 基础适配器前向传播
        features = self.base_adapter(x, text_embeds, time_embeds)
        
        # 注入环境和颜色指导（如果有）
        if env_guidance is not None and color_guidance is not None:
            enhanced_features = []
            for i, feat in enumerate(features):
                # 投影环境指导
                env_cond = self.env_guidance_projectors[i](env_guidance)
                env_cond = env_cond.unsqueeze(-1).unsqueeze(-1)
                
                # 投影颜色指导
                color_cond = self.color_injectors[i](color_guidance)
                color_cond = color_cond.unsqueeze(-1).unsqueeze(-1)
                
                # 应用条件，增加颜色指导权重
                feat = feat + 0.15 * env_cond + 0.25 * color_cond
                enhanced_features.append(feat)
            
            return enhanced_features
        else:
            return features


class SDXLSmartReferenceAdapter(nn.Module):
    """
    智能参考适配器 - 最终版
    职责: 1. 分析环境; 2. 将【所有】引导信息传递给下一层
    """
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin_base=4, cin_ref=3):
        super().__init__()
        self.env_analyzer = EnhancedEnvAnalyzer(guidance_dim=256, color_bins=4)
        self.ref_adapter = SDXLAdapterWithReference(channels, nums_rb, cin_base, cin_ref)

    def forward(self, x, ref=None, text_embeds=None, time_embeds=None):
        image_01 = x[:, :3, :, :]
        mask = x[:, 3:4, :, :] if x.shape[1] >= 4 else None

        # 1. **先**进行环境分析
        analysis = self.env_analyzer(image_01, mask)
        env_guidance = analysis['guidance_vector']
        color_guidance = analysis['color_histogram']

        # 2. **后**调用下一层适配器，并传入【所有】引导信息
        #    不再进行任何后期修正，所有引导都在底层完成
        features = self.ref_adapter(
            x, ref, text_embeds, time_embeds,
            env_guidance=env_guidance,
            color_guidance=color_guidance
        )
        return features


# --- 最终版工厂函数 ---
def create_sdxl_adapter(smart=False, use_reference=False):
    """
    创建SDXL适配器的工厂函数 - 最终简化版
    只要需要任何高级功能，就直接创建最强大的“智能参考适配器”。
    """
    # --- 核心修改：将多个 elif 合并为一个 if ---
    if smart or use_reference:
        # 只要 smart 或 use_reference 中任何一个为 True...

        # 打印具体创建信息，方便调试
        if smart and use_reference:
            print("✅ 创建【智能参考适配器】 (环境分析 + 参考图)")
        elif smart:
            print("✅ 创建【智能适配器】 (仅环境分析模式，使用终极适配器架构)")
        else:  # use_reference is True
            print("✅ 创建【标准参考适配器】 (使用终极适配器架构)")

        # ...都返回最强大的 SDXLSmartReferenceAdapter 实例。
        # 这个实例内部已经包含了环境分析和参考图融合的所有逻辑。
        return SDXLSmartReferenceAdapter()

    else:
        # 只有当 smart 和 use_reference 都为 False 时，才创建最基础的适配器。
        print("✅ 创建【基础适配器】 (无额外功能)")
        return SDXLAdapter(cin=4 * 64)
