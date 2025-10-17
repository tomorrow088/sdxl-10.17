"""
SDXL迷彩生成器模块 - 精简版
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Dict
import os
import random
from torchvision import transforms
from diffusers import (
    StableDiffusionXLPipeline,        # 文本到图像
    StableDiffusionXLImg2ImgPipeline,        # 图像到图像
    StableDiffusionXLInpaintPipeline,            # 图像修复
    DDIMScheduler        # 环境感知调度器
)
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model

from config import model_config, training_config
from sdxl_advanced_adapter import create_sdxl_adapter



class SDXLCamouflageGenerator(nn.Module):
    """SDXL迷彩生成器 - 精简版"""
    
    def __init__(self, 
                 model_path: str = None,
                 device: str = "cuda",
                 use_lora: bool = True,
                 use_smart_adapter: bool = True):
        """
        初始化SDXL迷彩生成器
        
        Args:
            model_path: SDXL模型路径
            device: 计算设备
            use_lora: 是否使用LoRA训练
            use_smart_adapter: 是否使用智能适配器
        """
        super().__init__()
        
        self.device = device
        self.model_path = model_path or model_config.sdxl_model_path
        self.use_lora = use_lora
        self.use_smart_adapter = use_smart_adapter
        
        # 缓存优化：避免重复初始化
        self._trainable_params_cached = False
        self._cached_params = []
        
        # 加载管道
        self._load_pipelines()
        
        # 初始化CLIP（用于参考图像选择）
        self._init_clip()
        
        # 设置LoRA
        if use_lora:
            self._setup_lora()
        
        # 参考图案缓存
        self.reference_db = []
        self._load_reference_patterns()

    def set_training_mode(self, is_training: bool = True) -> None:
        """与训练器接口兼容的模式切换
        - 训练时：仅将可训练模块设为train（adapter、unet_lora）
        - 其余大模型（VAE、文本编码器）保持eval，避免不必要的dropout/BN扰动
        """
        # 本模块本身的nn.Module开关（不影响diffusers子模块）
        super().train(is_training)
        
        # 1) 适配器
        if hasattr(self, 'adapter') and self.adapter is not None:
            self.adapter.train(is_training)
        
        # 2) LoRA-UNet（训练用）
        if hasattr(self, 'unet_lora') and self.unet_lora is not None:
            self.unet_lora.train(is_training)
        
        # 3) 大模型组件保持eval，除非明确需要训练
        pipe = getattr(self, 'txt2img_pipe', None)
        if pipe is not None:
            # 文本编码器
            if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
                pipe.text_encoder.eval()
            if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
                pipe.text_encoder_2.eval()
            # VAE/UNet
            if hasattr(pipe, 'vae') and pipe.vae is not None:
                pipe.vae.eval()
            if hasattr(pipe, 'unet') and pipe.unet is not None:
                pipe.unet.eval()
        
        # 其他复用管道同样保持eval
        for aux_pipe_name in ['img2img_pipe', 'inpaint_pipe']:
            aux_pipe = getattr(self, aux_pipe_name, None)
            if aux_pipe is not None:
                if hasattr(aux_pipe, 'vae') and aux_pipe.vae is not None:
                    aux_pipe.vae.eval()
                if hasattr(aux_pipe, 'unet') and aux_pipe.unet is not None:
                    aux_pipe.unet.eval()
                if hasattr(aux_pipe, 'text_encoder') and aux_pipe.text_encoder is not None:
                    aux_pipe.text_encoder.eval()
                if hasattr(aux_pipe, 'text_encoder_2') and aux_pipe.text_encoder_2 is not None:
                    aux_pipe.text_encoder_2.eval()

    def _load_pipelines(self):
        """加载SDXL管道（精简内存占用版本）"""
        print(f"🔄 加载SDXL模型: {self.model_path}")
        
        # 仅加载一个主管线，避免重复占用显存
        # 尽量以半精度加载以节省显存
        load_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_path,
            torch_dtype=load_dtype,
            use_safetensors=True,
            variant="fp16" if load_dtype == torch.float16 else None
        )
        # 张量布局优化
        try:
            self.txt2img_pipe.unet.to(memory_format=torch.channels_last)
            self.txt2img_pipe.vae.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        
        # 开启省显存选项（如果可用）
        try:
            if hasattr(self.txt2img_pipe, 'enable_attention_slicing'):
                self.txt2img_pipe.enable_attention_slicing("max")
                print("✅ 已启用注意力切片")
            # xFormers内存优化（若可用）
            try:
                if hasattr(self.txt2img_pipe, 'enable_xformers_memory_efficient_attention'):
                    self.txt2img_pipe.enable_xformers_memory_efficient_attention()
                    print("✅ 已启用xFormers内存优化")
            except Exception:
                pass
            if hasattr(self.txt2img_pipe, 'vae') and hasattr(self.txt2img_pipe.vae, 'enable_tiling'):
                self.txt2img_pipe.vae.enable_tiling()
                print("✅ 已启用VAE平铺")
        except Exception:
            pass
        
        # 将主管线移动到设备
        self.txt2img_pipe = self.txt2img_pipe.to(self.device)

        # 兼容旧训练器：挂接常用子模块引用
        self.unet = self.txt2img_pipe.unet
        self.vae = self.txt2img_pipe.vae
        self.text_encoder = getattr(self.txt2img_pipe, 'text_encoder', None)
        self.text_encoder_2 = getattr(self.txt2img_pipe, 'text_encoder_2', None)

        # 梯度检查点已禁用以提升训练速度（牺牲部分显存）
        # try:
        #     if hasattr(self.unet, 'enable_gradient_checkpointing'):
        #         self.unet.enable_gradient_checkpointing()
        #         print("✅ 已启用UNet梯度检查点，降低显存占用")
        # except Exception:
        #     pass

        # 不再创建额外的管线，节省显存（训练用不到）
        self.img2img_pipe = None
        self.inpaint_pipe = None
        
        # 噪声调度器
        self.noise_scheduler = DDIMScheduler.from_config(self.txt2img_pipe.scheduler.config)
        
        # 为训练稳定性，将VAE切换到FP32精度（避免Conv反传NaN）
        try:
            self.vae = self.vae.to(dtype=torch.float32)
            print("✅ VAE已切换到float32以增强数值稳定性")
        except Exception:
            pass
        
        print("✅ SDXL管道加载完成（精简模式）")
    
    def _init_clip(self):
        """初始化CLIP模型"""
        # 尝试本地CLIP模型路径
        local_clip_paths = [
            "models/laion-CLIP-ViT-H-14-laion2B-s32B-b79K",  # 首选：完整的HF格式
            "models/ViT-L-14.pt",  # 备选：OpenCLIP格式
            "openai/clip-vit-base-patch32"  # 最后备选：在线下载
        ]
        
        for clip_path in local_clip_paths:
            try:
                if clip_path.endswith('.pt'):
                    # OpenCLIP格式 - 需要特殊处理
                    print(f"🔄 尝试加载OpenCLIP模型: {clip_path}")
                    import open_clip
                    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=clip_path)
                    self.clip_model = model.to(self.device)
                    self.clip_processor = preprocess
                    print(f"✅ OpenCLIP模型加载成功: {clip_path}")
                    self._clip_type = 'openclip'
                    return
                else:
                    # Hugging Face格式
                    print(f"🔄 尝试加载HF CLIP模型: {clip_path}")
                    if os.path.exists(clip_path):
                        self.clip_model = CLIPModel.from_pretrained(clip_path, local_files_only=True).to(self.device)
                        self.clip_processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
                    else:
                        self.clip_model = CLIPModel.from_pretrained(clip_path).to(self.device)
                        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
                    
                    print(f"✅ HF CLIP模型加载成功: {clip_path}")
                    self._clip_type = 'huggingface'
                    return
                    
            except Exception as e:
                print(f"⚠️ CLIP模型加载失败 {clip_path}: {e}")
                continue
        
        # 所有尝试都失败
        print("❌ 所有CLIP模型加载失败，禁用CLIP功能")
        self.clip_model = None
        self.clip_processor = None
        self._clip_type = None
    
    def _setup_lora(self):
        """设置LoRA"""
        lora_rank = getattr(model_config, 'lora_rank', 8)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1,
        )

        self.unet_lora = get_peft_model(self.txt2img_pipe.unet, lora_config)
        print("✅ LoRA设置完成")
    
    def _load_reference_patterns(self):
        """加载参考图案"""
        ref_dir = getattr(training_config, 'reference_patterns_dir', None)
        if ref_dir and os.path.isdir(ref_dir):
            for file in os.listdir(ref_dir)[:20]:  # 限制数量
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(ref_dir, file)
                        img = Image.open(img_path).convert('RGB')
                        self.reference_db.append({
                            'image': img,
                            'path': img_path,
                            'tensor': None  # 延迟加载
                        })
                    except Exception as e:
                        print(f"⚠️ 加载参考图案失败 {file}: {e}")
            
            print(f"✅ 加载了 {len(self.reference_db)} 个参考图案")

    def get_trainable_parameters(self):
        """获取可训练参数（带缓存优化）"""
        # 如果已缓存，直接返回
        if self._trainable_params_cached and self._cached_params:
            return self._cached_params

        trainable_params = []

        # 创建适配器（仅首次）
        if not hasattr(self, 'adapter'):
            # 1. 从 config.py 读取两个功能开关
            use_reference = getattr(training_config, 'adapter_use_reference', False)

            # 2. 将两个开关同时传递给工厂函数
            self.adapter = create_sdxl_adapter(
                smart=self.use_smart_adapter,
                use_reference=use_reference
            )

            self.adapter = self.adapter.to(self.device)
            self.add_module('adapter', self.adapter)

            # 特征投影层 - 使用更复杂的投影来保留颜色信息
            self.feature_projector = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(320, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 3, 3, 1, 1)
                ),   # 64x64
                nn.Sequential(
                    nn.Conv2d(640, 128, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(128, 3, 3, 1, 1)
                ),   # 32x32
                nn.Sequential(
                    nn.Conv2d(1280, 256, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(256, 3, 3, 1, 1)
                ),  # 16x16
                nn.Sequential(
                    nn.Conv2d(1280, 256, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(256, 3, 3, 1, 1)
                ),  # 8x8
            ]).to(self.device)
            self.add_module('feature_projector', self.feature_projector)

            # 适配器→UNet 残差注入投影（T2I-Adapter式注入）
            # 将各尺度适配器特征投影到与UNet down blocks 对齐的通道数
            self.adapter_residual_proj_down = nn.ModuleList([
                nn.Conv2d(320, 320, 1, 1, 0),
                nn.Conv2d(640, 640, 1, 1, 0),
                nn.Conv2d(1280, 1280, 1, 1, 0),
                nn.Conv2d(1280, 1280, 1, 1, 0)
            ]).to(self.device)
            self.adapter_residual_proj_mid = nn.Conv2d(1280, 1280, 1, 1, 0).to(self.device)
            self.add_module('adapter_residual_proj_down', self.adapter_residual_proj_down)
            self.add_module('adapter_residual_proj_mid', self.adapter_residual_proj_mid)

        # 收集可训练参数
        if self.use_lora and hasattr(self, 'unet_lora'):
            trainable_params.extend([p for p in self.unet_lora.parameters() if p.requires_grad])
        
        trainable_params.extend(list(self.adapter.parameters()))
        trainable_params.extend(list(self.feature_projector.parameters()))
        # 注入投影层一并训练
        if hasattr(self, 'adapter_residual_proj_down'):
            trainable_params.extend(list(self.adapter_residual_proj_down.parameters()))
        if hasattr(self, 'adapter_residual_proj_mid'):
            trainable_params.extend(list(self.adapter_residual_proj_mid.parameters()))
        
        # 缓存参数列表
        self._cached_params = trainable_params
        self._trainable_params_cached = True
        
        print(f"可训练参数数量: {sum(p.numel() for p in trainable_params):,}")
        return trainable_params

    def apply_adapter_fast_training(self, batch_images: torch.Tensor, batch_masks: torch.Tensor, 
                                   prompts: Optional[List[str]] = None) -> torch.Tensor:
        """
        快速适配器训练模式 - 不进行扩散采样，直接生成迷彩
        大幅提升训练速度，降低显存使用
        """
        if not hasattr(self, 'adapter'):
            self.get_trainable_parameters()

        # 规范化输入
        batch_images = batch_images.to(self.device)
        batch_masks = batch_masks.to(self.device).float()
        
        if batch_masks.max() > 1.0:
            batch_masks = batch_masks / 255.0
        
        # 调整尺寸
        h, w = batch_images.shape[-2], batch_images.shape[-1]
        if batch_masks.shape[-2:] != (h, w):
            batch_masks = F.interpolate(batch_masks, size=(h, w), mode='nearest')

        # 转换到[0,1]范围
        images_01 = (batch_images + 1.0) / 2.0
        
        # 准备文本条件（如果需要）
        text_embeds = None
        time_embeds = None
        
        if prompts is not None:
            with torch.no_grad():
                cond1, cond2, pooled2 = self._encode_prompts(prompts)
                cond1_pooled = cond1.mean(dim=1)
                text_embeds = torch.cat([pooled2, cond1_pooled], dim=-1)  # [B, 2048]
                
                B = batch_images.shape[0]
                time_embeds = torch.tensor(
                    [h, w, 0, 0, h, w], 
                    device=self.device, dtype=torch.float32
                ).unsqueeze(0).repeat(B, 1)

        adapter_input = torch.cat([images_01, batch_masks], dim=1)
        ref_tensor = None

        # 检查是否应该使用参考适配器
        if isinstance(self.adapter, getattr(__import__('sdxl_advanced_adapter'), 'SDXLAdapterWithReference', object)):
            # (为了简化，这里我们为整个批次选择一个最佳参考图)
            # 1. 将输入转换为PIL图像用于CLIP分析
            first_image_pil = self._tensor_to_pil(batch_images[0])
            first_mask_pil = self._tensor_to_pil(batch_masks[0, 0], is_mask=True)

            # 2. 调用CLIP选择最佳参考图
            best_ref = self._select_best_reference_by_clip(first_image_pil, first_mask_pil)

            # 3. 预处理参考图并构建批次
            if best_ref and 'image' in best_ref:
                ref_pil = best_ref['image'].resize((w, h))
                # 转换为 [-1, 1] 范围的张量
                ref_tensor_single = ((transforms.ToTensor()(ref_pil) * 2 - 1)).to(self.device)
                # 扩展到整个批次
                ref_tensor = ref_tensor_single.unsqueeze(0).repeat(batch_images.shape[0], 1, 1, 1)

        # 适配器输入
        adapter_input = torch.cat([images_01, batch_masks], dim=1)
        
        # 直接通过适配器生成特征
        if hasattr(self.adapter, 'forward'):
            features = self.adapter(adapter_input, ref=ref_tensor, text_embeds=text_embeds, time_embeds=time_embeds)
        else:
            features = self.adapter(adapter_input)

        # 通过特征投影层生成最终迷彩
        # 选择最高分辨率的特征（通常是第一个）
        main_feature = features[0]  # [B, 320, H/4, W/4]
        
        # 投影到RGB
        camo_raw = self.feature_projector[0](main_feature)  # [B, 3, H/4, W/4]
        
        # 上采样到原始分辨率
        camo = F.interpolate(camo_raw, size=(h, w), mode='bilinear', align_corners=False)
        
        # 应用激活函数，确保输出在[-1,1]范围
        camo = torch.tanh(camo)
        
        # 按掩码融合：掩码区域使用生成的迷彩，非掩码区域保持原图
        blend_alpha = 0.9  # 强混合，确保迷彩效果明显
        result = batch_images * (1 - batch_masks * blend_alpha) + camo * (batch_masks * blend_alpha)
        
        # 确保输出范围正确
        result = torch.clamp(result, -1, 1)
        
        return result

    def apply_adapter(self, batch_images: torch.Tensor, batch_masks: torch.Tensor, 
                     prompts: Optional[List[str]] = None) -> torch.Tensor:
        """
        应用适配器生成迷彩
        
        Args:
            batch_images: [B, 3, H, W] 输入图像，范围[-1,1]
            batch_masks: [B, 1, H, W] 掩码
            prompts: 提示词列表
        
        Returns:
            生成的图像 [B, 3, H, W]
        """
        if not hasattr(self, 'adapter'):
            self.get_trainable_parameters()

        # 规范化输入
        batch_images = batch_images.to(self.device)
        batch_masks = batch_masks.to(self.device).float()
        
        if batch_masks.max() > 1.0:
            batch_masks = batch_masks / 255.0
        
        # 调整尺寸
        h, w = batch_images.shape[-2], batch_images.shape[-1]
        if batch_masks.shape[-2:] != (h, w):
            batch_masks = F.interpolate(batch_masks, size=(h, w), mode='nearest')

        # 转换到[0,1]范围
        images_01 = (batch_images + 1.0) / 2.0
        
        # 准备文本条件
        text_embeds = None
        time_embeds = None
        
        if prompts is not None:
            with torch.no_grad():
                cond1, cond2, pooled2 = self._encode_prompts(prompts)
                cond1_pooled = cond1.mean(dim=1)
                text_embeds = torch.cat([pooled2, cond1_pooled], dim=-1)  # [B, 2048]
                
                B = batch_images.shape[0]
                time_embeds = torch.tensor(
                    [h, w, 0, 0, h, w], 
                    device=self.device, dtype=torch.float32
                ).unsqueeze(0).repeat(B, 1)

        # 适配器输入
        adapter_input = torch.cat([images_01, batch_masks], dim=1)
        
        # -------- 方案B：通过 added_cond_kwargs 向UNet注入适配器残差，并进行简化多步采样 --------
        # 1) 先取适配器多尺度特征
        if hasattr(self.adapter, 'forward'):
            features = self.adapter(adapter_input, text_embeds=text_embeds, time_embeds=time_embeds)
        else:
            features = self.adapter(adapter_input)

        # 2) 构造 down_block_additional_residuals 与 mid_block_additional_residual
        unet = self.unet
        unet_dtype = next(unet.parameters()).dtype
        down_residuals = []
        for i in range(min(len(features), len(self.adapter_residual_proj_down))):
            feat = features[i]
            proj = self.adapter_residual_proj_down[i](feat).to(unet_dtype)
            down_residuals.append(proj)
        mid_residual = self.adapter_residual_proj_mid(features[-1]).to(unet_dtype)

        # 3) 文本条件
        if prompts is not None:
            with torch.no_grad():
                cond1, cond2, pooled2 = self._encode_prompts(prompts)
                encoder_hidden_states = torch.cat([cond2, cond1], dim=-1).to(unet_dtype)
                pooled2 = pooled2.to(unet_dtype)
        else:
            # 无提示词则使用零向量（保持形状正确）
            B = batch_images.shape[0]
            device = self.device
            hs = 2048
            encoder_hidden_states = torch.zeros((B, 77, hs), device=device, dtype=unet_dtype)
            pooled2 = torch.zeros((B, 1280), device=device, dtype=unet_dtype)

        target_h, target_w = model_config.image_size[1], model_config.image_size[0]
        time_ids = torch.tensor([target_h, target_w, 0, 0, target_h, target_w], 
                               device=self.device, dtype=unet_dtype).repeat(batch_images.shape[0], 1)

        added = {
            'text_embeds': pooled2,
            'time_ids': time_ids,
            'down_block_additional_residuals': [r for r in down_residuals],
            'mid_block_additional_residual': mid_residual
        }

        # 4) 以噪声为起点做简化DDIM采样（少步数），得到纹理再按掩码混合回去
        vae = self.txt2img_pipe.vae
        scheduler = self.noise_scheduler
        # 采样从纯噪声开始（如需img2img效果，可改为与图像latent按strength混合）
        with torch.no_grad():
            B, _, H, W = batch_images.shape
            latent_h, latent_w = H // 8, W // 8
            latents = torch.randn((B, vae.config.latent_channels if hasattr(vae, 'config') else 4, latent_h, latent_w),
                                  device=self.device, dtype=unet_dtype)
            latents = latents * scheduler.init_noise_sigma

        # 步数：训练/验证分开配置
        num_steps = int(training_config.train_num_steps if self.training else training_config.eval_num_steps)
        scheduler.set_timesteps(num_steps, device=self.device)
        latents = latents.to(unet_dtype)
        # classifier-free guidance：训练时简化为单前向，验证时使用完整CFG
        guidance_scale = getattr(model_config, 'guidance_scale', 7.5)
        
        if self.training:
            # 训练时：仅有条件前向（大幅提速）
            for t in scheduler.timesteps:
                noise_pred = unet(latents, t, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample
        else:
            # 验证时：完整CFG以保证质量
            uncond = torch.zeros_like(encoder_hidden_states)
            with torch.no_grad():
                for t in scheduler.timesteps:
                    noise_pred_uncond = unet(latents, t, encoder_hidden_states=uncond, added_cond_kwargs=added).sample
                    noise_pred_text = unet(latents, t, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added).sample
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 在解码前释放部分中间张量（保留必要的显存清理）
        try:
            if 'noise_pred_uncond' in locals():
                del noise_pred_uncond
            if 'noise_pred_text' in locals():
                del noise_pred_text
            if 'uncond' in locals():
                del uncond
        except Exception:
            pass

        # 解码前清理显存碎片
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 解码到图像：训练保留梯度，评估 no_grad；强制使用FP32以防Conv反传NaN
        latents = latents / 0.18215
        latents = torch.nan_to_num(latents, nan=0.0, posinf=1e3, neginf=-1e3)
        target_dtype = torch.float32
        if self.training:
            decoded = vae.decode(latents.to(target_dtype)).sample
        else:
            with torch.no_grad():
                decoded = vae.decode(latents.to(target_dtype)).sample
        # 数值安全：去除NaN/Inf并限制到[-1,1]
        camo = torch.nan_to_num(decoded, nan=0.0, posinf=1.0, neginf=-1.0)
        camo = camo.clamp(-1, 1).to(batch_images.dtype)

        # 按掩码融合
        masks_expanded = batch_masks.repeat(1, 3, 1, 1)
        blend_alpha = 0.85
        result = batch_images * (1.0 - blend_alpha * masks_expanded) + camo * (blend_alpha * masks_expanded)
        # 最终返回前再次确保数值稳定
        result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
        return result

    def forward(self, images: torch.Tensor, masks: torch.Tensor, prompts: Optional[List[str]] = None) -> torch.Tensor:
        """nn.Module 兼容前向：委托到 apply_adapter。"""
        return self.apply_adapter(images, masks, prompts)

    @torch.no_grad()
    def _encode_prompts(self, prompts: List[str]):
        """编码提示词"""
        device = self.device
        te1 = self.txt2img_pipe.text_encoder
        te2 = self.txt2img_pipe.text_encoder_2
        tok1 = self.txt2img_pipe.tokenizer
        tok2 = self.txt2img_pipe.tokenizer_2
        
        ids1 = tok1(prompts, padding='max_length', max_length=tok1.model_max_length, 
                   truncation=True, return_tensors='pt').input_ids.to(device)
        ids2 = tok2(prompts, padding='max_length', max_length=tok2.model_max_length, 
                   truncation=True, return_tensors='pt').input_ids.to(device)
        
        out1 = te1(ids1, output_hidden_states=False, return_dict=True)
        out2 = te2(ids2, output_hidden_states=False, return_dict=True)
        
        emb1 = out1.last_hidden_state
        emb2 = out2.last_hidden_state
        pooled2 = out2.pooler_output if hasattr(out2, 'pooler_output') and out2.pooler_output is not None else emb2.mean(dim=1)
        
        return emb1, emb2, pooled2

    def compute_diffusion_loss(self, images: torch.Tensor, masks: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """计算扩散损失"""
        if not (self.use_lora and hasattr(self, 'unet_lora')):
            return torch.tensor(0.0, device=self.device)
            
        device = self.device
        vae = self.txt2img_pipe.vae
        unet = self.unet_lora
        scheduler = self.noise_scheduler
        
        # 编码文本
        with torch.no_grad():
            cond1, cond2, pooled2 = self._encode_prompts(prompts)
            encoder_hidden_states = torch.cat([cond2, cond1], dim=-1)
            
            # 类型转换
            unet_dtype = next(unet.parameters()).dtype
            encoder_hidden_states = encoder_hidden_states.to(unet_dtype)
            pooled2 = pooled2.to(unet_dtype)

        # 编码图像（尽量保持半精度，不强制到FP32）
        with torch.no_grad():
            img_01 = (images * 0.5) + 0.5
            img_01 = img_01.to(next(vae.parameters()).dtype)
            latents = vae.encode(img_01).latent_dist.sample() * 0.18215

        # 添加噪声
        bsz = latents.shape[0]
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # SDXL条件（先用FP32生成，再与UNet精度对齐）
        target_h, target_w = model_config.image_size[1], model_config.image_size[0]
        time_ids = torch.tensor([target_h, target_w, 0, 0, target_h, target_w], 
                               device=device, dtype=torch.float32).repeat(bsz, 1)
        
        pooled2 = pooled2.to(torch.float32)
        time_ids = time_ids.to(torch.float32)

        # 预测目标
        pred_type = getattr(scheduler.config, 'prediction_type', 'epsilon')
        if pred_type == 'epsilon':
            target = noise
        elif pred_type in ('v_prediction', 'v-prediction', 'v'):
            try:
                target = scheduler.get_velocity(latents, noise, timesteps)
            except Exception:
                target = noise
        else:
            target = noise

        # UNet预测
        # 维持与UNet参数一致的精度，避免无谓的dtype转换占用显存
        target_dtype = next(unet.parameters()).dtype
        pooled2 = pooled2.to(target_dtype)
        time_ids = time_ids.to(target_dtype)
        added = {'text_embeds': pooled2, 'time_ids': time_ids}

        noisy_latents = torch.nan_to_num(noisy_latents.to(target_dtype), nan=0.0, posinf=1e3, neginf=-1e3)
        encoder_hidden_states = torch.nan_to_num(encoder_hidden_states.to(target_dtype), nan=0.0, posinf=1e3, neginf=-1e3)
        
        model_pred = unet(noisy_latents, timesteps, 
                         encoder_hidden_states=encoder_hidden_states, 
                         added_cond_kwargs=added).sample
        
        # 数值稳健：清理NaN/Inf并裁剪极端值，避免PowBackward0 NaN
        model_pred = torch.nan_to_num(model_pred, nan=0.0, posinf=1e3, neginf=-1e3)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e3, neginf=-1e3)
        diff = (model_pred.float() - target.float())
        diff = torch.clamp(diff, -1e3, 1e3)
        loss = torch.mean(diff * diff)
        return loss

    def generate_adaptive_camouflage_prompt(self, image: torch.Tensor, 
                                          mask: Optional[torch.Tensor] = None) -> Tuple[str, str]:
        """生成严格符合迷彩风格的提示词（纹理细节优化版）"""
        
        # 1. 强化基础提示词：强调细节和真实感
        base_prompts = [
        "digital camouflage pattern, pixelated, 8-bit style, seamlessly blending with the background environment, matching background colors",
        "hyperrealistic pixel-level camouflage texture, intricate pixel grid, perfect environmental integration, pixelated color harmony, matches the background",
        "a pixelated pattern with crisp micro and macro pixel details, perfectly matching the environmental color and light, sharp focus, digital camo",
        "advanced digital camouflage, ultra-realistic pixelated forms, seamless environmental adaptation, 4k high resolution, blends into the background",
        "a pixel-level camouflage pattern that precisely matches the background texture and color, high quality pixelated texture, 8-bit camo style"
        ]
        
        # 2. 丰富风格库：加入更多关于质感、光影和清晰度的关键词
        style_elements = [
            "intricate textures", "sharp focus", "hyperrealistic", "professional photography",
            "natural lighting", "subtle shadows", "UHD", "trending on ArtStation",
            "filmic", "physically-based rendering", "cinematic lighting",
            "environmental color matching", "seamless background integration", 
            "organic shape distribution", "tactical effectiveness"
        ]
        

        
        # 确保每次都有质量保证词缀
        quality_boilerplate = "(best quality, 4k, highres, masterpiece:1.2)"
        
        positive_prompt = f"{base_prompts}, {', '.join(style_elements)}, {quality_boilerplate}"
        
        # 从config加载整合后的负面提示词
        from config import NEGATIVE_PROMPTS_CONFIG
        
        # 增强负面提示，避免模糊和低质量
        additional_negative = [
            "blurry, soft focus, out of focus, low quality, jpeg artifacts",
            "smooth, plain, flat texture, glossy, plastic look",
            "painting, drawing, cartoon, anime, illustration",
            "watermark, text, signature, logo"
        ]

        # 组合所有负面提示，并去重
        all_negative_elements = (NEGATIVE_PROMPTS_CONFIG['base'] +
                                 NEGATIVE_PROMPTS_CONFIG['color_strict'] +
                                 additional_negative)
        negative_prompt = ", ".join(list(dict.fromkeys(all_negative_elements))) # 使用dict.fromkeys去重并保持顺序
        
        return positive_prompt, negative_prompt

    def _tensor_to_pil(self, tensor: torch.Tensor, is_mask: bool = False) -> Image.Image:
        """
        将张量转换为PIL图像 (完整功能版)
        - 支持普通图像和单通道掩码
        """
        if is_mask:
            # --- 掩码处理逻辑 ---
            # 掩码张量通常是 [H, W] 或 [1, H, W]，范围 [0, 1]
            if tensor.dim() == 3:  # 如果是 [1, H, W]
                tensor = tensor.squeeze(0)  # 降维到 [H, W]

            # 将 [0, 1] 范围的张量转换为 [0, 255] 的numpy数组
            np_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
            # 从数组创建灰度图 ('L' mode)
            return Image.fromarray(np_array, mode='L')
        else:
            # --- 普通图像处理逻辑 ---
            # 反归一化，从 [-1, 1] 范围转换到 [0, 1]
            tensor = (tensor + 1.0) / 2.0
            tensor = torch.clamp(tensor, 0, 1)

            # 转换为 [H, W, C] 的numpy数组并调整数值范围
            img_array = tensor.permute(1, 2, 0).cpu().numpy()
            img_array = (img_array * 255).astype(np.uint8)
            # 从数组创建彩色图 ('RGB' mode)
            return Image.fromarray(img_array, mode='RGB')

    def _select_best_reference_by_clip(self, image_pil: Image.Image, mask_pil: Image.Image):
        """使用CLIP选择最佳参考图像（基于背景区域）"""
        if not self.reference_db or self.clip_model is None:
            return {'image': image_pil, 'tensor': None}
        
        # 基于背景区域的图像（掩码外）
        try:
            if mask_pil is not None:
                # 背景 = 原图 * (1 - mask)
                inv_mask = Image.eval(mask_pil.convert('L'), lambda p: 255 - p)
                bg_image = Image.composite(image_pil, Image.new('RGB', image_pil.size, (0,0,0)), inv_mask)
            else:
                bg_image = image_pil
        except Exception:
            bg_image = image_pil
        
        try:
            if self._clip_type == 'huggingface':
                inputs = self.clip_processor(images=bg_image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                
                best_similarity = -1
                best_ref = self.reference_db[0]
                
                for ref in self.reference_db[:5]:
                    ref_inputs = self.clip_processor(images=ref['image'], return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        ref_features = self.clip_model.get_image_features(**ref_inputs)
                    
                    similarity = torch.cosine_similarity(image_features, ref_features).item()
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_ref = ref
                
                return best_ref
                
            elif self._clip_type == 'openclip':
                image_tensor = self.clip_processor(bg_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_tensor)
                
                best_similarity = -1
                best_ref = self.reference_db[0]
                
                for ref in self.reference_db[:5]:
                    ref_tensor = self.clip_processor(ref['image']).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        ref_features = self.clip_model.encode_image(ref_tensor)
                    
                    similarity = torch.cosine_similarity(image_features, ref_features).item()
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_ref = ref
                
                return best_ref
            else:
                import random
                return random.choice(self.reference_db) if self.reference_db else {'image': image_pil, 'tensor': None}
        except Exception as e:
            print(f"⚠️ CLIP参考选择失败: {e}")
            return self.reference_db[0] if self.reference_db else {'image': image_pil, 'tensor': None}

    def save_model(self, save_path: str):
        """保存模型"""
        state_dict = {}
        
        if hasattr(self, 'adapter'):
            state_dict['adapter_state_dict'] = self.adapter.state_dict()
        
        if hasattr(self, 'feature_projector'):
            state_dict['feature_projector_state_dict'] = self.feature_projector.state_dict()
        
        if hasattr(self, 'unet_lora'):
            state_dict['unet_lora_state_dict'] = self.unet_lora.state_dict()
        
        torch.save(state_dict, save_path)
        print(f"✅ 模型已保存到: {save_path}")

    def load_model(self, load_path: str):
        """加载模型"""
        if not os.path.exists(load_path):
            print(f"⚠️ 模型文件不存在: {load_path}")
            return
        
        state_dict = torch.load(load_path, map_location=self.device)
        
        # 确保适配器存在
        if not hasattr(self, 'adapter'):
            self.get_trainable_parameters()
        
        # 加载状态
        if 'adapter_state_dict' in state_dict:
            self.adapter.load_state_dict(state_dict['adapter_state_dict'])
        
        if 'feature_projector_state_dict' in state_dict:
            self.feature_projector.load_state_dict(state_dict['feature_projector_state_dict'])
        
        if 'unet_lora_state_dict' in state_dict and hasattr(self, 'unet_lora'):
            self.unet_lora.load_state_dict(state_dict['unet_lora_state_dict'])
        
        print(f"✅ 模型已加载: {load_path}")
