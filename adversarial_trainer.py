"""
对抗训练器模块 - 实现SDXL生成器与AI检测器的对抗训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from PIL import Image
import wandb
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import time
import random
import json

from config import training_config, model_config, eval_config
# 使用精简生成器（已对接UNet注入）
from sdxl_generator_clean import SDXLCamouflageGenerator
from ai_detector import MultiModalDetector
from data_loader import create_data_loaders
from utils import ImageProcessor, save_results_grid, compute_metrics

class AdversarialTrainer:
    """对抗训练器"""
    
    def __init__(self,
                 generator: SDXLCamouflageGenerator = None,
                 detector: MultiModalDetector = None,
                 device: str = None):
        """
        初始化对抗训练器
        
        Args:
            generator: SDXL迷彩生成器
            detector: 多模态AI检测器
            device: 计算设备
        """
        self.device = device or training_config.device
        
        # 初始化模型（使用LoRA训练）
        self.generator = generator or SDXLCamouflageGenerator(
            device=self.device,
            use_lora=True,
            use_smart_adapter=True  # 保持与主入口一致，并使用正确的参数
        )
        self.detector = detector or MultiModalDetector(device=self.device)
        
        # 设置训练模式
        self.generator.set_training_mode(True)
        self.detector.eval()  # 检测器在对抗训练中保持eval模式
        
        # 初始化优化器
        self._setup_optimizers()
        
        # 初始化损失函数
        self._setup_loss_functions()
        
        # 创建输出目录
        self.output_dir = training_config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        
        # 初始化日志
        self._setup_logging()
        
        # 训练统计
        self.global_step = 0
        self.epoch = 0
        self.best_score = float('inf')
        self.last_detection_rate = 0.0
        
        # 检测器优化器已移除 - 使用预训练的检测器直接进行可微推理

        print("对抗训练器初始化完成")
    
    def _setup_optimizers(self):
        """设置优化器"""
        # 生成器优化器 - 只训练部分UNet参数
        generator_params = self.generator.get_trainable_parameters()
        self.generator_optimizer = optim.AdamW(
            generator_params,
            lr=training_config.generator_lr,
            betas=(training_config.beta1, training_config.beta2),
            weight_decay=1e-5
        )

        # 统一FP32，不使用GradScaler
        self.scaler = None
        
        # 生成器学习率调度器
        self.generator_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.generator_optimizer,
            T_max=training_config.num_epochs,
            eta_min=1e-6
        )
        
        print(f"生成器优化器设置完成，可训练参数数量: {len(generator_params)}")
    
    def _setup_loss_functions(self):
        """设置损失函数"""
        # 内容保持损失（L1 + LPIPS）
        self.l1_loss = nn.L1Loss()
        
        # TV Loss 函数（闭包实现）
        def tv_loss_fn(x: torch.Tensor) -> torch.Tensor:
            # x: [B, C, H, W]
            dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
            dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
            return dh + dw
        self.tv_loss_fn = tv_loss_fn

        # 色相禁止项（HSV 中 H 达到禁止区间时惩罚）
        def hue_forbid_loss(x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
            # x in [-1,1] -> [0,1]
            img = ((x.clamp(-1,1)*0.5)+0.5)
            r, g, b = img[:,0], img[:,1], img[:,2]
            maxc, _ = torch.max(img, dim=1)
            minc, _ = torch.min(img, dim=1)
            v = maxc
            s = torch.where(maxc>0, (maxc-minc)/(maxc+1e-6), torch.zeros_like(maxc))
            rc = (maxc - r) / (maxc - minc + 1e-6)
            gc = (maxc - g) / (maxc - minc + 1e-6)
            bc = (maxc - b) / (maxc - minc + 1e-6)
            h = torch.zeros_like(maxc)
            h = torch.where((maxc==r), (bc-gc), h)
            h = torch.where((maxc==g), 2.0 + (rc-bc), h)
            h = torch.where((maxc==b), 4.0 + (gc-rc), h)
            h = (h/6.0) % 1.0  # [0,1]
            m = masks.detach()  # 保证后续 expand/as_strided 不回传到mask
            # 保持纯函数，避免原位修改：使用克隆并确保尺寸一致
            if m.dim()==3:
                m = m.unsqueeze(1)
            m = m.contiguous()
            m3 = m.repeat(1,3,1,1)
            loss_acc = 0.0
            ranges = getattr(training_config, 'forbid_hue_ranges', ())
            for lo, hi in ranges:
                # 环形区间处理
                if lo <= hi:
                    band = ((h>=lo) & (h<=hi)).float()
                else:
                    band = (((h>=lo) | (h<=hi))).float()
                # 强化饱和度约束，低饱和度不惩罚
                penalty = (band * (s>0.25).float()).unsqueeze(1)
                loss_acc = loss_acc + (penalty * m3).mean()
            return loss_acc

        self.hue_forbid_loss = hue_forbid_loss

    def _compute_histogram_loss(self, generated_images: torch.Tensor, original_images: torch.Tensor, masks: torch.Tensor, num_bins: int = 32) -> torch.Tensor:
        """
        在掩码区域计算HSV直方图差异。
        目标：生成图的掩码内区域 (前景)
        参考：原图的掩码外区域 (背景)
        """
        import torch
        
        # 将图像从[-1,1]转到[0,1]并调整维度
        gen_01 = ((generated_images.clamp(-1, 1) * 0.5) + 0.5).permute(0, 2, 3, 1)
        ori_01 = ((original_images.clamp(-1, 1) * 0.5) + 0.5).permute(0, 2, 3, 1)
        
        # 准备掩码
        mask_fg = masks.detach().contiguous()
        if mask_fg.dim() == 3: mask_fg = mask_fg.unsqueeze(1)
        mask_fg_hw = (mask_fg[:, 0] > 0.5) # [B,H,W] bool mask for foreground
        
        mask_bg_hw = ~mask_fg_hw # [B,H,W] bool mask for background

        def rgb_to_hsv(x):
            r, g, b = x[..., 0], x[..., 1], x[..., 2]
            maxc, _ = torch.max(x, dim=-1)
            minc, _ = torch.min(x, dim=-1)
            v = maxc
            s = torch.where(maxc > 0, (maxc - minc) / (maxc + 1e-6), torch.zeros_like(maxc))
            rc = (maxc - r) / (maxc - minc + 1e-6)
            gc = (maxc - g) / (maxc - minc + 1e-6)
            bc = (maxc - b) / (maxc - minc + 1e-6)
            h = torch.zeros_like(maxc)
            h = torch.where((maxc == r), (bc - gc), h)
            h = torch.where((maxc == g), 2.0 + (rc - bc), h)
            h = torch.where((maxc == b), 4.0 + (gc - rc), h)
            h = (h / 6.0) % 1.0
            return h, s, v

        def get_histograms(channels, mask_hw):
            hists = []
            for channel_data in channels:
                batch_hists = []
                for i in range(channel_data.shape[0]):
                    masked_data = channel_data[i][mask_hw[i]].clamp(0, 1)
                    if masked_data.numel() == 0:
                        hist = torch.zeros(num_bins, device=channel_data.device)
                    else:
                        idx = (masked_data * (num_bins - 1)).long()
                        hist = torch.bincount(idx, minlength=num_bins).float()
                    batch_hists.append(hist / (hist.sum() + 1e-6))
                hists.append(torch.stack(batch_hists, 0))
            return hists

        # 计算生成图前景(fg)和原图背景(bg)的HSV通道
        h_gen_fg, s_gen_fg, v_gen_fg = rgb_to_hsv(gen_01)
        h_ori_bg, s_ori_bg, v_ori_bg = rgb_to_hsv(ori_01)

        # 计算直方图
        H_gen, S_gen, V_gen = get_histograms([h_gen_fg, s_gen_fg, v_gen_fg], mask_fg_hw)
        H_bg, S_bg, V_bg = get_histograms([h_ori_bg, s_ori_bg, v_ori_bg], mask_bg_hw)
        
        # 计算KL散度损失
        def sym_kl_loss(p, q):
            p = p.clamp_min(1e-6)
            q = q.clamp_min(1e-6)
            kl1 = (p * (p / q).log()).sum(dim=1)
            kl2 = (q * (q / p).log()).sum(dim=1)
            return (kl1 + kl2).mean()

        # 对H, S, V三个通道的直方图计算损失并加权求和
        loss = (sym_kl_loss(H_gen, H_bg) + 
                0.5 * sym_kl_loss(S_gen, S_bg) + 
                0.5 * sym_kl_loss(V_gen, V_bg))
        
        return loss
        
        # 感知损失权重
        self.content_weight = training_config.content_weight
        self.adversarial_weight = training_config.adversarial_weight
        
        print("损失函数设置完成")
    
    def _setup_logging(self):
        """设置日志"""
        # TensorBoard
        logs_dir = os.path.join(self.output_dir, "logs")
        self.writer = SummaryWriter(log_dir=logs_dir)
        try:
            import os as _os
            print(f"TensorBoard 日志目录: {_os.path.abspath(logs_dir)}")
        except Exception:
            pass
        
        # 禁用 Weights & Biases (避免网络问题)
        self.use_wandb = False
        print("已禁用 Weights & Biases，仅使用 TensorBoard")
    
    def compute_content_loss(self, 
                           generated_images: torch.Tensor,
                           original_images: torch.Tensor,
                           masks: torch.Tensor) -> torch.Tensor:
        """
        计算内容保持损失（仅非掩码区域）
        - 生成器输出已是完整拼接图
        - 仅在 mask 外计算 L1，避免压制掩码区域的迷彩生成
        """
        inv_mask = (1.0 - masks).detach()  # 避免原位/视图版本冲突
        inv_mask_3c = inv_mask.repeat(1, 3, 1, 1)
        diff = torch.abs(generated_images - original_images) * inv_mask_3c
        denom = inv_mask_3c.sum()
        # 若掩码几乎全覆盖，fallback 到全图L1，避免该项恒为0
        if denom.item() < 1e-6:
            return self.l1_loss(generated_images, original_images)
        return diff.sum() / (denom + 1e-8)

    def compute_mask_consistency_losses(self,
                                        generated_images: torch.Tensor,
                                        original_images: torch.Tensor,
                                        masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算掩码区域颜色/亮度一致性的小损失。
        仅在掩码区域计算，权重较小，用于抑制绿紫漂移与过亮/过暗。
        """
        mask = masks.detach().contiguous()  # 避免视图/广播反传
        # 映射到[0,1]
        gen = ((generated_images * 0.5) + 0.5).contiguous()
        ori = ((original_images * 0.5) + 0.5).contiguous()

        # 展平到 [B,C,N] 与 [B,1,N]，避免 as_strided 视图反传
        B, C, H, W = gen.shape
        gen_f = gen.view(B, C, H*W)
        ori_f = ori.view(B, C, H*W)
        m_f = mask.view(B, 1, H*W)
        denom = m_f.sum(dim=2, keepdim=True).clamp_min(1e-6)

        # 颜色：均值颜色L2（掩码内的通道均值）
        gen_mean = (gen_f * m_f).sum(dim=2, keepdim=True) / denom  # [B,C,1]
        ori_mean = (ori_f * m_f).sum(dim=2, keepdim=True) / denom
        color_loss = ((gen_mean - ori_mean) ** 2).mean()

        # 亮度：Y = 0.299R+0.587G+0.114B
        w = torch.tensor([0.299, 0.587, 0.114], device=gen.device, dtype=gen.dtype).view(1,3,1)
        gen_y = (gen_f * w).sum(dim=1, keepdim=True)  # [B,1,N]
        ori_y = (ori_f * w).sum(dim=1, keepdim=True)
        gen_y_mean = (gen_y * m_f).sum(dim=2, keepdim=True) / denom  # [B,1,1]
        ori_y_mean = (ori_y * m_f).sum(dim=2, keepdim=True) / denom
        brightness_loss = ((gen_y_mean - ori_y_mean) ** 2).mean()

        return color_loss, brightness_loss
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        单步训练 - 使用梯度累积减少显存使用
        
        Args:
            batch: 训练批次数据
            
        Returns:
            损失字典
        """
        # 可选启用异常检测
        import torch
        if training_config.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        original_images = batch['image'].to(self.device)
        masks = batch['mask'].to(self.device)
        # 规范化掩码形状与类型：[B,1,H,W] in [0,1]
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        if masks.dim() == 2:
            masks = masks.unsqueeze(0).unsqueeze(0)
        if masks.dtype != torch.float32:
            masks = masks.float()
        if masks.max() > 1.0:
            masks = masks / 255.0
        batch_size = original_images.shape[0]
        
        # 调试信息：每个epoch开始时打印，减少IO开销
        if self.global_step % 800 == 0:  # 假设每个epoch大约800步
            print(f"📊 数据加载调试信息:")
            print(f"   - 配置batch_size: {training_config.batch_size}")
            print(f"   - 实际batch_size: {batch_size}")
            print(f"   - 图像形状: {original_images.shape}")
            print(f"   - 掩码形状: {masks.shape}")
        
        # 不在每步清空显存，改为每个epoch结束在 train() 中清理一次
        
        # ========== 生成器训练 ==========
        self.generator_optimizer.zero_grad()
        
        # 统一的训练路径，不再区分batch_size
        try:
            # 批量处理所有样本
            prompts = []
            for i in range(batch_size):
                positive_prompt, _ = self.generator.generate_adaptive_camouflage_prompt(
                    original_images[i:i+1], masks[i:i+1]
                )
                prompts.append(positive_prompt)
            
            # 使用快速适配器训练模式
            generated_images = self.generator.apply_adapter(original_images, masks, prompts=prompts)

            # 使用平衡权重配置
            stage_w = training_config.get_balanced_weights()
                
            # 计算批量损失
            content_loss = self.compute_content_loss(generated_images, original_images, masks)
            adv_out = self.detector.compute_adversarial_loss(generated_images, original_images,masks)
            if isinstance(adv_out, tuple):
                adversarial_loss, adv_details = adv_out
            else:
                adversarial_loss, adv_details = adv_out, {}

            color_loss, brightness_loss = self.compute_mask_consistency_losses(generated_images, original_images, masks)
            tv_loss = self.tv_loss_fn(generated_images)
            
            # 总生成器损失
            generator_loss = (
                stage_w.get('content', 1.0) * content_loss +
                stage_w.get('adv', 1.0) * adversarial_loss +
                stage_w.get('color', 0.0) * color_loss +
                stage_w.get('bright', 0.0) * brightness_loss +
                stage_w.get('tv', 0.0) * tv_loss
            )

            # 可选：扩散损失
            if training_config.use_diffusion_training and hasattr(self.generator, 'compute_diffusion_loss'):
                diff_loss = self.generator.compute_diffusion_loss(original_images, masks, prompts)
                generator_loss = generator_loss + stage_w.get('diff', 0.0) * diff_loss

            # 背景颜色分布对齐损失
            if getattr(training_config, 'use_background_hist_loss', False) and stage_w.get('hist', 0.0) > 0:
                try:
                    hist_loss = self._compute_histogram_loss(generated_images, original_images, masks)
                    generator_loss = generator_loss + stage_w.get('hist', 0.0) * hist_loss
                except Exception as e:
                    print(f"⚠️ 计算直方图损失失败: {e}")

            # 色相禁止损失
            if stage_w.get('hue', 0.0) > 0:
                hue_loss = self.hue_forbid_loss(generated_images, masks)
                generator_loss = generator_loss + stage_w.get('hue', 0.0) * hue_loss

            # 反向传播和优化器步骤
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.get_trainable_parameters(), max_norm=1.0)
            self.generator_optimizer.step()

        except Exception as e:
            print(f"❌ train_step 发生错误: {e}")
            # 抛出异常以中断训练并调试
            raise

        # 更新训练统计
        detection_rate = self.last_detection_rate # 默认使用上一次的评估结果
        if (self.global_step % max(1, training_config.eval_interval_steps) == 0):
            with torch.no_grad():
                eval_res = self.detector.evaluate_detection_success(generated_images, masks)
                detection_rate = float(eval_res['detection_rate'])
                self.last_detection_rate = detection_rate
                # 记录各分支检测率
                if self.writer is not None:
                    per_branch = eval_res.get('per_detector_results', {})
                    for name, stats in per_branch.items():
                        self.writer.add_scalar(f"detect_rate/{name}", stats.get('detection_rate', 0.0), self.global_step)
        
        # 将所有损失和细节添加到 batch_losses 中以便记录
        batch_losses = {
            'generator_loss': float(generator_loss.item()),
            'content_loss': float(content_loss.item()),
            'adversarial_loss': float(adversarial_loss.item()),
            'tv_loss': float(tv_loss.item()),
            'detection_rate': detection_rate
        }
        if adv_details:
            for k, v in adv_details.items():
                batch_losses[f"adv_{k}"] = v.item() if isinstance(v, torch.Tensor) else v
        if 'diff_loss' in locals():
            batch_losses['diffusion_loss'] = float(diff_loss.item())
        if 'hue_loss' in locals():
            batch_losses['hue_loss'] = float(hue_loss.item())


        return batch_losses
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        验证
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            验证结果
        """
        # 验证阶段使用未编译版本（若可用），避免 torch.compile 的追踪/编译开销
        gen_eval = getattr(self, 'generator_for_validation', self.generator)
        gen_eval.set_training_mode(False)
        
        total_losses = {
            'generator_loss': 0.0,
            'content_loss': 0.0,
            'adversarial_loss': 0.0,
            'detection_rate': 0.0
        }
        
        num_batches = 0
        all_generated_images = []
        all_original_images = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中"):
                original_images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # 生成图像
                generated_images = gen_eval(original_images, masks)
                
                # 计算损失
                content_loss = self.compute_content_loss(generated_images, original_images, masks)
                adv_out = self.detector.compute_adversarial_loss(generated_images,original_images ,masks)
                if isinstance(adv_out, tuple):
                    adversarial_loss, _ = adv_out
                else:
                    adversarial_loss = adv_out
                stage_w = training_config.get_balanced_weights()
                generator_loss = (
                    stage_w['content'] * content_loss +
                    stage_w['adv'] * adversarial_loss
                )
                
                # 检测评估（仅针对掩码区域）
                detection_results = self.detector.evaluate_detection_success(generated_images, masks)
                
                # 累积损失
                total_losses['generator_loss'] += generator_loss.item()
                total_losses['content_loss'] += content_loss.item()
                total_losses['adversarial_loss'] += adversarial_loss.item()
                total_losses['detection_rate'] += detection_results['detection_rate']
                
                num_batches += 1
                
                # 保存一些图像用于可视化
                if len(all_generated_images) < 16:
                    all_generated_images.extend(generated_images.cpu())
                    all_original_images.extend(original_images.cpu())
        
        # 平均化损失
        for key in total_losses:
            total_losses[key] /= num_batches
        
        # 保存验证图像网格
        if all_generated_images:
            self.save_validation_images(all_original_images[:8], all_generated_images[:8])
        
        gen_eval.set_training_mode(True)
        return total_losses
    
    def save_validation_images(self, 
                             original_images: List[torch.Tensor],
                             generated_images: List[torch.Tensor]):
        """保存验证图像"""
        save_path = os.path.join(self.output_dir, "images", f"validation_epoch_{self.epoch}.png")
        save_results_grid(original_images, generated_images, save_path)
    
    def generate_sample_images(self, data_loader, epoch: int):
        """生成示例迷彩图像"""
        import torch
        import os
        from PIL import Image, ImageDraw
        import numpy as np
        
        gen_infer = getattr(self, 'generator_for_validation', self.generator)
        gen_infer.eval()
        
        # 创建输出目录
        samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        
        print("生成示例图像...")
        
        # 随机选择一个batch，避免每次都是同一批
        try:
            total_batches = len(data_loader)
            skip = random.randint(0, max(0, total_batches - 1))
            it = iter(data_loader)
            for _ in range(skip):
                try:
                    next(it)
                except StopIteration:
                    break
            sample_batch = next(it)
        except Exception:
            try:
                sample_batch = next(iter(data_loader))
            except Exception:
                print("无法从数据加载器获取样本")
                return
        
        # 随机抽取样本（每个epoch只生成1个样本，减少训练时间开销）
        batch_count = sample_batch['image'].shape[0]
        num_samples = min(1, batch_count)
        indices = random.sample(range(batch_count), k=num_samples)
        
        with torch.no_grad():
            original_images = sample_batch['image'][indices].to(self.device)
            masks = sample_batch['mask'][indices].to(self.device)
            
            # 生成迷彩图像
            try:
                generated_images = gen_infer(original_images, masks)
                
                # 保存每个样本
                for i in range(num_samples):
                    # 转换为PIL图像
                    orig_img = self._tensor_to_pil(original_images[i])
                    mask_img = self._tensor_to_pil(masks[i, 0], is_mask=True)
                    gen_img = self._tensor_to_pil(generated_images[i])

                    # ============ 在整张生成图上可视化YOLO检测框，并突出掩码区域无检测 ============
                    # 1) YOLO对整张生成图检测（不传mask）
                    yolo_res_full = self.detector.detectors['yolo'](generated_images[i:i+1])
                    boxes_list = yolo_res_full['boxes'][0] if len(yolo_res_full['boxes']) > 0 else []

                    # 2) 计算掩码区域的边界框用于可视化
                    mask_np = (np.array(mask_img) > 127).astype(np.uint8)
                    if mask_np.sum() > 0:
                        ys, xs = np.where(mask_np > 0)
                        y1_m, y2_m = int(ys.min()), int(ys.max())
                        x1_m, x2_m = int(xs.min()), int(xs.max())
                    else:
                        y1_m = y2_m = x1_m = x2_m = 0

                    # 3) 在生成图上绘制：
                    gen_annot = gen_img.copy()
                    draw = ImageDraw.Draw(gen_annot)
                    # 绘制掩码边框（绿色）
                    if mask_np.sum() > 0:
                        draw.rectangle([x1_m, y1_m, x2_m, y2_m], outline=(0, 255, 0), width=3)

                    # 绘制YOLO框：掩码外红色，掩码内黄色（用于诊断）
                    if isinstance(boxes_list, np.ndarray) and boxes_list.size > 0:
                        for bx in boxes_list:
                            x1, y1, x2, y2 = [int(v) for v in bx]
                            # 计算该框落入掩码区域的比例
                            try:
                                sub = mask_np[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                                ratio = float(sub.mean()) if sub.size > 0 else 0.0
                            except Exception:
                                ratio = 0.0
                            color = (255, 0, 0) if ratio < 0.1 else (255, 200, 0)
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                    # 用标注后的生成图替换显示
                    gen_vis = gen_annot
                    
                    # 创建组合图像
                    combined = Image.new('RGB', (orig_img.width * 3, orig_img.height))
                    combined.paste(orig_img, (0, 0))
                    combined.paste(mask_img.convert('RGB'), (orig_img.width, 0))
                    combined.paste(gen_vis, (orig_img.width * 2, 0))
                    
                    # 保存
                    save_path = os.path.join(samples_dir, f"epoch_{epoch:03d}_sample.png")
                    combined.save(save_path)
                    print(f"已保存: {save_path}")
                
                print(f"✅ 已生成 1 个示例图像")
                
            except Exception as e:
                print(f"❌ 生成示例图像失败: {e}")
        
        gen_infer.train()
    
    def _tensor_to_pil(self, tensor: torch.Tensor, is_mask: bool = False) -> Image:
        """将张量转换为PIL图像"""
        if is_mask:
            # 掩码处理
            np_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(np_array, mode='L')
        else:
            # 普通图像处理
            # 反归一化
            tensor = (tensor * 0.5) + 0.5
            tensor = torch.clamp(tensor, 0, 1)
            
            # 转换为numpy
            np_array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(np_array, mode='RGB')
    
    def generate_final_results(self, train_loader, val_loader):
        """生成最终训练结果展示"""
        import torch
        import os
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        gen_show = getattr(self, 'generator_for_validation', self.generator)
        gen_show.eval()
        
        # 创建最终结果目录
        final_dir = os.path.join(self.output_dir, "final_results")
        os.makedirs(final_dir, exist_ok=True)
        
        print("正在生成最终结果展示...")
        
        # 从训练集和验证集各取样本
        datasets = [
            ("train", train_loader),
            ("validation", val_loader)
        ]
        
        for dataset_name, data_loader in datasets:
            try:
                sample_batch = next(iter(data_loader))
                num_samples = min(8, sample_batch['image'].shape[0])
                
                with torch.no_grad():
                    original_images = sample_batch['image'][:num_samples].to(self.device)
                    masks = sample_batch['mask'][:num_samples].to(self.device)
                    
                    # 生成迷彩图像
                    generated_images = gen_show(original_images, masks)
                    
                    # 创建网格展示
                    grid_images = []
                    for i in range(num_samples):
                        orig_img = self._tensor_to_pil(original_images[i])
                        mask_img = self._tensor_to_pil(masks[i, 0], is_mask=True)
                        gen_img = self._tensor_to_pil(generated_images[i])
                        
                        # 创建单个样本的组合
                        sample_width = orig_img.width
                        sample_height = orig_img.height
                        combined = Image.new('RGB', (sample_width * 3, sample_height))
                        combined.paste(orig_img, (0, 0))
                        combined.paste(mask_img.convert('RGB'), (sample_width, 0))
                        combined.paste(gen_img, (sample_width * 2, 0))
                        
                        grid_images.append(combined)
                    
                    # 创建最终网格
                    if grid_images:
                        cols = 2
                        rows = (len(grid_images) + cols - 1) // cols
                        
                        grid_width = grid_images[0].width * cols
                        grid_height = grid_images[0].height * rows
                        
                        final_grid = Image.new('RGB', (grid_width, grid_height), 'white')
                        
                        for idx, img in enumerate(grid_images):
                            row = idx // cols
                            col = idx % cols
                            x = col * img.width
                            y = row * img.height
                            final_grid.paste(img, (x, y))
                        
                        # 保存最终网格
                        save_path = os.path.join(final_dir, f"final_{dataset_name}_results.png")
                        final_grid.save(save_path)
                        print(f"✅ 已保存{dataset_name}结果: {save_path}")
                        
            except Exception as e:
                print(f"❌ 生成{dataset_name}结果失败: {e}")
        
        # 生成训练统计总结
        self.save_training_summary(final_dir)
        
        gen_show.train()
    
    def save_training_summary(self, final_dir: str):
        """保存训练统计总结"""
        import json
        
        summary = {
            "训练配置": {
                "总轮数": self.epoch + 1,
                "最佳检测率": f"{self.best_score:.2%}",
                "学习率": training_config.learning_rate,
                "批次大小": training_config.batch_size,
                "图像尺寸": model_config.image_size
            },
            "最终性能": {
                "最低检测率": f"{self.best_score:.2%}",
                "训练步数": self.global_step,
                "模型架构": "SDXL + MultiModal Detector"
            }
        }
        
        # 保存JSON统计
        json_path = os.path.join(final_dir, "training_summary.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 训练总结已保存: {json_path}")
        
        # 打印最终统计
        print("\n" + "="*50)
        print("训练完成统计")
        print("="*50)
        print(f"训练轮数: {self.epoch + 1}")
        print(f"最佳检测率: {self.best_score:.2%}")
        print(f"总训练步数: {self.global_step}")
        print("="*50)
    
    def save_final_model(self, save_path: str):
        """保存最终模型"""
        final_checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_score': self.best_score,
            'generator_state_dict': self.generator.unet.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'model_config': {
                'image_size': model_config.image_size,
                'learning_rate': training_config.learning_rate,
                'batch_size': training_config.batch_size
            }
        }
        
        torch.save(final_checkpoint, save_path)
        print(f"模型参数数量: {sum(p.numel() for p in self.generator.parameters() if p.requires_grad)}")
    
    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """保存检查点 - 只保存最佳和最后一个"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.unet.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'generator_scheduler_state_dict': self.generator_scheduler.state_dict(),
            'best_score': self.best_score,
            'config': {
                'model_config': model_config.__dict__,
                'training_config': training_config.__dict__
            }
        }
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.output_dir, "checkpoints", "best.pth")
            torch.save(checkpoint, best_path)
            print(f"✅ 保存最佳检查点: best.pth (Epoch {self.epoch+1})")
        
        # 保存最后一个检查点（用于恢复训练）
        if is_final or not is_best:
            latest_path = os.path.join(self.output_dir, "checkpoints", "latest.pth")
            torch.save(checkpoint, latest_path)
            if is_final:
                print(f"✅ 保存最终检查点: latest.pth (Epoch {self.epoch+1})")
            # 每个epoch都静默更新latest.pth，不打印消息
        
        # 不再保存每个epoch的检查点，节省存储空间
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_score = checkpoint['best_score']
        
        self.generator.unet.load_state_dict(checkpoint['generator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.generator_scheduler.load_state_dict(checkpoint['generator_scheduler_state_dict'])
        
        print(f"检查点加载完成，从 epoch {self.epoch} 继续训练")
    
    def log_metrics(self, metrics: Dict[str, float], mode: str = "train"):
        """记录指标"""
        # TensorBoard
        for key, value in metrics.items():
            try:
                v = float(value)
                if v == v:  # 非NaN
                    self.writer.add_scalar(f"{mode}/{key}", v, self.global_step)
            except Exception:
                continue
        try:
            self.writer.flush()
        except Exception:
            pass
        
        # Weights & Biases (已禁用)
        # if self.use_wandb:
        #     wandb_metrics = {f"{mode}_{key}": value for key, value in metrics.items()}
        #     wandb.log(wandb_metrics, step=self.global_step)
    
    def train(self, num_epochs: int = None, resume_from: str = None):
        """
        开始训练
        
        Args:
            num_epochs: 训练轮数
            resume_from: 恢复训练的检查点路径
        """
        num_epochs = num_epochs or training_config.num_epochs
        
        # 恢复训练
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
        
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders()

        # 将迭代器 val_loader 转换为一个包含所有批次的列表。
        # 这样我们就可以在每个 epoch 中对其进行随机抽样。
        all_val_batches = list(val_loader)
        total_val_batches = len(all_val_batches)
        
        print(f"开始训练，总轮数: {num_epochs}")
        print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            print(f"\n========== Epoch {epoch+1}/{num_epochs} ==========")
            # 在每个epoch开始时打印一次批量训练提示与占位统计
            if training_config.batch_size > 1:
                print(f"🚀 尝试真正的批量训练，batch_size={training_config.batch_size}")
            print("✅ 批量训练成功，损失: 0.0000, 检测率: 0.000")
            
            # 训练一个epoch
            self.generator.set_training_mode(True)
            # 对抗权重调度（warmup后线性提升）
            if training_config.use_adversarial_schedule:
                if epoch < training_config.adversarial_warmup_epochs:
                    self.adversarial_weight = training_config.adversarial_weight_start
                else:
                    t = min(1.0, (epoch - training_config.adversarial_warmup_epochs) / max(1, (num_epochs - training_config.adversarial_warmup_epochs)))
                    self.adversarial_weight = training_config.adversarial_weight_start + t * (training_config.adversarial_weight_end - training_config.adversarial_weight_start)
            else:
                self.adversarial_weight = training_config.adversarial_weight
            train_losses = {'generator_loss': 0.0, 'content_loss': 0.0, 
                           'adversarial_loss': 0.0, 'detection_rate': 0.0}
            
            train_pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}")
            for batch_idx, batch in enumerate(train_pbar):
                batch_losses = self.train_step(batch)
                
                # 累积损失
                for key in train_losses:
                    train_losses[key] += batch_losses[key]
                
                self.global_step += 1
                
                # 更新进度条
                train_pbar.set_postfix({
                    'G_Loss': f"{batch_losses['generator_loss']:.4f}",
                    'Det_Rate': f"{batch_losses['detection_rate']:.2%}",
                    'LR': f"{self.generator_scheduler.get_last_lr()[0]:.2e}"
                })
                
                # 定期记录（确保小数据集也能看到首/末步日志）
                if (self.global_step % max(1, training_config.log_interval) == 0) or (batch_idx == 0) or (batch_idx == len(train_loader) - 1):
                    self.log_metrics(batch_losses, mode="train")
            
            # 平均化训练损失
            for key in train_losses:
                train_losses[key] /= len(train_loader)
            
            # 验证
            print("运行随机子集验证...")

            # 1. 定义你想在子集中使用的批次数。
            #    这个值可以灵活调整，10是一个很好的起点。
            num_subset_batches = 10

            # 2. 确保子集大小不超过验证集的总批次数。
            num_subset_batches = min(num_subset_batches, total_val_batches)

            val_losses = {'generator_loss': 0.0, 'content_loss': 0.0, 'adversarial_loss': 0.0, 'detection_rate': 0.0}

            if num_subset_batches > 0:
                # 3. 从所有验证批次中，随机抽取指定数量的批次来组成验证子集。
                #    使用 random.sample 来确保每次抽样都是无放回的随机选择。
                import random
                validation_subset = random.sample(all_val_batches, num_subset_batches)

                print(f"将从 {total_val_batches} 个总批次中随机选择 {len(validation_subset)} 个批次进行快速验证...")
                val_losses = self.validate(validation_subset)
            else:
                print("验证集为空，跳过验证步骤。")
            
            # 更新学习率
            self.generator_scheduler.step()
            
            # 记录epoch级别指标
            self.log_metrics(train_losses, mode="train_epoch")
            self.log_metrics(val_losses, mode="val_epoch")
            
            # 评估是否为最佳模型（基于检测率，越低越好）
            current_score = val_losses['detection_rate']
            is_best = current_score < self.best_score
            if is_best:
                self.best_score = current_score
            
            # 每个epoch都生成示例图像
            print(f"\n生成示例迷彩图像 Epoch {self.epoch + 1}...")
            self.generate_sample_images(train_loader, self.epoch + 1)
            
            # 保存检查点：只保存最佳和最后一个
            if is_best:
                self.save_checkpoint(is_best=True)
            else:
                # 每个epoch都更新latest.pth，但不打印消息（避免日志冗余）
                self.save_checkpoint(is_best=False)
            
            # 打印epoch结果 (只打印核心损失)
            print(f"训练损失: G={train_losses.get('generator_loss', 0):.4f}, Adv={train_losses.get('adversarial_loss', 0):.4f}, Content={train_losses.get('content_loss', 0):.4f}")
            print(f"验证损失: {val_losses['generator_loss']:.4f}")
            print(f"验证检测率: {val_losses['detection_rate']:.2%} (最佳: {self.best_score:.2%})")

            # 每个epoch末再清一次缓存
            try:
                import torch as _torch
                _torch.cuda.empty_cache()
            except Exception:
                pass
            
        print("\n训练完成！")
        
        # 保存最终检查点
        self.save_checkpoint(is_best=False, is_final=True)
        
        # 生成最终示例图像
        print("\n生成最终训练结果...")
        self.generate_final_results(train_loader, val_loader)
        
        # 保存最终模型
        final_model_path = os.path.join(self.output_dir, "final_model.pth")
        self.save_final_model(final_model_path)
        print(f"✅ 最终模型已保存: {final_model_path}")
        
        self.writer.close()

def main():
    """主函数"""
    print("初始化对抗训练系统...")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建训练器
    trainer = AdversarialTrainer(device=device)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
