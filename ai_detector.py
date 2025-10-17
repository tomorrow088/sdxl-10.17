"""
AI检测器模块 - 多模态迷彩检测系统
包含YOLOv8目标检测、CLIP语义理解、EfficientNet分类等多个检测器
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import clip
import cv2
from ultralytics import YOLO
from torchvision import transforms
import timm
from typing import List, Dict, Tuple, Optional, Union

from config import model_config, eval_config, training_config


class YOLODetector(nn.Module):
    """YOLO目标检测器"""
    
    def __init__(self, model_name: str = "yolov8x.pt", device: str = "cuda"):
        """
        初始化YOLO检测器
        
        Args:
            model_name: YOLO模型名称
            device: 计算设备
        """
        super().__init__()
        self.device = device
        
        # 尝试使用本地YOLO模型
        local_yolo_path = "models/yolov8x.pt"
        
        if os.path.exists(local_yolo_path):
            try:
                print(f"尝试加载本地YOLO模型: {local_yolo_path}")
                self.model = YOLO(local_yolo_path)
                print("✅ 成功加载本地YOLO模型")
            except Exception as e:
                raise RuntimeError(f"本地YOLO模型加载失败: {e}. 请确保文件完整: {local_yolo_path}")
        else:
            raise FileNotFoundError(f"未找到本地YOLO权重: {local_yolo_path}。请将 yolov8x.pt 放到 models/ 目录下。")
        
        self.model.to(device)
        
        # 检测阈值
        self.confidence_threshold = eval_config.detection_confidence
        self.iou_threshold = eval_config.detection_iou_threshold
        
        print(f"YOLO检测器初始化完成，使用模型: {model_name}")
    
    def forward(self, images: Union[torch.Tensor, List[Image.Image]], masks: Optional[Union[torch.Tensor, List[Image.Image]]] = None) -> Dict:
        """
        前向推理
        
        Args:
            images: 输入图像
            masks: 可选掩码（只检测掩码区域）。支持[B,1,H,W]张量或PIL列表
            
        Returns:
            检测结果字典
        """
        if isinstance(images, torch.Tensor):
            # 将张量转换为PIL图像列表
            images = self._tensor_to_pil_list(images)
        
        mask_list: Optional[List[Image.Image]] = None
        if masks is not None:
            if isinstance(masks, torch.Tensor):
                mask_list = self._tensor_mask_to_pil_list(masks)
            else:
                mask_list = [m.convert('L') for m in masks]
        
        # 若给定掩码，则仅对掩码区域进行检测
        if mask_list is not None:
            inference_images: List[Image.Image] = []
            for img, m in zip(images, mask_list):
                black = Image.new('RGB', img.size, (0, 0, 0))
                masked_img = Image.composite(img, black, m)
                inference_images.append(masked_img)
        else:
            inference_images = images
        
        # YOLO推理
        results = self.model(
            inference_images,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # 解析结果
        detection_results = {
            'boxes': [],         # 检测框坐标 [x1, y1, x2, y2]
            'scores': [],        # 检测置信度 [0.0-1.0]
            'classes': [],       # 检测类别ID [0=person, 1=bicycle, ...]
            'detected': []       # 是否检测到 [True/False]
        }
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                detection_results['boxes'].append(boxes)
                detection_results['scores'].append(scores)
                detection_results['classes'].append(classes)
                detection_results['detected'].append(len(boxes) > 0)
            else:
                detection_results['boxes'].append(np.array([]))     
                detection_results['scores'].append(np.array([]))
                detection_results['classes'].append(np.array([]))
                detection_results['detected'].append(False)
        
        return detection_results
    
    def _tensor_to_pil_list(self, tensor: torch.Tensor) -> List[Image.Image]:
        """将张量转换为PIL图像列表（梯度安全版本）"""
        # 分离梯度以避免计算图问题
        tensor = tensor.detach()
        
        # 反归一化
        tensor = (tensor * 0.5) + 0.5
        tensor = torch.clamp(tensor, 0, 1)
        
        pil_images = []
        for i in range(tensor.shape[0]):
            img = tensor[i]
            # [C,H,W]，若是3通道按RGB；若是1通道则挤掉通道维
            if img.shape[0] == 3:
                arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(arr, mode='RGB'))
            elif img.shape[0] == 1:
                arr = (img[0].cpu().numpy() * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(arr, mode='L'))
            else:
                # 兜底：尝试按RGB
                arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(arr))
        
        return pil_images

    def _tensor_mask_to_pil_list(self, tensor: torch.Tensor) -> List[Image.Image]:
        """掩码专用：将 [B,1,H,W] 或 [B,H,W] 张量转为PIL L图像列表"""
        t = tensor.detach()
        if t.dim() == 3:
            # [B,H,W] -> [B,1,H,W]
            t = t.unsqueeze(1)
        # 确保为[0,1]
        if t.min() < 0 or t.max() > 1:
            t = t.clamp(0, 1)
        masks: List[Image.Image] = []
        for i in range(t.shape[0]):
            arr = (t[i, 0].cpu().numpy() * 255).astype(np.uint8)
            masks.append(Image.fromarray(arr, mode='L'))
        return masks
    
    def compute_detection_loss(self, images: torch.Tensor, target_detected: bool = False) -> torch.Tensor:
        """
        计算检测损失（用于对抗训练）
        
        Args:
            images: 输入图像张量
            target_detected: 目标检测状态（False表示希望不被检测到）
            
        Returns:
            检测损失
        """
        # 使用真实AI检测器计算损失（梯度安全版本）
        try:
            # 分离梯度进行AI检测，但保留检测结果用于损失计算
            with torch.no_grad():
                results = self.forward(images)
            
            batch_size = len(results['detected'])
            
            # 计算基于真实AI检测的损失
            detection_scores = []
            for i in range(batch_size):
                detected = results['detected'][i]
                confidence = results['confidence'][i] if 'confidence' in results else 0.5
                
                # 将检测结果转换为连续值用于梯度计算
                if detected:
                    score = confidence  # 检测到，置信度越高越"被检测"
                else:
                    score = 0.0  # 未检测到
                
                detection_scores.append(score)
            
            # 转换为张量
            detection_tensor = torch.tensor(detection_scores, device=images.device, dtype=torch.float32)
            
            # 根据目标计算损失
            if target_detected:
                # 希望被检测到：最大化检测分数
                loss = -detection_tensor.mean()
            else:
                # 希望不被检测到（迷彩目标）：最小化检测分数
                loss = detection_tensor.mean()
            
            # 为了保持梯度流，添加一个与输入图像相关的微小项
            # 这确保损失对输入图像有依赖性
            image_variance = torch.var(images, dim=[1,2,3]).mean()
            loss = loss + 0.001 * image_variance  # 微小的图像相关项
            
            return loss
            
        except Exception as e:
            print(f"AI检测损失计算失败，使用备用方案: {e}")
            # 备用方案：基于图像统计的简单损失
            image_mean = torch.mean(images, dim=[1,2,3])
            if target_detected:
                return -torch.var(image_mean)  # 希望被检测：增加图像差异
            else:
                return torch.var(image_mean)   # 希望不被检测：减少图像差异

class CLIPDetector(nn.Module):
    """CLIP语义检测器"""
    
    def __init__(self, model_name: str = "ViT-L/14", device: str = "cuda"):
        """
        初始化CLIP检测器
        
        Args:
            model_name: CLIP模型名称
            device: 计算设备
        """
        super().__init__()
        self.device = device
        
        # 统一的tokenize函数占位
        self.tokenize = None
        
        # 1) 优先尝试 openai/CLIP 接口（若环境中的 clip 拥有 load/tokenize）
        use_openai_clip = hasattr(clip, "load") and hasattr(clip, "tokenize")
        local_clip_path = "models/ViT-L-14.pt"
        if use_openai_clip:
            try:
                if os.path.exists(local_clip_path):
                    print(f"尝试加载本地CLIP模型(OpenAI后端): {local_clip_path}")
                    self.model, self.preprocess = clip.load(model_name, device=device, download_root="models")
                else:
                    raise FileNotFoundError(f"未找到本地CLIP权重: {local_clip_path}。请将 ViT-L-14.pt 放到 models/ 目录下，或使用 open_clip 回退。")
                self.tokenize = clip.tokenize
                print("✅ 成功加载CLIP(OpenAI)")
            except Exception as e:
                print(f"⚠️ OpenAI CLIP 加载失败: {e}")
                use_openai_clip = False
        
        # 2) 回退到 open_clip（无需从GitHub安装，可pip安装 open_clip_torch；支持本地 .pt 权重）
        if not use_openai_clip:
            try:
                import open_clip
                print("尝试加载CLIP(OpenCLIP后端)...")
                if os.path.exists(local_clip_path):
                    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=local_clip_path)
                else:
                    raise FileNotFoundError(f"未找到本地CLIP权重: {local_clip_path}。请将 ViT-L-14.pt 放到 models/ 目录下。")
                self.model = model.to(device)
                self.preprocess = preprocess
                self.tokenize = open_clip.tokenize
                print("✅ 成功加载CLIP(OpenCLIP)")
            except Exception as e:
                raise RuntimeError("无法加载CLIP：既没有可用的 openai/CLIP，也无法回退到 open_clip。请安装 open_clip_torch 或修正 clip 包。") from e
        
        # 定义检测文本
        self.detection_texts = [
            "a person hiding in camouflage",
            "military camouflage pattern", 
            "person in military uniform",
            "camouflaged object",
            "hidden person",
            "person blending with background",
            "military personnel in camouflage",
            "concealed person"
        ]
        
        # 不再需要固定的normal_texts
        # self.normal_texts = [ ... ]
        
        # 只预编码detection_texts
        with torch.no_grad():
            camouflage_tokens = self.tokenize(self.detection_texts).to(device)
            self.camouflage_features = self.model.encode_text(camouflage_tokens)
            self.camouflage_features /= self.camouflage_features.norm(dim=-1, keepdim=True)
        
        print(f"CLIP检测器初始化完成，使用模型: {model_name}")
    
    def forward(self, images: Union[torch.Tensor, List[Image.Image]]) -> Dict:
        """
        前向推理 (注意：此版本不接收背景图，主要用于无背景参考的快速评估)
        
        Args:
            images: 输入图像
            
        Returns:
            检测结果字典
        """
        if isinstance(images, torch.Tensor):
            images = self._tensor_to_pil_list(images)
        
        # 预处理图像
        image_tensors = []
        for img in images:
            image_tensors.append(self.preprocess(img))
        
        image_batch = torch.stack(image_tensors).to(self.device)
        
        # 编码图像
        with torch.no_grad():
            image_features = self.model.encode_image(image_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # 计算与迷彩的相似度
        camouflage_similarities = torch.matmul(image_features, self.camouflage_features.T)
        max_camouflage_sim = camouflage_similarities.max(dim=-1)[0]
        
        # 由于没有背景参考，这里的检测逻辑简化
        # 实际的对抗损失将使用更复杂的逻辑
        detection_threshold = 0.25 # 设定一个绝对阈值
        detected = max_camouflage_sim > detection_threshold
        
        return {
            'camouflage_similarity': max_camouflage_sim,
            'normal_similarity': torch.zeros_like(max_camouflage_sim), # 占位
            'detected': detected,
            'confidence': max_camouflage_sim
        }
    
    def _tensor_to_pil_list(self, tensor: torch.Tensor) -> List[Image.Image]:
        """将张量转换为PIL图像列表（梯度安全版本）"""
        # 分离梯度以避免计算图问题
        tensor = tensor.detach()
        
        # 反归一化
        tensor = (tensor * 0.5) + 0.5
        tensor = torch.clamp(tensor, 0, 1)
        
        pil_images = []
        for i in range(tensor.shape[0]):
            img_array = (tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_array))
        
        return pil_images
    
    def compute_detection_loss(self, images: torch.Tensor, target_detected: bool = False) -> torch.Tensor:
        """计算CLIP检测损失"""
        # 注意：该方法仅用于评估与记录，不参与反传
        # 不使用detach，以便在需要时保留可微路径；但调用该方法的上层通常包裹no_grad()
        results = self.forward(images)
        
        confidence = results['confidence']
        
        if target_detected:
            # 希望被检测到：最大化迷彩相似度
            loss = -confidence.mean()
        else:
            # 希望不被检测到：最小化迷彩相似度
            loss = torch.relu(confidence + 0.1).mean()  # ReLU确保只惩罚正的confidence
        
        return loss

    def compute_detection_loss_diff(self, 
                                  images: torch.Tensor, 
                                  background_images: torch.Tensor,
                                  target_detected: bool = False) -> torch.Tensor:
        """
        可微分的CLIP检测损失，使用真实的背景图像作为基准。
        """
        def encode_image_for_clip(img_tensor):
            # 将图像从[-1,1]映射到[0,1]
            x = (img_tensor * 0.5) + 0.5
            # 调整到CLIP输入分辨率
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            # 按CLIP规范化
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
            clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
            x = (x - clip_mean) / clip_std
            # 图像编码（保持可微）
            features = self.model.encode_image(x)
            return features / features.norm(dim=-1, keepdim=True)

        # 编码生成的图像和背景图像
        generated_features = encode_image_for_clip(images)
        background_features = encode_image_for_clip(background_images)

        # 计算生成图与“迷彩”文本的相似度
        camouflage_similarities = torch.matmul(generated_features, self.camouflage_features.T)
        max_camouflage_sim = camouflage_similarities.max(dim=-1)[0]

        # 计算生成图与“真实背景”的相似度 (逐元素)
        # background_features.unsqueeze(1) -> [B, 1, D]
        # generated_features.unsqueeze(2) -> [B, D, 1]
        # bmm -> [B, 1, 1] -> squeeze -> [B]
        background_similarity = torch.bmm(generated_features.unsqueeze(1), background_features.unsqueeze(2)).squeeze()

        # 对抗目标：最小化(迷彩相似度 - 背景相似度)
        # 即，让迷彩相似度尽可能低，背景相似度尽可能高
        confidence = max_camouflage_sim - background_similarity
        
        if target_detected:
            # 这个分支在对抗训练中通常不用
            loss = -confidence.mean()
        else:
            # 我们希望confidence是负数（背景相似度 > 迷彩相似度）
            # 所以我们惩罚所有正的confidence值
            loss = torch.relu(confidence).mean()
        
        return loss

    def compute_adversarial_loss(self, 
                                 images: torch.Tensor, 
                                 background_images: torch.Tensor,
                                 target_detected: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """统一接口：返回 (loss, details)"""
        loss = self.compute_detection_loss_diff(images, background_images, target_detected=target_detected)
        details = {
            'clip_semantic_loss': loss.detach()
        }
        return loss, details

class EfficientNetClassifier(nn.Module):
    """EfficientNet分类器"""
    
    def __init__(self, 
                 model_name: str = "efficientnet_b4",  # 改用更常见的b4
                 num_classes: int = 2,  # 0: 无迷彩, 1: 有迷彩
                 pretrained: bool = True,
                 device: str = "cuda"):
        """
        初始化EfficientNet分类器
        
        Args:
            model_name: EfficientNet模型名称
            num_classes: 分类数量
            pretrained: 是否使用预训练权重
            device: 计算设备
        """
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        
        # 统一使用timm加载模型，避免transformers的torch.load版本限制问题
        local_model_path = "models/efficientnet-b4"
        model_name_to_load = model_config.efficientnet_model or "efficientnet_b4"
        model_loaded = False
        
        # 离线模式下禁用预训练权重下载
        offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        use_pretrained = pretrained and (not offline)
        if offline and pretrained:
            print("🌐 已启用离线模式(HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE)，禁用timm预训练权重下载，改为随机初始化。")

        try:
            print(f"尝试使用timm加载模型: {model_name_to_load}")
            self.backbone = timm.create_model(
                model_name_to_load, 
                pretrained=use_pretrained,
                num_classes=num_classes
            )
            init_msg = "(预训练权重)" if use_pretrained else "(随机初始化)"
            print(f"✅ 成功加载 {model_name_to_load} {init_msg}")
            model_loaded = True
            self.use_transformers = False # 统一设置为False
            
        except Exception as e:
            print(f"⚠️ 使用timm加载 {model_name_to_load} 失败: {e}")
            # 如果加载失败，可以添加备选方案，例如ResNet
            try:
                print("尝试加载备选模型: resnet50")
                self.backbone = timm.create_model(
                    "resnet50",
                    pretrained=use_pretrained,
                    num_classes=num_classes
                )
                init_msg = "(预训练权重)" if use_pretrained else "(随机初始化)"
                print(f"✅ 成功加载 resnet50 {init_msg}")
                model_loaded = True
                self.use_transformers = False
            except Exception as fallback_e:
                print(f"⚠️ 备选模型resnet50加载失败: {fallback_e}")

        if not model_loaded:
            # 最后的备选方案：使用 ResNet18 随机初始化
            print("⚠️  所有预训练模型失败，使用 ResNet18 随机初始化")
            self.backbone = timm.create_model(
                "resnet18", 
                pretrained=False,
                num_classes=num_classes
            )
            self.use_transformers = False
        
        self.backbone.to(device)
        
        # 检查并报告模型状态
        self._report_model_status()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 减小输入尺寸节省显存
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"EfficientNet分类器初始化完成，使用模型: {model_name}")
    
    def _report_model_status(self):
        """报告模型状态和配置信息"""
        try:
            if hasattr(self, 'use_transformers') and self.use_transformers:
                # Transformers模型
                config = self.backbone.config
                print(f"📋 模型架构: {config.architectures[0] if hasattr(config, 'architectures') else 'EfficientNet'}")
                print(f"🎯 分类类别: {config.num_labels}")
                print(f"🔧 模型状态: 预训练主干 + 自定义分类器")
                
                # 检查哪些层被重新初始化
                classifier_layer = None
                for name, module in self.backbone.named_modules():
                    if 'classifier' in name and hasattr(module, 'weight'):
                        classifier_layer = name
                        break
                
                if classifier_layer and self.num_classes != 1000:
                    print(f"🔄 重新初始化层: {classifier_layer}")
                    print(f"💡 这是Transfer Learning的正常行为")
            else:
                # TIMM模型
                print(f"📋 模型架构: {self.backbone.__class__.__name__}")
                print(f"🎯 分类类别: {self.num_classes}")
                print(f"🔧 模型状态: TIMM预训练模型")
                
        except Exception as e:
            print(f"📊 模型状态检查跳过: {e}")
    
    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Dict:
        """
        前向推理
        
        Args:
            images: 输入图像张量 [B, C, H, W]
            
        Returns:
            分类结果字典
        """
        # 预处理图像
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 反归一化后重新归一化到ImageNet标准
        images = (images * 0.5) + 0.5  # 反归一化到[0,1]
        
        if hasattr(self, 'use_transformers') and self.use_transformers:
            # 使用transformers模型
            # 转换为PIL图像列表给processor处理（梯度安全版本）
            pil_images = []
            for i in range(images.shape[0]):
                img_array = (images[i].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_array))
            
            # 使用processor处理
            inputs = self.processor(pil_images, return_tensors="pt").to(self.device)
            
            # 前向推理
            with torch.no_grad():
                outputs = self.backbone(**inputs)
                logits = outputs.logits
        else:
            # 使用timm模型
            images = self.transform(images)
            logits = self.backbone(images)
        
        probs = torch.softmax(logits, dim=-1)
        
        # 获取预测
        predicted_classes = torch.argmax(probs, dim=-1)
        confidence = torch.max(probs, dim=-1)[0]
        
        return {
            'logits': logits,
            'probabilities': probs,
            'predicted_classes': predicted_classes,
            'confidence': confidence,
            'detected': predicted_classes == 1  # 类别1表示检测到迷彩
        }
    
    def compute_detection_loss(self, images: torch.Tensor, target_detected: bool = False) -> torch.Tensor:
        """计算分类检测损失"""
        results = self.forward(images)
        
        # 构建目标标签
        batch_size = images.shape[0]
        target_labels = torch.full((batch_size,), 1 if target_detected else 0, 
                                 dtype=torch.long, device=images.device)
        
        # 交叉熵损失
        loss = F.cross_entropy(results['logits'], target_labels)
        
        return loss

    def compute_detection_loss_diff(self, images: torch.Tensor, target_detected: bool = False) -> torch.Tensor:
        """可微分分类损失：纯张量路径，不经PIL与no_grad。"""
        x = images
        # 调整分辨率
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # 将[-1,1]映射到[0,1]
        x = (x * 0.5) + 0.5
        # 归一化到ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        x = (x - mean) / std

        if hasattr(self, 'use_transformers') and self.use_transformers:
            outputs = self.backbone(pixel_values=x)
            logits = outputs.logits
        else:
            logits = self.backbone(x)

        batch_size = images.shape[0]
        target_labels = torch.full((batch_size,), 1 if target_detected else 0, dtype=torch.long, device=images.device)
        loss = F.cross_entropy(logits, target_labels)
        return loss

    def compute_adversarial_loss(self, images: torch.Tensor, target_detected: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """统一接口：返回 (loss, details)"""
        loss = self.compute_detection_loss_diff(images, target_detected=target_detected)
        details = {
            'efficientnet_ce': loss.detach()
        }
        return loss, details

class MultiModalDetector(nn.Module):
    """多模态检测器集成"""
    
    def __init__(self, 
                 use_yolo: bool = True,   # 您有本地yolov8x.pt，可以启用
                 use_clip: bool = True,   # 启用本地CLIP模型
                 use_efficientnet: bool = True,  # 使用本地EfficientNet-B4
                 device: str = "cuda"):
        """
        初始化多模态检测器
        
        Args:
            use_yolo: 是否使用YOLO检测器
            use_clip: 是否使用CLIP检测器
            use_efficientnet: 是否使用EfficientNet分类器
            device: 计算设备
        """
        super().__init__()
        self.device = device
        self.generator = None # 添加一个生成器的引用
        
        # 初始化各个检测器
        self.detectors = {}
        
        if use_yolo:
            self.detectors['yolo'] = YOLODetector(device=device)
        
        if use_clip:
            self.detectors['clip'] = CLIPDetector(device=device)
            # 冻结到eval，确保BN/Dropout稳定
            self.detectors['clip'].eval()
        
        if use_efficientnet:
            self.detectors['efficientnet'] = EfficientNetClassifier(device=device)
            self.detectors['efficientnet'].eval()
        
        # 检测器权重（以YOLO为主，可从config接入）
        self.detector_weights = {
            'yolo': training_config.yolo_branch_weight,
            'clip': training_config.clip_branch_weight,
            'efficientnet': training_config.efficientnet_branch_weight
        }
        
        print(f"多模态检测器初始化完成，启用检测器: {list(self.detectors.keys())}")
    
    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Dict:
        """
        多模态检测前向推理
        
        Args:
            images: 输入图像张量
            masks: 可选掩码，仅用于YOLO分支在掩码区域内检测
            
        Returns:
            综合检测结果
        """
        results = {}
        detection_scores = []
        
        # 运行各个检测器
        for detector_name, detector in self.detectors.items():
            # 将掩码传递给YOLO分支，其它分支忽略掩码
            if detector_name == 'yolo':
                detector_result = detector(images, masks)
            else:
                detector_result = detector(images)
            results[detector_name] = detector_result
            
            # 提取检测分数
            if detector_name == 'yolo':
                # YOLO: 基于是否有检测框 + 置信度的软分数
                soft_scores = []
                for i, detected in enumerate(detector_result['detected']):
                    if detected:
                        # 取该图像最大置信度作为强度
                        confs = detector_result['scores'][i]
                        max_conf = float(confs.max()) if len(confs) > 0 else 1.0
                        soft_scores.append(max(0.5, max_conf))  # 至少0.5，表示被检测
                    else:
                        soft_scores.append(0.0)
                scores = torch.tensor(soft_scores, device=images.device)
            elif detector_name == 'clip':
                # CLIP: 基于置信度
                scores = torch.sigmoid(detector_result['confidence'])
            elif detector_name == 'efficientnet':
                # EfficientNet: 基于概率
                scores = detector_result['probabilities'][:, 1]  # 迷彩类别的概率
            
            detection_scores.append(scores * self.detector_weights[detector_name])
        
        # 融合检测结果
        if detection_scores:
            combined_scores = torch.stack(detection_scores, dim=0).sum(dim=0)
            combined_detected = combined_scores > 0.5
        else:
            combined_scores = torch.zeros(images.shape[0], device=images.device)
            combined_detected = torch.zeros(images.shape[0], dtype=torch.bool, device=images.device)
        
        results['combined'] = {
            'scores': combined_scores,
            'detected': combined_detected,
            'confidence': combined_scores
        }
        
        return results
    
    def compute_adversarial_loss(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算对抗损失（目标是不被检测到）
        
        Args:
            images: 输入图像张量
            
        Returns:
            对抗损失
        """
        # 从输入图像中提取背景，用于CLIP损失
        background_images = None
        if masks is not None:
            background_images = images * (1.0 - masks.float())

        # 1) YOLO 置信度损失 (方案A)
        yolo_conf_loss = torch.tensor(0.0, device=images.device)
        yolo_details = {}
        if 'yolo' in self.detectors:
            try:
                # 直接调用YOLO模型进行推理。虽然其中有NMS等不可微操作，
                # 但置信度分数通常仍能传递一部分稀疏的梯度。
                yolo_results = self.detectors['yolo'].model(images, verbose=False)
                
                all_confidences = []
                for res in yolo_results:
                    if res.boxes is not None and len(res.boxes.conf) > 0:
                        all_confidences.append(res.boxes.conf)
                
                if len(all_confidences) > 0:
                    # 将所有检测到的置信度拼接起来，计算平均值作为损失
                    yolo_conf_loss = torch.cat(all_confidences).mean()
                
                yolo_details = {'yolo_conf_mean': yolo_conf_loss.detach()}

            except Exception as e:
                # 如果YOLO前向失败，打印警告但不中断训练
                print(f"⚠️ YOLO confidence loss calculation failed: {e}")

        # 2) CLIP可微损失（统一接口）
        clip_loss = torch.tensor(0.0, device=images.device)
        clip_details = {}
        if 'clip' in self.detectors and hasattr(self.detectors['clip'], 'compute_adversarial_loss') and background_images is not None:
            clip_loss, clip_details = self.detectors['clip'].compute_adversarial_loss(images, background_images, target_detected=False)
        
        # 3) EfficientNet可微损失（统一接口）
        eff_loss = torch.tensor(0.0, device=images.device)
        eff_details = {}
        if 'efficientnet' in self.detectors and hasattr(self.detectors['efficientnet'], 'compute_adversarial_loss'):
            eff_loss, eff_details = self.detectors['efficientnet'].compute_adversarial_loss(images, target_detected=False)

        # 按config权重融合
        total_loss = (
            training_config.yolo_adv_weight * yolo_conf_loss +
            training_config.clip_adv_weight * clip_loss +
            training_config.efficientnet_adv_weight * eff_loss
        )
        details = {}
        details.update(yolo_details)
        details.update({f'clip_{k}': v for k, v in clip_details.items()})
        details.update({f'eff_{k}': v for k, v in eff_details.items()})
        return total_loss, details
    
    def evaluate_detection_success(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Dict:
        """
        评估检测成功率
        
        Args:
            images: 输入图像张量
            
        Returns:
            评估结果
        """
        results = self.forward(images, masks)
        
        evaluation = {
            'total_images': images.shape[0],
            'detected_count': 0,
            'detection_rate': 0.0,
            'per_detector_results': {}
        }
        
        # 统计各检测器结果
        for detector_name in self.detectors.keys():
            detector_result = results[detector_name]
            # 将bool列表/张量安全转换
            det = detector_result['detected']
            if isinstance(det, torch.Tensor):
                detected_count = int(det.sum().item())
            else:
                detected_count = int(sum(det))
            detection_rate = detected_count / images.shape[0]
            
            evaluation['per_detector_results'][detector_name] = {
                'detected_count': detected_count,
                'detection_rate': detection_rate
            }
        
        # 综合结果
        comb = results['combined']['detected']
        if isinstance(comb, torch.Tensor):
            combined_detected = int(comb.sum().item())
        else:
            combined_detected = int(sum(comb))
        evaluation['detected_count'] = combined_detected
        evaluation['detection_rate'] = combined_detected / images.shape[0]
        
        return evaluation

if __name__ == "__main__":
    # 测试多模态检测器
    print("测试多模态AI检测器...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = MultiModalDetector(device=device)
    
    # 创建测试图像
    test_images = torch.randn(2, 3, 1024, 1024).to(device)
    test_images = test_images * 0.5 + 0.5  # 归一化到[-1,1]
    test_images = (test_images - 0.5) / 0.5
    
    # 测试检测
    print("运行检测...")
    results = detector(test_images)
    print(f"检测结果: {results['combined']['detected']}")
    
    # 测试对抗损失
    print("计算对抗损失...")
    loss = detector.compute_adversarial_loss(test_images)
    print(f"对抗损失: {loss.item()}")
    
    print("AI检测器测试完成！")
