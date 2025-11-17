# models/ablation_models.py - 完整修复的消融实验模型定义
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from nets.light_cnn_sim import LightCNN_29Layers_v2
from models.sgpd_net import (
    PDSLRM, StructureFeatureExtractor, SGCLFA,
    SubCenterArcFaceLoss, FGDLossHead, CenterLoss, DomainAdaptationLoss,
    window_partition, window_reverse
)


class SGPDNet_NoPDSLRM(nn.Module):
    """消融实验: 禁用PDSLRM模块"""

    def __init__(self, num_classes, latent_channels=4, structure_dim=256,
                 lightcnn_embedding_dim=512, window_size=7, **kwargs):
        super(SGPDNet_NoPDSLRM, self).__init__()
        self.num_classes = num_classes
        self.structure_dim = structure_dim
        self.lightcnn_embedding_dim = lightcnn_embedding_dim
        self.window_size = window_size

        # VAE (保持不变)
        try:
            self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
            self.vae_scaling_factor = self.vae.config.scaling_factor
            for param in self.vae.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"警告: 无法加载VAE模型: {e}")
            self.vae = None
            self.vae_scaling_factor = 0.18215

        # 直接从原始潜变量提取结构特征（跳过PDSLRM）
        self.structure_feature_extractor = StructureFeatureExtractor(
            latent_channels=latent_channels, feature_dim=structure_dim
        )

        # LightCNN
        self.lightcnn = LightCNN_29Layers_v2(num_classes=num_classes)

        # SGCLFA和其他组件保持不变
        channels_after_group1 = 96
        channels_after_group2 = 192

        self.film_param_generator_c1 = nn.Sequential(
            nn.Linear(structure_dim, structure_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(structure_dim // 2, channels_after_group1 * 2),
            nn.Tanh()
        )
        self.film_param_generator_c2 = nn.Sequential(
            nn.Linear(structure_dim, structure_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(structure_dim // 2, channels_after_group2 * 2),
            nn.Tanh()
        )

        self.sgclfa1 = SGCLFA(channels_after_group1, structure_dim, window_size, 0)
        self.sgclfa2 = SGCLFA(channels_after_group2, structure_dim, window_size, window_size // 2)

        # 跨层聚合
        self.cross_layer_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.cross_layer_proj = nn.Conv2d(channels_after_group1, channels_after_group2, kernel_size=1, bias=False)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels_after_group2 * 2, channels_after_group2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_after_group2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels_after_group2, channels_after_group2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels_after_group2),
            nn.ReLU(inplace=True)
        )

        # 损失函数
        self.auxiliary_head = FGDLossHead(structure_dim, num_classes)
        self.arcface_loss_calc = SubCenterArcFaceLoss(lightcnn_embedding_dim, num_classes)
        self.center_loss = CenterLoss(num_classes, lightcnn_embedding_dim, use_gpu=torch.cuda.is_available())

        # 修改LightCNN
        self._modify_lightcnn_forward()

        # 占位符
        self.fg = None
        self.fd = None
        self.rec_img = None
        self.zs = None
        self.inversion_criterion = nn.MSELoss(reduction='mean')

    def _modify_lightcnn_forward(self):
        """修改LightCNN，保持跨层聚合"""
        original_forward = self.lightcnn.forward
        self.lightcnn.original_forward = original_forward
        sgpd_self = self

        def modified_forward(instance, x_input_vaenorm, fg=None, subject_mask=None):
            x = (x_input_vaenorm + 1.0) / 2.0

            x = instance.conv1(x)
            x = instance.eca1(x)
            x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
            x = instance.block1(x)
            x_group1_out = instance.group1(x)

            # SGCLFA 1
            if fg is not None:
                mask_for_layer1 = None
                if subject_mask is not None:
                    mask_for_layer1 = F.interpolate(subject_mask, size=x_group1_out.shape[2:], mode='nearest')
                film_params1 = sgpd_self.film_param_generator_c1(fg)
                x_guided1 = sgpd_self.sgclfa1(x_group1_out, film_params1, fg, mask_for_layer1)
            else:
                x_guided1 = x_group1_out

            feature_from_layer1 = x_guided1
            x = instance.eca2(x_guided1)
            x_pooled_after_eca2 = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

            x = instance.block2(x_pooled_after_eca2)
            x_group2_out = instance.group2(x)

            # SGCLFA 2
            if fg is not None:
                mask_for_layer2 = None
                if subject_mask is not None:
                    mask_for_layer2 = F.interpolate(subject_mask, size=x_group2_out.shape[2:], mode='nearest')
                film_params2 = sgpd_self.film_param_generator_c2(fg)
                x_guided2 = sgpd_self.sgclfa2(x_group2_out, film_params2, fg, mask_for_layer2)
            else:
                x_guided2 = x_group2_out

            # 跨层聚合
            feature_from_layer1_pooled = sgpd_self.cross_layer_pool(feature_from_layer1)
            feature_from_layer1_proj = sgpd_self.cross_layer_proj(feature_from_layer1_pooled)
            x_concatenated = torch.cat([x_guided2, feature_from_layer1_proj], dim=1)
            x_fused = sgpd_self.fusion_conv(x_concatenated)

            # 剩余层
            x = instance.eca3(x_fused)
            x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
            x = instance.block3(x)
            x = instance.group3(x)
            x = instance.block4(x)
            x = instance.group4(x)
            x = instance.eca4(x)
            x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

            x = x.view(x.size(0), -1)
            fc_out = instance.fc(x)
            feat_embedding = instance.features(F.dropout(fc_out, training=instance.training))

            return feat_embedding

        self.lightcnn.forward = lambda x, fg=None, subject_mask=None: modified_forward(self.lightcnn, x, fg,
                                                                                       subject_mask)

    def _perform_latent_inversion(self, image_norm_minus_1_1):
        """简化的潜变量反演"""
        if self.vae is None:
            # 如果没有VAE，返回虚拟潜变量
            B, C, H, W = image_norm_minus_1_1.shape
            z0 = torch.randn(B, 4, H // 8, W // 8, device=image_norm_minus_1_1.device)
            return z0, None

        with torch.no_grad():
            latent_dist = self.vae.encode(image_norm_minus_1_1).latent_dist
            z0 = latent_dist.mean * self.vae_scaling_factor
        return z0, None

    def forward(self, img1, img2=None, subject_mask=None, label=None, mode="train"):
        if mode == "train":
            z0, _ = self._perform_latent_inversion(img1)
            # 直接使用z0，跳过PDSLRM
            self.zs = z0
            self.fg = self.structure_feature_extractor(self.zs)
            self.fd = self.lightcnn(img1, fg=self.fg, subject_mask=subject_mask)

            if self.vae is not None:
                # 移除no_grad以允许重建损失梯度回传
                self.rec_img = self.vae.decode(self.zs / self.vae_scaling_factor).sample
            else:
                self.rec_img = img1  # 虚拟重建

            return self.arcface_loss_calc(self.fd, label)

        elif mode == "eval_logits":
            z0, _ = self._perform_latent_inversion(img1)
            zs = z0  # 跳过PDSLRM
            fg = self.structure_feature_extractor(zs)
            fd = self.lightcnn(img1, fg=fg, subject_mask=subject_mask)
            return self.arcface_loss_calc(fd, label)

        elif mode == "predict":
            z0, _ = self._perform_latent_inversion(img1)
            zs = z0
            fg = self.structure_feature_extractor(zs)
            fd = self.lightcnn(img1, fg=fg, subject_mask=subject_mask)
            return F.normalize(fd)

    def compute_auxiliary_loss(self, label):
        if self.fg is None:
            raise RuntimeError("Run forward(train) first.")
        logits = self.auxiliary_head(self.fg)
        return F.cross_entropy(logits, label)

    def compute_reconstruction_loss(self, orig_img):
        if self.rec_img is None:
            raise RuntimeError("Run forward(train) first.")
        return F.l1_loss(self.rec_img, orig_img)

    def compute_center_loss(self, label):
        if self.fd is None:
            raise RuntimeError("Run forward(train) first.")
        return self.center_loss(self.fd, label)


class SimpleSGCLFA_NoFiLM(nn.Module):
    """简化的SGCLFA，不使用FiLM参数生成器"""

    def __init__(self, in_channels, structure_dim, window_size=7, shift_size=0, reduction=8):
        super(SimpleSGCLFA_NoFiLM, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.shift_size = shift_size
        self.reduction_ratio = max(1, reduction)
        reduced_channels = max(1, in_channels // self.reduction_ratio)

        # 结构特征处理
        self.prior_mlp = nn.Sequential(
            nn.Linear(structure_dim, in_channels),
            nn.LayerNorm(in_channels)
        )

        # 注意力机制
        self.query_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.gamma_sp = nn.Parameter(torch.zeros(1))

        # 通道注意力
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // self.reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // self.reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        self.gamma_ch = nn.Parameter(torch.zeros(1))

    def forward(self, x, fg_for_direct_guidance, subject_mask=None):
        batch_size, C, H, W = x.size()

        # 应用掩码
        if subject_mask is not None:
            if subject_mask.size(2) != H or subject_mask.size(3) != W:
                subject_mask = F.interpolate(subject_mask, size=(H, W), mode='nearest')
            x = x * (subject_mask * 1.0 + (1 - subject_mask) * 0.7)

        # 窗口处理（简化版）
        if self.shift_size > 0:
            x_shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            x_shifted = x

        x_windows, original_dims = window_partition(x_shifted, self.window_size)
        x_windows = x_windows.permute(0, 3, 1, 2).contiguous()

        # 窗口特征处理（无FiLM调制）
        num_windows = x_windows.size(0) // batch_size
        x_windows_flat = x_windows.reshape(-1, C, self.window_size * self.window_size).permute(0, 2, 1)

        # 直接使用原始特征，无FiLM调制
        x_windows_modulated = x_windows_flat
        x_windows_modulated = x_windows_modulated.permute(0, 2, 1).view(-1, C, self.window_size, self.window_size)

        # 注意力处理
        fg_prime = self.prior_mlp(fg_for_direct_guidance)
        fg_prime_expanded = fg_prime.unsqueeze(1).repeat_interleave(repeats=num_windows, dim=0)

        # 空间注意力
        q_sp = self.query_conv(x_windows_modulated).reshape(-1, C // self.reduction_ratio,
                                                            self.window_size * self.window_size).permute(0, 2, 1)
        fg_prime_conv = fg_prime_expanded.view(-1, C, 1, 1)
        k_sp = self.key_conv(fg_prime_conv).view(-1, C // self.reduction_ratio, 1).permute(0, 2, 1)

        attention_scores = torch.bmm(q_sp, k_sp.transpose(1, 2))
        scale = math.sqrt(self.query_conv.out_channels)
        attention_map = torch.sigmoid(attention_scores / scale)

        x_windows_sp = x_windows_modulated.reshape(-1, C, self.window_size * self.window_size).permute(0, 2, 1)
        x_windows_sp = x_windows_sp * (1 + self.gamma_sp * attention_map)

        # 通道注意力
        avg_pool = x_windows_sp.mean(dim=1)
        channel_input = torch.cat([avg_pool, fg_prime_expanded.squeeze(1)], dim=1)
        channel_att = self.channel_mlp(channel_input)

        x_windows_ch = x_windows_sp * (1 + self.gamma_ch * channel_att.unsqueeze(1))

        # 窗口反向拼接
        x_windows_out = x_windows_ch.permute(0, 2, 1).reshape(-1, C, self.window_size, self.window_size)
        x_windows_out_hwc = x_windows_out.permute(0, 2, 3, 1)
        x_out = window_reverse(x_windows_out_hwc, self.window_size, original_dims)

        if self.shift_size > 0:
            x_out = torch.roll(x_out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return x_out


class SGPDNet_NoFiLM(nn.Module):
    """消融实验: 禁用FiLM参数生成器"""

    def __init__(self, num_classes, latent_channels=4, structure_dim=256,
                 lightcnn_embedding_dim=512, window_size=7, **kwargs):
        super(SGPDNet_NoFiLM, self).__init__()
        self.num_classes = num_classes
        self.structure_dim = structure_dim
        self.lightcnn_embedding_dim = lightcnn_embedding_dim
        self.window_size = window_size

        # VAE
        try:
            self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
            self.vae_scaling_factor = self.vae.config.scaling_factor
            for param in self.vae.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"警告: 无法加载VAE模型: {e}")
            self.vae = None
            self.vae_scaling_factor = 0.18215

        # 生成分支
        self.pdslrm = PDSLRM(latent_channels=latent_channels)
        self.structure_feature_extractor = StructureFeatureExtractor(latent_channels, structure_dim)

        # LightCNN
        self.lightcnn = LightCNN_29Layers_v2(num_classes=num_classes)

        # 简化的SGCLFA（无FiLM调制）
        channels_after_group1 = 96
        channels_after_group2 = 192

        self.simple_sgclfa1 = SimpleSGCLFA_NoFiLM(channels_after_group1, structure_dim, window_size, 0)
        self.simple_sgclfa2 = SimpleSGCLFA_NoFiLM(channels_after_group2, structure_dim, window_size, window_size // 2)

        # 跨层聚合
        self.cross_layer_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.cross_layer_proj = nn.Conv2d(channels_after_group1, channels_after_group2, kernel_size=1, bias=False)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels_after_group2 * 2, channels_after_group2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_after_group2),
            nn.ReLU(inplace=True)
        )

        # 损失函数
        self.auxiliary_head = FGDLossHead(structure_dim, num_classes)
        self.arcface_loss_calc = SubCenterArcFaceLoss(lightcnn_embedding_dim, num_classes)
        self.center_loss = CenterLoss(num_classes, lightcnn_embedding_dim, use_gpu=torch.cuda.is_available())

        # 修改LightCNN
        self._modify_lightcnn_forward()

        # 占位符
        self.fg = None
        self.fd = None
        self.rec_img = None
        self.zs = None
        self.inversion_criterion = nn.MSELoss(reduction='mean')

    def _modify_lightcnn_forward(self):
        """修改LightCNN，使用无FiLM的SGCLFA"""
        original_forward = self.lightcnn.forward
        self.lightcnn.original_forward = original_forward
        sgpd_self = self

        def modified_forward(instance, x_input_vaenorm, fg=None, subject_mask=None):
            x = (x_input_vaenorm + 1.0) / 2.0

            x = instance.conv1(x)
            x = instance.eca1(x)
            x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
            x = instance.block1(x)
            x_group1_out = instance.group1(x)

            # 简化SGCLFA 1 (无FiLM)
            if fg is not None:
                mask_for_layer1 = None
                if subject_mask is not None:
                    mask_for_layer1 = F.interpolate(subject_mask, size=x_group1_out.shape[2:], mode='nearest')
                x_guided1 = sgpd_self.simple_sgclfa1(x_group1_out, fg, mask_for_layer1)
            else:
                x_guided1 = x_group1_out

            feature_from_layer1 = x_guided1
            x = instance.eca2(x_guided1)
            x_pooled_after_eca2 = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

            x = instance.block2(x_pooled_after_eca2)
            x_group2_out = instance.group2(x)

            # 简化SGCLFA 2 (无FiLM)
            if fg is not None:
                mask_for_layer2 = None
                if subject_mask is not None:
                    mask_for_layer2 = F.interpolate(subject_mask, size=x_group2_out.shape[2:], mode='nearest')
                x_guided2 = sgpd_self.simple_sgclfa2(x_group2_out, fg, mask_for_layer2)
            else:
                x_guided2 = x_group2_out

            # 跨层聚合
            feature_from_layer1_pooled = sgpd_self.cross_layer_pool(feature_from_layer1)
            feature_from_layer1_proj = sgpd_self.cross_layer_proj(feature_from_layer1_pooled)
            x_concatenated = torch.cat([x_guided2, feature_from_layer1_proj], dim=1)
            x_fused = sgpd_self.fusion_conv(x_concatenated)

            x = instance.eca3(x_fused)
            x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
            x = instance.block3(x)
            x = instance.group3(x)
            x = instance.block4(x)
            x = instance.group4(x)
            x = instance.eca4(x)
            x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

            x = x.view(x.size(0), -1)
            fc_out = instance.fc(x)
            feat_embedding = instance.features(F.dropout(fc_out, training=instance.training))

            return feat_embedding

        self.lightcnn.forward = lambda x, fg=None, subject_mask=None: modified_forward(self.lightcnn, x, fg,
                                                                                       subject_mask)

    def _perform_latent_inversion(self, image_norm_minus_1_1):
        """简化的潜变量反演"""
        if self.vae is None:
            B, C, H, W = image_norm_minus_1_1.shape
            z0 = torch.randn(B, 4, H // 8, W // 8, device=image_norm_minus_1_1.device)
            return z0, None

        with torch.no_grad():
            latent_dist = self.vae.encode(image_norm_minus_1_1).latent_dist
            z0 = latent_dist.mean * self.vae_scaling_factor
        return z0, None

    def forward(self, img1, img2=None, subject_mask=None, label=None, mode="train"):
        if mode == "train":
            z0, _ = self._perform_latent_inversion(img1)
            self.zs = self.pdslrm(z0)
            self.fg = self.structure_feature_extractor(self.zs)
            self.fd = self.lightcnn(img1, fg=self.fg, subject_mask=subject_mask)

            if self.vae is not None:
                # 移除no_grad以允许重建损失梯度回传
                self.rec_img = self.vae.decode(self.zs / self.vae_scaling_factor).sample
            else:
                self.rec_img = img1

            return self.arcface_loss_calc(self.fd, label)

        elif mode == "eval_logits":
            z0, _ = self._perform_latent_inversion(img1)
            zs = self.pdslrm(z0)
            fg = self.structure_feature_extractor(zs)
            fd = self.lightcnn(img1, fg=fg, subject_mask=subject_mask)
            return self.arcface_loss_calc(fd, label)

        elif mode == "predict":
            z0, _ = self._perform_latent_inversion(img1)
            zs = self.pdslrm(z0)
            fg = self.structure_feature_extractor(zs)
            fd = self.lightcnn(img1, fg=fg, subject_mask=subject_mask)
            return F.normalize(fd)

    def compute_auxiliary_loss(self, label):
        if self.fg is None:
            raise RuntimeError("Run forward(train) first.")
        logits = self.auxiliary_head(self.fg)
        return F.cross_entropy(logits, label)

    def compute_reconstruction_loss(self, orig_img):
        if self.rec_img is None:
            raise RuntimeError("Run forward(train) first.")
        return F.l1_loss(self.rec_img, orig_img)

    def compute_center_loss(self, label):
        if self.fd is None:
            raise RuntimeError("Run forward(train) first.")
        return self.center_loss(self.fd, label)


class SGPDNet_NoStructureGuidance(nn.Module):
    """消融实验: 禁用结构引导（PDSLRM + 结构特征提取器）"""

    def __init__(self, num_classes, lightcnn_embedding_dim=512, **kwargs):
        super(SGPDNet_NoStructureGuidance, self).__init__()
        self.num_classes = num_classes
        self.lightcnn_embedding_dim = lightcnn_embedding_dim

        # 只保留判别分支
        self.lightcnn = LightCNN_29Layers_v2(num_classes=num_classes)

        # 损失函数
        self.arcface_loss_calc = SubCenterArcFaceLoss(lightcnn_embedding_dim, num_classes)
        self.center_loss = CenterLoss(num_classes, lightcnn_embedding_dim, use_gpu=torch.cuda.is_available())

        # 占位符
        self.fd = None

    def forward(self, img1, img2=None, subject_mask=None, label=None, mode="train"):
        # 直接使用原始LightCNN
        self.fd = self.lightcnn((img1 + 1.0) / 2.0)  # 转换到[0,1]

        if mode == "train" or mode == "eval_logits":
            return self.arcface_loss_calc(self.fd, label)
        elif mode == "predict":
            return F.normalize(self.fd)

    def compute_center_loss(self, label):
        if self.fd is None:
            raise RuntimeError("Run forward first.")
        return self.center_loss(self.fd, label)

    def compute_auxiliary_loss(self, label):
        # 没有结构特征，返回零损失
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def compute_reconstruction_loss(self, orig_img):
        # 没有重建，返回零损失
        return torch.tensor(0.0, device=next(self.parameters()).device)


class SimpleLightCNNBaseline(nn.Module):
    """简单CNN基线（仅LightCNN + 标准分类损失）"""

    def __init__(self, num_classes, lightcnn_embedding_dim=512, **kwargs):
        super(SimpleLightCNNBaseline, self).__init__()
        self.num_classes = num_classes
        self.lightcnn_embedding_dim = lightcnn_embedding_dim

        # 原始LightCNN
        self.lightcnn = LightCNN_29Layers_v2(num_classes=num_classes)

        # 简单分类头
        self.classifier = nn.Linear(lightcnn_embedding_dim, num_classes)

        # 占位符
        self.fd = None

    def forward(self, img1, img2=None, subject_mask=None, label=None, mode="train"):
        # 转换输入范围
        img_input = (img1 + 1.0) / 2.0  # [-1,1] -> [0,1]

        # 提取特征
        self.fd = self.lightcnn(img_input)

        # 分类
        logits = self.classifier(self.fd)
        return logits

    def compute_center_loss(self, label):
        # 简单基线不使用center loss
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def compute_auxiliary_loss(self, label):
        # 简单基线不使用辅助损失
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def compute_reconstruction_loss(self, orig_img):
        # 简单基线不使用重建损失
        return torch.tensor(0.0, device=next(self.parameters()).device)


class SGPDNet_NoSGCLFA(nn.Module):
    """消融实验: 禁用SGCLFA注意力机制"""

    def __init__(self, num_classes, latent_channels=4, structure_dim=256,
                 lightcnn_embedding_dim=512, **kwargs):
        super(SGPDNet_NoSGCLFA, self).__init__()
        self.num_classes = num_classes
        self.structure_dim = structure_dim
        self.lightcnn_embedding_dim = lightcnn_embedding_dim

        # VAE
        try:
            self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
            self.vae_scaling_factor = self.vae.config.scaling_factor
            for param in self.vae.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"警告: 无法加载VAE模型: {e}")
            self.vae = None
            self.vae_scaling_factor = 0.18215

        # 生成分支
        self.pdslrm = PDSLRM(latent_channels=latent_channels)
        self.structure_feature_extractor = StructureFeatureExtractor(latent_channels, structure_dim)

        # LightCNN (不修改，使用原始前向传播)
        self.lightcnn = LightCNN_29Layers_v2(num_classes=num_classes)

        # 损失函数
        self.auxiliary_head = FGDLossHead(structure_dim, num_classes)
        self.arcface_loss_calc = SubCenterArcFaceLoss(lightcnn_embedding_dim, num_classes)
        self.center_loss = CenterLoss(num_classes, lightcnn_embedding_dim, use_gpu=torch.cuda.is_available())

        # 占位符
        self.fg = None
        self.fd = None
        self.rec_img = None
        self.zs = None
        self.inversion_criterion = nn.MSELoss(reduction='mean')

    def _perform_latent_inversion(self, image_norm_minus_1_1):
        """简化的潜变量反演"""
        if self.vae is None:
            B, C, H, W = image_norm_minus_1_1.shape
            z0 = torch.randn(B, 4, H // 8, W // 8, device=image_norm_minus_1_1.device)
            return z0, None

        with torch.no_grad():
            latent_dist = self.vae.encode(image_norm_minus_1_1).latent_dist
            z0 = latent_dist.mean * self.vae_scaling_factor
        return z0, None

    def forward(self, img1, img2=None, subject_mask=None, label=None, mode="train"):
        if mode == "train":
            # 潜变量反演
            z0, _ = self._perform_latent_inversion(img1)
            self.zs = self.pdslrm(z0)

            # 结构特征提取
            self.fg = self.structure_feature_extractor(self.zs)

            # 直接使用原始LightCNN (无SGCLFA)
            self.fd = self.lightcnn((img1 + 1.0) / 2.0)  # 转换到[0,1]

            # 重建
            if self.vae is not None:
                # 移除no_grad以允许重建损失梯度回传
                self.rec_img = self.vae.decode(self.zs / self.vae_scaling_factor).sample
            else:
                self.rec_img = img1

            return self.arcface_loss_calc(self.fd, label)

        elif mode == "eval_logits":
            z0, _ = self._perform_latent_inversion(img1)
            zs = self.pdslrm(z0)
            fg = self.structure_feature_extractor(zs)
            fd = self.lightcnn((img1 + 1.0) / 2.0)
            return self.arcface_loss_calc(fd, label)

        elif mode == "predict":
            z0, _ = self._perform_latent_inversion(img1)
            zs = self.pdslrm(z0)
            fg = self.structure_feature_extractor(zs)
            fd = self.lightcnn((img1 + 1.0) / 2.0)
            return F.normalize(fd)

    def compute_auxiliary_loss(self, label):
        if self.fg is None:
            raise RuntimeError("Run forward(train) first.")
        logits = self.auxiliary_head(self.fg)
        return F.cross_entropy(logits, label)

    def compute_reconstruction_loss(self, orig_img):
        if self.rec_img is None:
            raise RuntimeError("Run forward(train) first.")
        return F.l1_loss(self.rec_img, orig_img)

    def compute_center_loss(self, label):
        if self.fd is None:
            raise RuntimeError("Run forward(train) first.")
        return self.center_loss(self.fd, label)


class SimpleWindowAttention(nn.Module):
    """简化的窗口注意力（不使用窗口分割）"""

    def __init__(self, in_channels, structure_dim, reduction=8):
        super(SimpleWindowAttention, self).__init__()
        self.in_channels = in_channels
        reduced_channels = max(1, in_channels // reduction)

        self.prior_mlp = nn.Sequential(
            nn.Linear(structure_dim, in_channels),
            nn.LayerNorm(in_channels)
        )

        self.query_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.gamma_sp = nn.Parameter(torch.zeros(1))

        # 全局注意力而非窗口注意力
        self.global_attention = nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True)

    def forward(self, x, film_params, fg_for_direct_guidance, subject_mask=None):
        B, C, H, W = x.shape

        # 应用掩码
        if subject_mask is not None:
            if subject_mask.size(2) != H or subject_mask.size(3) != W:
                subject_mask = F.interpolate(subject_mask, size=(H, W), mode='nearest')
            x = x * (subject_mask * 1.0 + (1 - subject_mask) * 0.7)

        # FiLM调制
        gamma_beta = film_params.view(B, 2, C)
        gamma = gamma_beta[:, 0] * 0.3 + 1.0
        beta = gamma_beta[:, 1] * 0.05

        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # B, HW, C
        gamma = gamma.unsqueeze(1).expand(-1, H * W, -1)
        beta = beta.unsqueeze(1).expand(-1, H * W, -1)
        x_modulated = gamma * x_flat + beta

        # 全局自注意力
        attn_out, _ = self.global_attention(x_modulated, x_modulated, x_modulated)
        x_out = x_modulated + attn_out

        # 重塑回原始形状
        x_out = x_out.permute(0, 2, 1).view(B, C, H, W)

        return x_out


class SGPDNet_NoWindowAttention(nn.Module):
    """消融实验: 禁用窗口注意力，使用全局注意力"""

    def __init__(self, num_classes, latent_channels=4, structure_dim=256,
                 lightcnn_embedding_dim=512, **kwargs):
        super(SGPDNet_NoWindowAttention, self).__init__()
        self.num_classes = num_classes
        self.structure_dim = structure_dim
        self.lightcnn_embedding_dim = lightcnn_embedding_dim

        # VAE
        try:
            self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
            self.vae_scaling_factor = self.vae.config.scaling_factor
            for param in self.vae.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"警告: 无法加载VAE模型: {e}")
            self.vae = None
            self.vae_scaling_factor = 0.18215

        # 生成分支
        self.pdslrm = PDSLRM(latent_channels=latent_channels)
        self.structure_feature_extractor = StructureFeatureExtractor(latent_channels, structure_dim)

        # LightCNN
        self.lightcnn = LightCNN_29Layers_v2(num_classes=num_classes)

        # 简化的注意力机制（替代SGCLFA）
        channels_after_group1 = 96
        channels_after_group2 = 192

        self.film_param_generator_c1 = nn.Sequential(
            nn.Linear(structure_dim, channels_after_group1 * 2),
            nn.Tanh()
        )
        self.film_param_generator_c2 = nn.Sequential(
            nn.Linear(structure_dim, channels_after_group2 * 2),
            nn.Tanh()
        )

        # 使用简化的全局注意力替代窗口注意力
        self.simple_attention1 = SimpleWindowAttention(channels_after_group1, structure_dim)
        self.simple_attention2 = SimpleWindowAttention(channels_after_group2, structure_dim)

        # 跨层聚合
        self.cross_layer_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.cross_layer_proj = nn.Conv2d(channels_after_group1, channels_after_group2, kernel_size=1, bias=False)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels_after_group2 * 2, channels_after_group2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_after_group2),
            nn.ReLU(inplace=True)
        )

        # 损失函数
        self.auxiliary_head = FGDLossHead(structure_dim, num_classes)
        self.arcface_loss_calc = SubCenterArcFaceLoss(lightcnn_embedding_dim, num_classes)
        self.center_loss = CenterLoss(num_classes, lightcnn_embedding_dim, use_gpu=torch.cuda.is_available())

        # 修改LightCNN
        self._modify_lightcnn_forward()

        # 占位符
        self.fg = None
        self.fd = None
        self.rec_img = None
        self.zs = None
        self.inversion_criterion = nn.MSELoss(reduction='mean')

    def _modify_lightcnn_forward(self):
        """修改LightCNN使用简化注意力"""
        original_forward = self.lightcnn.forward
        self.lightcnn.original_forward = original_forward
        sgpd_self = self

        def modified_forward(instance, x_input_vaenorm, fg=None, subject_mask=None):
            x = (x_input_vaenorm + 1.0) / 2.0

            x = instance.conv1(x)
            x = instance.eca1(x)
            x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
            x = instance.block1(x)
            x_group1_out = instance.group1(x)

            # 简化注意力 1
            if fg is not None:
                mask_for_layer1 = None
                if subject_mask is not None:
                    mask_for_layer1 = F.interpolate(subject_mask, size=x_group1_out.shape[2:], mode='nearest')
                film_params1 = sgpd_self.film_param_generator_c1(fg)
                x_guided1 = sgpd_self.simple_attention1(x_group1_out, film_params1, fg, mask_for_layer1)
            else:
                x_guided1 = x_group1_out

            feature_from_layer1 = x_guided1
            x = instance.eca2(x_guided1)
            x_pooled_after_eca2 = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

            x = instance.block2(x_pooled_after_eca2)
            x_group2_out = instance.group2(x)

            # 简化注意力 2
            if fg is not None:
                mask_for_layer2 = None
                if subject_mask is not None:
                    mask_for_layer2 = F.interpolate(subject_mask, size=x_group2_out.shape[2:], mode='nearest')
                film_params2 = sgpd_self.film_param_generator_c2(fg)
                x_guided2 = sgpd_self.simple_attention2(x_group2_out, film_params2, fg, mask_for_layer2)
            else:
                x_guided2 = x_group2_out

            # 跨层聚合
            feature_from_layer1_pooled = sgpd_self.cross_layer_pool(feature_from_layer1)
            feature_from_layer1_proj = sgpd_self.cross_layer_proj(feature_from_layer1_pooled)
            x_concatenated = torch.cat([x_guided2, feature_from_layer1_proj], dim=1)
            x_fused = sgpd_self.fusion_conv(x_concatenated)

            # 剩余层
            x = instance.eca3(x_fused)
            x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
            x = instance.block3(x)
            x = instance.group3(x)
            x = instance.block4(x)
            x = instance.group4(x)
            x = instance.eca4(x)
            x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

            x = x.view(x.size(0), -1)
            fc_out = instance.fc(x)
            feat_embedding = instance.features(F.dropout(fc_out, training=instance.training))

            return feat_embedding

        self.lightcnn.forward = lambda x, fg=None, subject_mask=None: modified_forward(self.lightcnn, x, fg,
                                                                                       subject_mask)

    def _perform_latent_inversion(self, image_norm_minus_1_1):
        """简化的潜变量反演"""
        if self.vae is None:
            B, C, H, W = image_norm_minus_1_1.shape
            z0 = torch.randn(B, 4, H // 8, W // 8, device=image_norm_minus_1_1.device)
            return z0, None

        with torch.no_grad():
            latent_dist = self.vae.encode(image_norm_minus_1_1).latent_dist
            z0 = latent_dist.mean * self.vae_scaling_factor
        return z0, None

    def forward(self, img1, img2=None, subject_mask=None, label=None, mode="train"):
        if mode == "train":
            z0, _ = self._perform_latent_inversion(img1)
            self.zs = self.pdslrm(z0)
            self.fg = self.structure_feature_extractor(self.zs)
            self.fd = self.lightcnn(img1, fg=self.fg, subject_mask=subject_mask)

            if self.vae is not None:
                # 移除no_grad以允许重建损失梯度回传
                self.rec_img = self.vae.decode(self.zs / self.vae_scaling_factor).sample
            else:
                self.rec_img = img1

            return self.arcface_loss_calc(self.fd, label)

        elif mode == "eval_logits":
            z0, _ = self._perform_latent_inversion(img1)
            zs = self.pdslrm(z0)
            fg = self.structure_feature_extractor(zs)
            fd = self.lightcnn(img1, fg=fg, subject_mask=subject_mask)
            return self.arcface_loss_calc(fd, label)

        elif mode == "predict":
            z0, _ = self._perform_latent_inversion(img1)
            zs = self.pdslrm(z0)
            fg = self.structure_feature_extractor(zs)
            fd = self.lightcnn(img1, fg=fg, subject_mask=subject_mask)
            return F.normalize(fd)

    def compute_auxiliary_loss(self, label):
        if self.fg is None:
            raise RuntimeError("Run forward(train) first.")
        logits = self.auxiliary_head(self.fg)
        return F.cross_entropy(logits, label)

    def compute_reconstruction_loss(self, orig_img):
        if self.rec_img is None:
            raise RuntimeError("Run forward(train) first.")
        return F.l1_loss(self.rec_img, orig_img)

    def compute_center_loss(self, label):
        if self.fd is None:
            raise RuntimeError("Run forward(train) first.")
        return self.center_loss(self.fd, label)


class SGPDNet_NoCrossLayer(nn.Module):
    """消融实验: 禁用跨层聚合"""

    def __init__(self, num_classes, latent_channels=4, structure_dim=256,
                 lightcnn_embedding_dim=512, window_size=7, **kwargs):
        super(SGPDNet_NoCrossLayer, self).__init__()
        self.num_classes = num_classes
        self.structure_dim = structure_dim
        self.lightcnn_embedding_dim = lightcnn_embedding_dim
        self.window_size = window_size

        # VAE
        try:
            self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
            self.vae_scaling_factor = self.vae.config.scaling_factor
            for param in self.vae.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"警告: 无法加载VAE模型: {e}")
            self.vae = None
            self.vae_scaling_factor = 0.18215

        # 生成分支
        self.pdslrm = PDSLRM(latent_channels=latent_channels)
        self.structure_feature_extractor = StructureFeatureExtractor(latent_channels, structure_dim)

        # LightCNN
        self.lightcnn = LightCNN_29Layers_v2(num_classes=num_classes)

        # SGCLFA
        channels_after_group1 = 96
        channels_after_group2 = 192

        self.film_param_generator_c1 = nn.Sequential(
            nn.Linear(structure_dim, structure_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(structure_dim // 2, channels_after_group1 * 2),
            nn.Tanh()
        )
        self.film_param_generator_c2 = nn.Sequential(
            nn.Linear(structure_dim, structure_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(structure_dim // 2, channels_after_group2 * 2),
            nn.Tanh()
        )

        self.sgclfa1 = SGCLFA(channels_after_group1, structure_dim, window_size, 0)
        self.sgclfa2 = SGCLFA(channels_after_group2, structure_dim, window_size, window_size // 2)

        # 不使用跨层聚合，每层独立处理

        # 损失函数
        self.auxiliary_head = FGDLossHead(structure_dim, num_classes)
        self.arcface_loss_calc = SubCenterArcFaceLoss(lightcnn_embedding_dim, num_classes)
        self.center_loss = CenterLoss(num_classes, lightcnn_embedding_dim, use_gpu=torch.cuda.is_available())

        # 修改LightCNN（无跨层聚合）
        self._modify_lightcnn_forward()

        # 占位符
        self.fg = None
        self.fd = None
        self.rec_img = None
        self.zs = None
        self.inversion_criterion = nn.MSELoss(reduction='mean')

    def _modify_lightcnn_forward(self):
        """修改LightCNN，移除跨层聚合"""
        original_forward = self.lightcnn.forward
        self.lightcnn.original_forward = original_forward
        sgpd_self = self

        def modified_forward(instance, x_input_vaenorm, fg=None, subject_mask=None):
            x = (x_input_vaenorm + 1.0) / 2.0

            x = instance.conv1(x)
            x = instance.eca1(x)
            x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
            x = instance.block1(x)
            x_group1_out = instance.group1(x)

            # SGCLFA 1
            if fg is not None:
                mask_for_layer1 = None
                if subject_mask is not None:
                    mask_for_layer1 = F.interpolate(subject_mask, size=x_group1_out.shape[2:], mode='nearest')
                film_params1 = sgpd_self.film_param_generator_c1(fg)
                x_guided1 = sgpd_self.sgclfa1(x_group1_out, film_params1, fg, mask_for_layer1)
            else:
                x_guided1 = x_group1_out

            # 直接继续，不保存layer1特征用于跨层聚合
            x = instance.eca2(x_guided1)
            x_pooled_after_eca2 = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

            x = instance.block2(x_pooled_after_eca2)
            x_group2_out = instance.group2(x)

            # SGCLFA 2
            if fg is not None:
                mask_for_layer2 = None
                if subject_mask is not None:
                    mask_for_layer2 = F.interpolate(subject_mask, size=x_group2_out.shape[2:], mode='nearest')
                film_params2 = sgpd_self.film_param_generator_c2(fg)
                x_guided2 = sgpd_self.sgclfa2(x_group2_out, film_params2, fg, mask_for_layer2)
            else:
                x_guided2 = x_group2_out

            # 直接使用x_guided2，无跨层聚合
            x = instance.eca3(x_guided2)
            x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
            x = instance.block3(x)
            x = instance.group3(x)
            x = instance.block4(x)
            x = instance.group4(x)
            x = instance.eca4(x)
            x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

            x = x.view(x.size(0), -1)
            fc_out = instance.fc(x)
            feat_embedding = instance.features(F.dropout(fc_out, training=instance.training))

            return feat_embedding

        self.lightcnn.forward = lambda x, fg=None, subject_mask=None: modified_forward(self.lightcnn, x, fg,
                                                                                       subject_mask)

    def _perform_latent_inversion(self, image_norm_minus_1_1):
        """简化的潜变量反演"""
        if self.vae is None:
            B, C, H, W = image_norm_minus_1_1.shape
            z0 = torch.randn(B, 4, H // 8, W // 8, device=image_norm_minus_1_1.device)
            return z0, None

        with torch.no_grad():
            latent_dist = self.vae.encode(image_norm_minus_1_1).latent_dist
            z0 = latent_dist.mean * self.vae_scaling_factor
        return z0, None

    def forward(self, img1, img2=None, subject_mask=None, label=None, mode="train"):
        if mode == "train":
            z0, _ = self._perform_latent_inversion(img1)
            self.zs = self.pdslrm(z0)
            self.fg = self.structure_feature_extractor(self.zs)
            self.fd = self.lightcnn(img1, fg=self.fg, subject_mask=subject_mask)

            if self.vae is not None:
                # 移除no_grad以允许重建损失梯度回传
                self.rec_img = self.vae.decode(self.zs / self.vae_scaling_factor).sample
            else:
                self.rec_img = img1

            return self.arcface_loss_calc(self.fd, label)

        elif mode == "eval_logits":
            z0, _ = self._perform_latent_inversion(img1)
            zs = self.pdslrm(z0)
            fg = self.structure_feature_extractor(zs)
            fd = self.lightcnn(img1, fg=fg, subject_mask=subject_mask)
            return self.arcface_loss_calc(fd, label)

        elif mode == "predict":
            z0, _ = self._perform_latent_inversion(img1)
            zs = self.pdslrm(z0)
            fg = self.structure_feature_extractor(zs)
            fd = self.lightcnn(img1, fg=fg, subject_mask=subject_mask)
            return F.normalize(fd)

    def compute_auxiliary_loss(self, label):
        if self.fg is None:
            raise RuntimeError("Run forward(train) first.")
        logits = self.auxiliary_head(self.fg)
        return F.cross_entropy(logits, label)

    def compute_reconstruction_loss(self, orig_img):
        if self.rec_img is None:
            raise RuntimeError("Run forward(train) first.")
        return F.l1_loss(self.rec_img, orig_img)

    def compute_center_loss(self, label):
        if self.fd is None:
            raise RuntimeError("Run forward(train) first.")
        return self.center_loss(self.fd, label)