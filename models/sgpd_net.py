# models/sgpd_net.py - 改进版本
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import time

# 从本地nets目录导入LightCNN
from nets.light_cnn_sim import LightCNN_29Layers_v2

# 导入Stable Diffusion VAE
from diffusers import AutoencoderKL

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# --- 窗口操作函数 ---
def window_partition(x, window_size):
    """将特征图划分为不重叠的窗口"""
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1)  # B, H, W, C

    # 确保H和W是window_size的整数倍
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    H_padded, W_padded = H + pad_h, W + pad_w

    # 划分窗口
    x = x.view(B, H_padded // window_size, window_size, W_padded // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (H, W, H_padded, W_padded)


def window_reverse(windows, window_size, original_dims):
    """将窗口还原为特征图"""
    H, W, H_padded, W_padded = original_dims

    # 计算批次大小
    B = int(windows.shape[0] / ((H_padded // window_size) * (W_padded // window_size)))
    C = windows.shape[-1]

    # 重构填充后的特征图
    x = windows.view(B, H_padded // window_size, W_padded // window_size,
                     window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, C)

    # 去除填充
    if H_padded != H or W_padded != W:
        x = x[:, :H, :W, :]

    x = x.permute(0, 3, 1, 2)  # B, C, H, W
    return x


# --- 改进的Center Loss ---
class CenterLoss(nn.Module):
    """Center Loss for improving generalization"""

    def __init__(self, num_classes, feat_dim, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)

        # 修正：使用现代PyTorch API
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        # 使用现代API替代过时的addmm_
        distmat = distmat + torch.addmm(torch.zeros_like(distmat), x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


# --- 改进的Domain Adaptation Loss ---
class DomainAdaptationLoss(nn.Module):
    """Domain adaptation loss to improve generalization"""

    def __init__(self, feature_dim):
        super(DomainAdaptationLoss, self).__init__()
        self.feature_dim = feature_dim

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim // 2, 2)  # 2 domains: original vs augmented
        )

        # Gradient reversal layer weight
        self.grl_weight = 1.0

    def forward(self, features, domain_labels):
        """
        Args:
            features: Feature tensor (B, feature_dim)
            domain_labels: Domain labels (B,) - 0 for original, 1 for augmented
        """
        # Apply gradient reversal
        reversed_features = GradientReversalFunction.apply(features, self.grl_weight)
        domain_pred = self.domain_classifier(reversed_features)
        domain_loss = F.cross_entropy(domain_pred, domain_labels)

        return domain_loss


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# --- PDSLRM (保持不变) ---
class PDSLRM(nn.Module):
    def __init__(self, latent_channels=4, n_refine=2):
        super(PDSLRM, self).__init__()
        self.n_refine = n_refine
        internal_channels = 2 * latent_channels
        self.encoder_conv1 = nn.Conv2d(latent_channels, internal_channels, kernel_size=3, padding=1)
        self.encoder_relu = nn.ReLU(inplace=True)
        self.encoder_conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1)
        self.layer_norm = nn.LayerNorm(internal_channels)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=internal_channels, num_heads=max(1, internal_channels // 2),
                                  batch_first=True)
            for _ in range(n_refine)
        ])
        self.group_norm = nn.GroupNorm(num_groups=max(1, internal_channels // 2), num_channels=internal_channels)
        self.decoder_conv1 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1)
        self.decoder_relu = nn.ReLU(inplace=True)
        self.decoder_conv2 = nn.Conv2d(internal_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, x):
        encoded = self.encoder_conv1(x)
        encoded = self.encoder_relu(encoded)
        encoded = self.encoder_conv2(encoded)
        batch, channels, height, width = encoded.shape
        encoded_seq = encoded.view(batch, channels, height * width).permute(0, 2, 1)
        encoded_seq = self.layer_norm(encoded_seq)
        attn_input = encoded_seq
        for layer in self.attention_layers:
            attn_output, _ = layer(attn_input, attn_input, attn_input)
            attn_input = attn_input + attn_output
        refined = attn_input.permute(0, 2, 1).view(batch, channels, height, width)
        refined = self.group_norm(refined)
        decoded = self.decoder_conv1(refined)
        decoded = self.decoder_relu(decoded)
        decoded = self.decoder_conv2(decoded)
        output = decoded + x
        return output


# --- 改进的StructureFeatureExtractor ---
class StructureFeatureExtractor(nn.Module):
    def __init__(self, latent_channels=4, feature_dim=256):
        super(StructureFeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(latent_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.Dropout(0.3)  # 添加dropout提高泛化
        )

        # 添加结构正则化层
        self.structure_regularizer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim)
        )

    def forward(self, x):
        base_features = self.feature_extractor(x)
        # 添加结构正则化
        regularized_features = self.structure_regularizer(base_features)
        # 残差连接
        final_features = base_features + 0.1 * regularized_features
        return final_features


# --- 改进的SGCLFA ---
class SGCLFA(nn.Module):
    def __init__(self, in_channels, structure_dim, window_size=7, shift_size=0, reduction=8):
        super(SGCLFA, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.shift_size = shift_size
        self.reduction_ratio = max(1, reduction)
        reduced_channels = max(1, in_channels // self.reduction_ratio)

        # 结构特征处理 - 增加泛化能力
        self.prior_mlp = nn.Sequential(
            nn.Linear(structure_dim, structure_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 添加dropout
            nn.Linear(structure_dim // 2, in_channels),
            nn.LayerNorm(in_channels)  # 添加LayerNorm
        )

        # 注意力机制
        self.query_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.gamma_sp = nn.Parameter(torch.zeros(1))

        # 通道注意力 - 改进
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // self.reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # 添加dropout
            nn.Linear(in_channels // self.reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        self.gamma_ch = nn.Parameter(torch.zeros(1))

        # 添加特征正则化
        self.feature_norm = nn.LayerNorm(in_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, film_params, fg_for_direct_guidance, subject_mask=None):
        batch_size, C, H, W = x.size()

        # 1. 更温和的掩码处理
        if subject_mask is not None:
            if subject_mask.size(2) != H or subject_mask.size(3) != W:
                subject_mask = F.interpolate(subject_mask, size=(H, W), mode='nearest')
            # 减少背景抑制强度，提高泛化能力
            x = x * (subject_mask * 1.0 + (1 - subject_mask) * 0.7)  # 从0.4改为0.7

        # 2-3. 窗口处理 (保持原逻辑)
        if self.shift_size > 0:
            x_shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            mask_shifted = torch.roll(subject_mask, shifts=(-self.shift_size, -self.shift_size),
                                      dims=(2, 3)) if subject_mask is not None else None
        else:
            x_shifted = x
            mask_shifted = subject_mask

        x_windows, original_dims = window_partition(x_shifted, self.window_size)
        x_windows = x_windows.permute(0, 3, 1, 2).contiguous()

        if mask_shifted is not None:
            mask_windows, _ = window_partition(mask_shifted, self.window_size)
            mask_windows = mask_windows.permute(0, 3, 1, 2).contiguous()
        else:
            mask_windows = None

        # 4-6. 窗口特征处理和FiLM调制 (保持原逻辑)
        num_windows = x_windows.size(0) // batch_size
        x_windows_flat = x_windows.reshape(-1, C, self.window_size * self.window_size).permute(0, 2, 1)

        if mask_windows is not None:
            mask_windows_flat = mask_windows.reshape(-1, 1, self.window_size * self.window_size).permute(0, 2, 1)
            x_windows_weighted = x_windows_flat * (mask_windows_flat * 1.0 + (1 - mask_windows_flat) * 0.7)
        else:
            x_windows_weighted = x_windows_flat

        # FiLM参数应用 (调整范围)
        gamma_beta = film_params.view(batch_size, 2, C)
        gamma = gamma_beta[:, 0] * 0.3 + 1.0  # 减小调制范围：[0.7, 1.3]
        beta = gamma_beta[:, 1] * 0.05  # 减小偏移范围：[-0.05, 0.05]

        gamma = gamma.unsqueeze(1).expand(-1, self.window_size * self.window_size, -1)
        beta = beta.unsqueeze(1).expand(-1, self.window_size * self.window_size, -1)
        gamma = gamma.repeat_interleave(repeats=num_windows, dim=0)
        beta = beta.repeat_interleave(repeats=num_windows, dim=0)

        x_windows_modulated = gamma * x_windows_weighted + beta

        # 7-11. 注意力处理 (保持原逻辑但添加正则化)
        x_windows_modulated = x_windows_modulated.permute(0, 2, 1).view(-1, C, self.window_size, self.window_size)

        fg_prime = self.prior_mlp(fg_for_direct_guidance)

        x_windows_modulated_flat = x_windows_modulated.reshape(-1, C, self.window_size * self.window_size).permute(0, 2,
                                                                                                                   1)
        fg_prime_expanded = fg_prime.unsqueeze(1).repeat_interleave(repeats=num_windows, dim=0)

        # 空间注意力
        q_sp = self.query_conv(x_windows_modulated).reshape(-1, C // self.reduction_ratio,
                                                            self.window_size * self.window_size).permute(0, 2, 1)
        fg_prime_conv = fg_prime_expanded.view(-1, C, 1, 1)
        k_sp = self.key_conv(fg_prime_conv).view(-1, C // self.reduction_ratio, 1).permute(0, 2, 1)

        attention_scores = torch.bmm(q_sp, k_sp.transpose(1, 2))
        scale = math.sqrt(self.query_conv.out_channels)
        attention_map = torch.sigmoid(attention_scores / scale)

        x_windows_sp = x_windows_modulated_flat * (1 + self.gamma_sp * attention_map)

        # 通道注意力
        avg_pool = x_windows_sp.mean(dim=1)
        channel_input = torch.cat([avg_pool, fg_prime_expanded.squeeze(1)], dim=1)
        channel_att = self.channel_mlp(channel_input)

        x_windows_ch = x_windows_sp * (1 + self.gamma_ch * channel_att.unsqueeze(1))

        # 12-13. 窗口反向拼接 (保持原逻辑)
        x_windows_out = x_windows_ch.permute(0, 2, 1).reshape(-1, C, self.window_size, self.window_size)
        x_windows_out_hwc = x_windows_out.permute(0, 2, 3, 1)
        x_out = window_reverse(x_windows_out_hwc, self.window_size, original_dims)

        if self.shift_size > 0:
            x_out = torch.roll(x_out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return x_out


# --- 改进的SubCenterArcFaceLoss ---
class SubCenterArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, K=3, s=16.0, m=0.2):  # 降低参数激进程度
        super(SubCenterArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features * K, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embedding, label):
        # 添加特征正则化
        embedding_normalized = F.normalize(embedding)
        weight_normalized = F.normalize(self.weight)

        cosine_all_subs = F.linear(embedding_normalized, weight_normalized)
        cosine_reshaped = cosine_all_subs.view(-1, self.out_features, self.K)

        cosine_hardest_k, _ = torch.max(cosine_reshaped, dim=2)

        target_class_cosine = torch.gather(cosine_hardest_k, 1, label.view(-1, 1).long()).squeeze(1)

        eps = 1e-5
        target_class_cosine_clamped = torch.clamp(target_class_cosine, -1.0 + eps, 1.0 - eps)

        target_class_angle = torch.acos(target_class_cosine_clamped)
        target_class_angle_with_margin = target_class_angle + self.m

        cosine_with_margin = torch.cos(target_class_angle_with_margin)

        if torch.isnan(cosine_with_margin).any():
            print("警告: 在cosine_with_margin中检测到NaN值，使用回退方案")
            cosine_with_margin = target_class_cosine_clamped

        final_target_cosine = torch.where(
            target_class_cosine > self.threshold,
            cosine_with_margin,
            target_class_cosine - self.mm
        )

        output_logits = cosine_hardest_k.clone()
        output_logits.scatter_(1, label.view(-1, 1).long(), final_target_cosine.view(-1, 1))
        output_logits *= self.s

        if torch.isnan(output_logits).any():
            print("警告: 在输出logits中检测到NaN值")
            safe_logits = cosine_hardest_k * self.s
            return safe_logits

        return output_logits


# --- Auxiliary Loss Head ---
class FGDLossHead(nn.Module):
    def __init__(self, fg_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(fg_dim, fg_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 添加dropout
            nn.Linear(fg_dim // 2, num_classes)
        )

        # 权重初始化
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fg):
        return self.classifier(fg)


# --- 改进的SGPDNet ---
class SGPDNet(nn.Module):
    def __init__(self, num_classes, latent_channels=4, structure_dim=256, lightcnn_embedding_dim=512,
                 pretrained=False, sub_centers_k=3, arcface_s=16.0, arcface_m=0.2,  # 降低参数
                 latent_inversion_iters=50, latent_inversion_lr=0.05, window_size=7):
        super(SGPDNet, self).__init__()
        self.num_classes = num_classes
        self.K = sub_centers_k
        self.structure_dim = structure_dim
        self.lightcnn_embedding_dim = lightcnn_embedding_dim
        self.latent_inversion_iters = latent_inversion_iters
        self.latent_inversion_lr = latent_inversion_lr
        self.window_size = window_size

        # === Generative Branch ===
        try:
            self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
            print("Stable Diffusion VAE loaded successfully.")
        except Exception as e:
            print(f"Error loading SD VAE: {e}")
            raise e
        self.vae_scaling_factor = self.vae.config.scaling_factor
        for param in self.vae.parameters():
            param.requires_grad = False

        self.pdslrm = PDSLRM(latent_channels=latent_channels)
        self.structure_feature_extractor = StructureFeatureExtractor(latent_channels=latent_channels,
                                                                     feature_dim=structure_dim)

        # === Discriminative Branch ===
        self.lightcnn = LightCNN_29Layers_v2(num_classes=num_classes)
        if pretrained:
            print("Warning: LightCNN pretrained weights loading not implemented yet.")

        # === SGCLFA & Cross-Layer ===
        channels_after_group1 = 96
        channels_after_group2 = 192

        # FiLM参数生成网络 - 添加正则化
        self.film_param_generator_c1 = nn.Sequential(
            nn.Linear(structure_dim, structure_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 添加dropout
            nn.Linear(structure_dim // 2, channels_after_group1 * 2),
            nn.Tanh()
        )
        self.film_param_generator_c2 = nn.Sequential(
            nn.Linear(structure_dim, structure_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 添加dropout
            nn.Linear(structure_dim // 2, channels_after_group2 * 2),
            nn.Tanh()
        )

        # 改进的SGCLFA
        self.sgclfa1 = SGCLFA(in_channels=channels_after_group1, structure_dim=structure_dim,
                              window_size=window_size, shift_size=0)
        self.sgclfa2 = SGCLFA(in_channels=channels_after_group2, structure_dim=structure_dim,
                              window_size=window_size, shift_size=window_size // 2)

        # 跨层聚合
        self.cross_layer_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.cross_layer_proj = nn.Conv2d(channels_after_group1, channels_after_group2, kernel_size=1, bias=False)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels_after_group2 * 2, channels_after_group2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_after_group2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # 添加2D dropout
            nn.Conv2d(channels_after_group2, channels_after_group2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels_after_group2),
            nn.ReLU(inplace=True)
        )

        # === 改进的损失函数 ===
        self.auxiliary_head = FGDLossHead(structure_dim, num_classes)
        self.arcface_loss_calc = SubCenterArcFaceLoss(in_features=lightcnn_embedding_dim, out_features=num_classes,
                                                      K=sub_centers_k, s=arcface_s, m=arcface_m)

        # 添加新的损失函数
        self.center_loss = CenterLoss(num_classes, lightcnn_embedding_dim, use_gpu=True)
        self.domain_adaptation_loss = DomainAdaptationLoss(lightcnn_embedding_dim)

        # === Modify LightCNN Forward ===
        self._modify_lightcnn_forward()

        # === Placeholders ===
        self.rec_img = None
        self.fg = None
        self.fd = None
        self.zs = None
        self.aug_zs = None
        self.rec_img_from_z0 = None

        # === Latent Inversion Criterion ===
        self.inversion_criterion = nn.MSELoss(reduction='mean')

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化新增层的权重"""
        for m in [self.film_param_generator_c1, self.film_param_generator_c2]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        for m in self.fusion_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _perform_latent_inversion(self, image_norm_minus_1_1):
        """执行潜变量反演"""
        device = image_norm_minus_1_1.device
        # 1. 初始猜测
        with torch.no_grad():
            latent_dist = self.vae.encode(image_norm_minus_1_1).latent_dist
            init_latent = latent_dist.mean * self.vae_scaling_factor
            init_latent = init_latent.detach()

        # 2. 设置可优化参数
        current_latent = init_latent.clone().requires_grad_(True)
        # 3. 优化器 - 降低学习率提高稳定性
        optimizer = torch.optim.Adam([current_latent], lr=self.latent_inversion_lr * 0.8)
        # 4. 迭代优化
        best_loss = float('inf')
        best_latent = init_latent.clone()

        decoder_param_requires_grad_state = {}
        for name, param in self.vae.decoder.named_parameters():
            decoder_param_requires_grad_state[name] = param.requires_grad
            param.requires_grad = False

        original_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        try:
            for i in range(self.latent_inversion_iters):
                optimizer.zero_grad()
                latent_input = current_latent / self.vae_scaling_factor
                reconstructed_image = self.vae.decode(latent_input).sample
                loss = self.inversion_criterion(reconstructed_image, image_norm_minus_1_1)
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_latent = current_latent.detach().clone()
        finally:
            for name, param in self.vae.decoder.named_parameters():
                if name in decoder_param_requires_grad_state:
                    param.requires_grad = decoder_param_requires_grad_state[name]
            torch.set_grad_enabled(original_grad_enabled)

        # 使用best_latent重建最终图像
        with torch.no_grad():
            best_latent_input = best_latent / self.vae_scaling_factor
            inversion_reconstructed_image = self.vae.decode(best_latent_input).sample

        return best_latent, inversion_reconstructed_image

    def _modify_lightcnn_forward(self):
        """修改LightCNN的前向传播"""
        original_forward = self.lightcnn.forward
        self.lightcnn.original_forward = original_forward
        sgpd_self = self

        def modified_forward(instance, x_input_vaenorm, fg=None, subject_mask=None):
            x = (x_input_vaenorm + 1.0) / 2.0
            if torch.isnan(x).any():
                print("NaN detected in LightCNN input")
                raise ValueError

            # Stage 1
            x = instance.conv1(x)
            x = instance.eca1(x)
            x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
            x = instance.block1(x)
            x_group1_out = instance.group1(x)

            # 应用SGCLFA 1
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

            # Stage 2
            x = instance.block2(x_pooled_after_eca2)
            x_group2_out = instance.group2(x)

            # 应用SGCLFA 2
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

            # LightCNN剩余部分
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

    # --- 主要前向传播 ---
    def forward(self, img1, img2=None, subject_mask=None, label=None, mode="train"):
        if mode == "train":
            if img2 is None or label is None:
                raise ValueError("img2 and label needed for train mode")
            return self._forward_train(img1, img2, subject_mask, label)
        elif mode == "predict" or mode == "eval_logits":
            return self._forward_eval_predict(img1, subject_mask, label, mode)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _forward_train(self, img1, img2, subject_mask, label):
        """训练模式"""
        # 主图像路径：使用img1（原图）
        z0_img1_star, rec_img_from_z0_star = self._perform_latent_inversion(img1)
        self.rec_img_from_z0 = rec_img_from_z0_star.detach()

        self.zs = self.pdslrm(z0_img1_star)

        # 对差异化增强图像进行反演用于LSC损失
        z0_img2_star, _ = self._perform_latent_inversion(img2)
        self.augmented_zs = self.pdslrm(z0_img2_star)

        # 计算结构特征
        self.fg = self.structure_feature_extractor(self.zs)

        if torch.isnan(self.fg).any():
            print("NaN detected in feature fg")

        # 将结构特征和主体掩码传递给LightCNN
        self.fd = self.lightcnn(img1, fg=self.fg, subject_mask=subject_mask)

        if torch.isnan(self.fd).any():
            print("NaN detected in feature fd")

        # 计算重建图像
        with torch.no_grad():
            rec_latent_from_zs = self.zs / self.vae_scaling_factor
            self.rec_img = self.vae.decode(rec_latent_from_zs).sample

        # 计算ArcFace logits
        output_logits = self.arcface_loss_calc(self.fd, label)

        if torch.isnan(output_logits).any():
            print("NaN detected in output_logits")
            return F.cross_entropy(self.fd, label)

        return output_logits

    def _forward_eval_predict(self, img, subject_mask=None, label=None, mode="predict"):
        """评估/预测模式"""
        z0_star, _ = self._perform_latent_inversion(img)
        zs = self.pdslrm(z0_star)
        fg = self.structure_feature_extractor(zs)
        fd = self.lightcnn(img, fg=fg, subject_mask=subject_mask)

        if mode == "predict":
            return F.normalize(fd)
        elif mode == "eval_logits":
            if label is None:
                raise ValueError("Label needed for eval_logits mode")
            output_logits = self.arcface_loss_calc(fd, label)
            return output_logits

    # --- 损失计算方法 ---
    def compute_auxiliary_loss(self, label):
        if self.fg is None:
            raise RuntimeError("Run forward(train) first.")
        logits = self.auxiliary_head(self.fg)
        return F.cross_entropy(logits, label)

    def compute_reconstruction_loss(self, orig_img):
        if self.rec_img is None:
            raise RuntimeError("Run forward(train) first.")
        return F.l1_loss(self.rec_img, orig_img)

    def compute_lsc_loss(self):
        """计算潜变量一致性损失"""
        if self.zs is None or self.augmented_zs is None:
            raise RuntimeError("Run forward(train) first.")
        lsc_loss = F.mse_loss(self.zs, self.augmented_zs)
        return lsc_loss

    def compute_center_loss(self, label):
        """计算Center Loss"""
        if self.fd is None:
            raise RuntimeError("Run forward(train) first.")
        return self.center_loss(self.fd, label)

    def compute_domain_adaptation_loss(self, domain_labels):
        """计算域适应损失"""
        if self.fd is None:
            raise RuntimeError("Run forward(train) first.")
        return self.domain_adaptation_loss(self.fd, domain_labels)