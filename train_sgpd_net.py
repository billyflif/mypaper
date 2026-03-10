# train_sgpd_net.py
import os
import numpy as np
import torch
import torch.optim as optim
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, WeightedRandomSampler
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import math
import argparse

from nets.light_cnn_sim import LightCNN_29Layers_v2

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Import project modules
from models.sgpd_net import SGPDNet, PDSLRM, StructureFeatureExtractor, SGCLFA, FGDLossHead
from utils.dataloader_sgpd import SGPDDataset, sgpd_dataset_collate
from utils.ch4_protocol import (
    compute_checkpoint_score,
    compute_macro_f1,
    compute_topk_accuracies,
    rewrite_annotation_lines,
)
from utils.callback import LossHistory
from utils.utils import get_num_classes, seed_everything, show_config, worker_init_fn, get_lr

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def parse_args():
    parser = argparse.ArgumentParser(description='训练SGPD-Net模型')
    parser.add_argument('--debug', default=False, action='store_true', help='启用调试模式（简化训练）')
    parser.add_argument('--log-freq', type=int, default=6, help='日志记录频率（批次）')
    parser.add_argument('--resume', type=str, default='', help='要恢复训练的检查点路径')
    parser.add_argument('--start-epoch', type=int, default=0, help='恢复训练时的起始轮次')
    parser.add_argument('--legacy', action='store_true', help='指定加载的是旧格式检查点（只有模型权重）')
    parser.add_argument('--warmup-simple', default=False, action='store_true', help='启用简化预训练阶段')
    parser.add_argument('--sam-checkpoint', type=str, default='sam_vit_h_4b8939.pth', help='SAM模型检查点路径')
    parser.add_argument('--sam-model-type', type=str, default='vit_h', help='SAM模型类型 (vit_h, vit_b, vit_l)')
    parser.add_argument('--disable-sam', action='store_true', help='禁用SAM模型，使用标准数据增强')
    parser.add_argument('--thesis-tuned', action='store_true', help='启用第四章定稿预设')
    parser.add_argument('--dataset-root', type=str, default='', help='用于重写标注文件中图像路径的本地数据根目录')
    parser.add_argument('--window-size', type=int, default=7, help='窗口注意力的窗口大小')
    # 修改参数解析器，添加SAM2配置路径
    parser.add_argument('--sam-model-cfg', type=str, default=r"E:\Mypaper2\conf\sam2.1_hiera_t.yaml",
                        help='SAM2模型配置文件路径')
    return parser.parse_args()


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """保存包含所有训练状态的完整检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'random_state': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state()
        }
    }
    torch.save(checkpoint, save_path)
    print(f"已保存检查点到 {save_path}")


# 新增简化的 ArcFace 损失
class SimplifiedArcFace(nn.Module):
    """简化版 ArcFace 损失，用于解决收敛问题"""

    def __init__(self, in_features, out_features, s=10.0, m=0.1):
        super(SimplifiedArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # 标准化输入和权重
        x = F.normalize(input)
        W = F.normalize(self.weight)

        # 计算余弦相似度
        cosine = F.linear(x, W)

        # 为目标类添加角度边际
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # 应用角度边际
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


# 修改子中心 ArcFace 损失
class ImprovedSubCenterArcFaceLoss(nn.Module):
    """改进的子中心ArcFace损失，提高数值稳定性"""

    def __init__(self, in_features, out_features, K=2, s=16.0, m=0.2):
        super(ImprovedSubCenterArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.s = s
        self.m = m

        # 使用更合理的初始化
        self.weight = nn.Parameter(torch.FloatTensor(out_features, K, in_features))
        stdv = 1. / math.sqrt(in_features)
        self.weight.data.uniform_(-stdv, stdv)

        # 预计算角度参数
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        # 数值稳定性参数
        self.eps = 1e-8

    def forward(self, embedding, label):
        """增强的子中心ArcFace实现"""
        # 检查输入是否包含NaN
        if torch.isnan(embedding).any():
            print("警告: embedding包含NaN值")
            embedding = torch.nan_to_num(embedding, nan=0.0)

        # 标准化输入特征
        embedding_norm = F.normalize(embedding, p=2, dim=1)

        # 标准化权重
        weight_norm = F.normalize(self.weight.view(-1, self.in_features), p=2, dim=1)

        # 计算余弦相似度
        cosine_all = F.linear(embedding_norm, weight_norm)
        cosine_all = cosine_all.view(-1, self.out_features, self.K)

        # 选择最大余弦相似度
        cosine, _ = torch.max(cosine_all, dim=2)

        # 为目标类添加角度边际
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # 条件性地应用角度边际
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # 创建 one-hot 编码
        one_hot = torch.zeros(cosine.size(), device=embedding.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # 应用边际
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


def get_warmup_lr(current_step, warmup_steps, base_lr, warmup_start_lr):
    """计算预热阶段的学习率"""
    if current_step < warmup_steps:
        lr = warmup_start_lr + (base_lr - warmup_start_lr) * (current_step / warmup_steps)
        return lr
    else:
        return base_lr


def build_weighted_sampler(labels):
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    sample_weights = [1.0 / label_counts[label] for label in labels]
    return WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)



def get_dynamic_loss_weights(epoch, total_epochs, use_warmup_simple, losses_weights, thesis_tuned=False):
    if thesis_tuned:
        warm_epochs = 5
        transition_end = 20
        warmup_weights = {
            'reid': 0.6,
            'rec': 0.2,
            'lsc': 0.0,
            'g_disc': 0.0,
            'center': 0.0,
            'domain_adapt': 0.0
        }
        final_weights = {
            'reid': 1.0,
            'rec': losses_weights.get('rec', 0.15),
            'lsc': losses_weights.get('lsc', 0.10),
            'g_disc': losses_weights.get('g_disc', 0.10),
            'center': losses_weights.get('center', 0.02),
            'domain_adapt': losses_weights.get('domain_adapt', 0.0)
        }

        if epoch < warm_epochs:
            print(f"Loss Strategy (Epoch {epoch + 1}): THESIS-TUNED WARMUP")
            return warmup_weights

        if epoch < transition_end:
            progress = (epoch - warm_epochs + 1) / max(transition_end - warm_epochs, 1)
            smooth_factor = 0.5 * (1 - math.cos(math.pi * progress))
            print(f"Loss Strategy (Epoch {epoch + 1}): THESIS-TUNED TRANSITION (Factor: {smooth_factor:.3f})")
            blended = {}
            for key, warm_value in warmup_weights.items():
                blended[key] = warm_value + (final_weights[key] - warm_value) * smooth_factor
            return blended

        print(f"Loss Strategy (Epoch {epoch + 1}): THESIS-TUNED FULL")
        return final_weights

    """
    动态调整损失权重 - 修复版本 (解决假设1)
    采用课程学习策略：
    1. 预训练引导分支 (Re-ID loss = 0)
    2. 淡入Re-ID损失
    3. 全损失训练
    """

    # --- 可在此处调整阶段长度 ---
    # 阶段1：主要训练引导分支，给Re-ID一个小权重避免突变
    # 现在重建损失已修复，可以适当缩短此阶段
    stage1_epochs = 10

    # 阶段2：淡入Re-ID损失，引导分支权重降低
    stage2_epochs = 30
    # --------------------------

    # losses_weights 包含了你设置的目标权重，例如：
    # losses_weights = {'rec': 0.3, 'lsc': 0.8, 'g_disc': 0.1, ...}

    if epoch < stage1_epochs:
        # 阶段1：强制模型学习重建和一致性。
        print(f"Loss Strategy (Epoch {epoch + 1}): STAGE 1 - Guidance Pre-training")
        return {
            'reid': 0.1,  # <- 给Re-ID一个小权重避免突变
            'rec': 1.0,  # 提高重建权重
            'lsc': 1.0,  # 提高一致性权重
            'g_disc': 0.0,  # 关闭其他辅助损失
            'center': 0.0,
            'domain_adapt': 0.0
        }
    elif epoch < stage2_epochs:
        # 阶段2：逐步淡入 Re-ID 损失和其他辅助损失
        # 计算 0 -> 1 的平滑过渡因子
        progress = (epoch - stage1_epochs) / (stage2_epochs - stage1_epochs)
        smooth_factor = 0.5 * (1 - math.cos(math.pi * progress))

        print(f"Loss Strategy (Epoch {epoch + 1}): STAGE 2 - Fading in Re-ID (Factor: {smooth_factor:.3f})")

        # Re-ID 损失从 0.1 缓慢增加到 1.0
        reid_weight = 0.1 + (1.0 - 0.1) * smooth_factor

        # 引导损失从 1.0 缓慢降低到你设定的目标值
        rec_weight = 1.0 - (1.0 - losses_weights.get('rec', 0.3)) * smooth_factor
        lsc_weight = 1.0 - (1.0 - losses_weights.get('lsc', 0.8)) * smooth_factor

        return {
            'reid': reid_weight,
            'rec': rec_weight,
            'lsc': lsc_weight,
            'g_disc': losses_weights.get('g_disc', 0.1) * smooth_factor,
            'center': losses_weights.get('center', 0.03) * smooth_factor,
            'domain_adapt': losses_weights.get('domain_adapt', 0.15) * smooth_factor
        }
    else:
        # 阶段3：使用你配置的标准损失权重
        print(f"Loss Strategy (Epoch {epoch + 1}): STAGE 3 - Full Training")
        return {
            'reid': 1.0,  # 启用 Re-ID 损失
            'rec': losses_weights.get('rec', 0.3),
            'lsc': losses_weights.get('lsc', 0.8),
            'g_disc': losses_weights.get('g_disc', 0.1),
            'center': losses_weights.get('center', 0.03),
            'domain_adapt': losses_weights.get('domain_adapt', 0.15)
        }


def visualize_model_architecture(model, save_dir):
    """可视化模型架构并输出到文件"""
    try:
        import os
        import io
        from torchvision.models import resnet18
        from torch.utils.tensorboard import SummaryWriter

        # 创建示例输入 (batch_size=2)
        dummy_img = torch.randn(2, 3, 128, 128).to(next(model.parameters()).device)
        dummy_aug = torch.randn(2, 3, 128, 128).to(next(model.parameters()).device)
        dummy_mask = torch.ones(2, 1, 128, 128).to(next(model.parameters()).device)
        dummy_label = torch.zeros(2, dtype=torch.long).to(next(model.parameters()).device)

        # 创建TensorBoard
        writer = SummaryWriter(os.path.join(save_dir, 'architecture'))

        # 添加图形
        try:
            with torch.no_grad():
                # 只记录PDSLRM和StructureFeatureExtractor的图
                dummy_z0 = torch.randn(2, 4, 16, 16).to(next(model.parameters()).device)
                writer.add_graph(model.pdslrm, dummy_z0)
                writer.add_graph(model.structure_feature_extractor, dummy_z0)

                # 无法添加完整模型图，因为SAM可能会导致复杂性过高
                print("已添加部分模型组件图到TensorBoard")
        except Exception as e:
            print(f"添加模型图时出错: {e}")

        # 记录模型参数统计
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f'parameters/{name}', param.data, 0)

        writer.close()
        print(f"模型架构可视化已保存到 {os.path.join(save_dir, 'architecture')}")
    except Exception as e:
        print(f"生成模型架构可视化时出错: {e}")




def fit_one_epoch(model_train, optimizer, scheduler, epoch, epoch_step, gen_train, device, losses_weights, loss_history,
                  use_warmup, warmup_steps, base_lr, warmup_start_lr, use_simplified_training=False,
                  scaler=None, grad_clip_norm=1.0, visualize_every_n_iters=100,
                  accumulation_steps=1):
    """改进的训练一个epoch函数 (已应用'reid'权重修复)"""

    total_loss_epoch = 0.0
    re_id_loss_unweighted_epoch = 0.0
    rec_loss_unweighted_epoch = 0.0
    lsc_loss_unweighted_epoch = 0.0
    g_disc_loss_unweighted_epoch = 0.0
    center_loss_unweighted_epoch = 0.0
    domain_adapt_loss_unweighted_epoch = 0.0
    correct_predictions = 0
    total_samples = 0

    model_train.train()
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1} Train', postfix=dict, mininterval=0.3)

    global_step_offset = epoch * epoch_step

    has_nan = False
    for name, param in model_train.named_parameters():
        if param.requires_grad and torch.isnan(param).any():
            print(f"警告: 参数 '{name}' 包含NaN值")
            has_nan = True

    if has_nan:
        print("模型参数存在NaN值，将尝试继续训练但请注意可能导致不稳定")

    optimizer.zero_grad()

    if epoch < 10:
        effective_clip_norm = grad_clip_norm * 0.5
    elif epoch < 30:
        effective_clip_norm = grad_clip_norm * 0.75
    else:
        effective_clip_norm = grad_clip_norm

    print(f"使用梯度裁剪阈值: {effective_clip_norm:.3f}")

    for iteration, batch_data in enumerate(gen_train):
        current_global_step = global_step_offset + iteration
        if iteration >= epoch_step: break
        if batch_data is None or batch_data[0] is None:
            print(f"Warn: Skip None batch iter {iteration + 1}")
            continue

        images1, images2, subject_masks, labels = batch_data
        images1 = images1.to(device)
        images2 = images2.to(device)
        subject_masks = subject_masks.to(device)
        labels = labels.to(device)
        batch_samples = labels.size(0)
        total_samples += batch_samples

        if use_warmup and current_global_step < warmup_steps:
            lr = get_warmup_lr(current_global_step, warmup_steps, base_lr, warmup_start_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        try:
            # --- 默认的损失值 ---
            re_id_loss = torch.tensor(0.0, device=device)
            rec_loss = torch.tensor(0.0, device=device)
            lsc_loss = torch.tensor(0.0, device=device)
            g_disc_loss = torch.tensor(0.0, device=device)
            center_loss = torch.tensor(0.0, device=device)
            domain_adapt_loss = torch.tensor(0.0, device=device)

            if scaler:  # FP16
                from torch.cuda.amp import autocast
                with autocast():
                    reid_logits = model_train(images1, images2, subject_masks, labels, mode="train")

                    if reid_logits is None:
                        print(f"警告: reid_logits 为 None，跳过批次")
                        continue

                    # 即使权重为0，也计算 Re-ID 损失（用于统计准确率）
                    re_id_loss = nn.CrossEntropyLoss()(reid_logits, labels)
                    re_id_loss_clamped = torch.clamp(re_id_loss, max=10.0)

                    # 只有在权重>0时才计算其他损失，以节省计算
                    if losses_weights.get('rec', 0) > 0:
                        rec_loss = model_train.compute_reconstruction_loss(images1)
                        rec_loss = torch.clamp(rec_loss, max=1.0)
                    if losses_weights.get('lsc', 0) > 0:
                        lsc_loss = model_train.compute_lsc_loss()
                        lsc_loss = torch.clamp(lsc_loss, max=1.0)
                    if losses_weights.get('g_disc', 0) > 0:
                        g_disc_loss = model_train.compute_auxiliary_loss(labels)
                        g_disc_loss = torch.clamp(g_disc_loss, max=5.0)
                    if losses_weights.get('center', 0) > 0:
                        center_loss = model_train.compute_center_loss(labels)
                        center_loss = torch.clamp(center_loss, max=2.0)
                    if losses_weights.get('domain_adapt', 0) > 0:
                        # 移除no_grad以允许梯度回传到增强分支
                        z0_aug_star, _ = model_train._perform_latent_inversion(images2)
                        zs_aug = model_train.pdslrm(z0_aug_star)
                        fg_aug = model_train.structure_feature_extractor(zs_aug)
                        fd_aug = model_train.lightcnn(images2, fg=fg_aug, subject_mask=subject_masks)
                        combined_features = torch.cat([model_train.fd, fd_aug], dim=0)
                        domain_labels = torch.cat([
                            torch.zeros(batch_samples, dtype=torch.long, device=device),
                            torch.ones(batch_samples, dtype=torch.long, device=device)
                        ])
                        domain_adapt_loss = model_train.compute_domain_adaptation_loss(combined_features, domain_labels)
                        domain_adapt_loss = torch.clamp(domain_adapt_loss, max=2.0)

                    # === 关键修改 ===
                    total_loss = (re_id_loss_clamped * losses_weights.get('reid', 1.0) +
                                  losses_weights.get('rec', 0) * rec_loss +
                                  losses_weights.get('lsc', 0) * lsc_loss +
                                  losses_weights.get('g_disc', 0) * g_disc_loss +
                                  losses_weights.get('center', 0) * center_loss +
                                  losses_weights.get('domain_adapt', 0) * domain_adapt_loss) / accumulation_steps

                    if torch.isnan(total_loss):
                        print(f"警告: 损失为NaN (FP16)，跳过更新")
                        optimizer.zero_grad()
                        continue

                scaler.scale(total_loss).backward()

                if (iteration + 1) % accumulation_steps == 0 or (iteration + 1 == epoch_step):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=effective_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            else:  # FP32
                reid_logits = model_train(images1, images2, subject_masks, labels, mode="train")

                if reid_logits is None:
                    print(f"警告: reid_logits 为 None，跳过批次")
                    continue

                re_id_loss = nn.CrossEntropyLoss(label_smoothing=0.1)(reid_logits, labels)
                re_id_loss_clamped = torch.clamp(re_id_loss, max=10.0)

                if losses_weights.get('rec', 0) > 0:
                    rec_loss = model_train.compute_reconstruction_loss(images1)
                    rec_loss = torch.clamp(rec_loss, max=1.0)
                if losses_weights.get('lsc', 0) > 0:
                    lsc_loss = model_train.compute_lsc_loss()
                    lsc_loss = torch.clamp(lsc_loss, max=1.0)
                if losses_weights.get('g_disc', 0) > 0:
                    g_disc_loss = model_train.compute_auxiliary_loss(labels)
                    g_disc_loss = torch.clamp(g_disc_loss, max=5.0)
                if losses_weights.get('center', 0) > 0:
                    center_loss = model_train.compute_center_loss(labels)
                    center_loss = torch.clamp(center_loss, max=2.0)
                if losses_weights.get('domain_adapt', 0) > 0:
                    # 移除no_grad以允许梯度回传到增强分支
                    z0_aug_star, _ = model_train._perform_latent_inversion(images2)
                    zs_aug = model_train.pdslrm(z0_aug_star)
                    fg_aug = model_train.structure_feature_extractor(zs_aug)
                    fd_aug = model_train.lightcnn(images2, fg=fg_aug, subject_mask=subject_masks)
                    combined_features = torch.cat([model_train.fd, fd_aug], dim=0)
                    domain_labels = torch.cat([
                        torch.zeros(batch_samples, dtype=torch.long, device=device),
                        torch.ones(batch_samples, dtype=torch.long, device=device)
                    ])
                    domain_adapt_loss = model_train.compute_domain_adaptation_loss(combined_features, domain_labels)
                    domain_adapt_loss = torch.clamp(domain_adapt_loss, max=2.0)

                # === 关键修改 ===
                total_loss = (re_id_loss_clamped * losses_weights.get('reid', 1.0) +
                              losses_weights.get('rec', 0) * rec_loss +
                              losses_weights.get('lsc', 0) * lsc_loss +
                              losses_weights.get('g_disc', 0) * g_disc_loss +
                              losses_weights.get('center', 0) * center_loss +
                              losses_weights.get('domain_adapt', 0) * domain_adapt_loss) / accumulation_steps

                if torch.isnan(total_loss):
                    print(f"警告: 损失为NaN (FP32)，跳过更新")
                    optimizer.zero_grad()
                    continue

                total_loss.backward()

                if (iteration + 1) % accumulation_steps == 0 or (iteration + 1 == epoch_step):
                    has_nan_grad = False
                    for param in model_train.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            has_nan_grad = True
                            break

                    if has_nan_grad:
                        print(f"警告: 检测到NaN梯度，跳过此次更新")
                        optimizer.zero_grad()
                        continue

                    torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=effective_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            # 更新损失统计 (使用未加权的损失)
            total_loss_epoch += (total_loss.item() * accumulation_steps) * batch_samples
            re_id_loss_unweighted_epoch += re_id_loss.item() * batch_samples
            rec_loss_unweighted_epoch += rec_loss.item() * batch_samples
            lsc_loss_unweighted_epoch += lsc_loss.item() * batch_samples
            g_disc_loss_unweighted_epoch += g_disc_loss.item() * batch_samples
            center_loss_unweighted_epoch += center_loss.item() * batch_samples
            domain_adapt_loss_unweighted_epoch += domain_adapt_loss.item() * batch_samples

            # 计算准确率 (不受损失权重影响)
            with torch.no_grad():
                preds = torch.argmax(F.softmax(reid_logits, dim=-1), dim=-1)
                correct_predictions += (preds == labels).sum().item()

        except Exception as e:
            print(f"\nError train iter {iteration + 1}: {e}")
            import traceback
            traceback.print_exc()
            optimizer.zero_grad()
            continue

        # 更新进度条
        current_total_samples_iter = total_samples if total_samples > 0 else 1
        pbar.set_postfix(**{
            'loss': total_loss_epoch / current_total_samples_iter,
            'acc': correct_predictions / current_total_samples_iter,
            'ReID_L': re_id_loss_unweighted_epoch / current_total_samples_iter,  # Re-ID
            'Rec_L': rec_loss_unweighted_epoch / current_total_samples_iter,  # Rec
            'LSC_L': lsc_loss_unweighted_epoch / current_total_samples_iter,  # LSC
            'ReID_W': losses_weights.get('reid', 1.0),  # Re-ID 权重
            'Rec_W': losses_weights.get('rec', 0.0),  # Rec 权重
            'lr': get_lr(optimizer)
        })
        pbar.update(1)

    pbar.close()

    # 计算平均指标
    avg_train_loss = total_loss_epoch / total_samples if total_samples > 0 else 0
    avg_train_acc = correct_predictions / total_samples if total_samples > 0 else 0
    avg_reid_loss = re_id_loss_unweighted_epoch / total_samples if total_samples > 0 else 0
    avg_rec_loss = rec_loss_unweighted_epoch / total_samples if total_samples > 0 else 0
    avg_lsc_loss = lsc_loss_unweighted_epoch / total_samples if total_samples > 0 else 0
    avg_g_disc_loss = g_disc_loss_unweighted_epoch / total_samples if total_samples > 0 else 0
    avg_center_loss = center_loss_unweighted_epoch / total_samples if total_samples > 0 else 0
    avg_domain_adapt_loss = domain_adapt_loss_unweighted_epoch / total_samples if total_samples > 0 else 0

    print(f'Epoch {epoch + 1} Train Summary: Avg Total Loss: {avg_train_loss:.4f}, Avg Acc: {avg_train_acc:.4f}')
    print(f'  Avg Losses => ReID: {avg_reid_loss:.4f}, Rec: {avg_rec_loss:.4f}, LSC: {avg_lsc_loss:.4f}')
    print(f'  Center: {avg_center_loss:.4f}, Domain: {avg_domain_adapt_loss:.4f}, G-Disc: {avg_g_disc_loss:.4f}')

    train_metrics = {
        'train_loss': avg_train_loss,
        'train_acc': avg_train_acc,
        'train_reid_l': avg_reid_loss,
        'train_rec_l': avg_rec_loss,
        'train_lsc_l': avg_lsc_loss,
        'train_gdisc_l': avg_g_disc_loss,
        'train_center_l': avg_center_loss,
        'train_domain_l': avg_domain_adapt_loss,
    }
    return train_metrics

def eval_one_epoch(model_eval, epoch, epoch_step_val, gen_val, device):
    """评估一个epoch - 只使用原图进行评估"""
    val_total_loss_epoch = 0.0
    val_correct_predictions = 0
    val_total_samples = 0
    all_preds = []
    all_labels = []
    top5_correct = 0.0

    model_eval.eval()
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1} Val', postfix=dict, mininterval=0.3)

    with torch.no_grad():
        for iteration, batch_data in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            if batch_data is None or batch_data[0] is None:
                print(f"Warning: Skipping invalid validation batch at iteration {iteration + 1}")
                continue

            # 接收两张图像，但验证时只使用原图
            images1, images2, subject_masks, labels = batch_data
            images1 = images1.to(device)  # 只使用原图
            subject_masks = subject_masks.to(device)
            labels = labels.to(device)

            batch_samples = labels.size(0)
            val_total_samples += batch_samples

            try:
                # 验证时只使用原图
                reid_logits = model_eval(images1, subject_mask=subject_masks, label=labels, mode="eval_logits")
                loss = nn.CrossEntropyLoss()(reid_logits, labels)
                val_total_loss_epoch += loss.item() * batch_samples

                preds = torch.argmax(F.softmax(reid_logits, dim=-1), dim=-1)
                val_correct_predictions += (preds == labels).sum().item()
                topk_result = compute_topk_accuracies(reid_logits, labels, topk=(1, 5))
                top5_correct += topk_result[5] * batch_samples
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

            except Exception as e:
                print(f"\nError val iter {iteration + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue

            pbar.set_postfix(**{
                'val_loss': val_total_loss_epoch / val_total_samples if val_total_samples else 0,
                'val_acc': val_correct_predictions / val_total_samples if val_total_samples else 0,
                'val_top5': top5_correct / val_total_samples if val_total_samples else 0,
            })
            pbar.update(1)

    pbar.close()

    avg_val_loss = val_total_loss_epoch / val_total_samples if val_total_samples > 0 else 0
    avg_val_acc = val_correct_predictions / val_total_samples if val_total_samples > 0 else 0
    avg_val_top5 = top5_correct / val_total_samples if val_total_samples > 0 else 0
    avg_val_macro_f1 = compute_macro_f1(all_preds, all_labels) if all_labels else 0.0
    print(
        f'Epoch {epoch + 1} Val Summary: Avg Loss: {avg_val_loss:.4f}, '
        f'Top-1: {avg_val_acc:.4f}, Top-5: {avg_val_top5:.4f}, Macro-F1: {avg_val_macro_f1:.4f}'
    )

    val_metrics = {
        'val_loss': avg_val_loss,
        'val_acc': avg_val_acc,
        'val_top1': avg_val_acc,
        'val_top5': avg_val_top5,
        'val_macro_f1': avg_val_macro_f1,
        'val_score': compute_checkpoint_score({
            'val_top1': avg_val_acc,
            'val_macro_f1': avg_val_macro_f1
        }),
        'val_reid_l': avg_val_loss
    }
    return val_metrics

def get_optimizer_grouped_parameters(model, base_lr, weight_decay):
    """为模型不同部分设置不同学习率"""
    no_decay = ['bias', 'LayerNorm.weight', 'BatchNorm']

    # ArcFace相关参数使用较小学习率
    arcface_params = ['arcface_loss_calc']

    # 反演相关参数使用较大学习率
    inversion_params = ['pdslrm']

    # FiLM参数生成器使用较大学习率
    film_params = ['film_param_generator']

    # 分组参数
    optimizer_grouped_parameters = [
        # 1. 基础参数，正常学习率和权重衰减
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(ap in n for ap in arcface_params)
                    and not any(ip in n for ip in inversion_params)
                    and not any(fp in n for fp in film_params)],
         'weight_decay': weight_decay,
         'lr': base_lr},

        # 2. 基础参数，无权重衰减
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(ap in n for ap in arcface_params)
                    and not any(ip in n for ip in inversion_params)
                    and not any(fp in n for fp in film_params)],
         'weight_decay': 0.0,
         'lr': base_lr},

        # 3. ArcFace参数，较小学习率
        {'params': [p for n, p in model.named_parameters()
                    if any(ap in n for ap in arcface_params)],
         'weight_decay': weight_decay,
         'lr': base_lr * 0.5},  # 调整为0.5

        # 4. 反演相关参数，较大学习率
        {'params': [p for n, p in model.named_parameters()
                    if any(ip in n for ip in inversion_params)],
         'weight_decay': weight_decay,
         'lr': base_lr * 1.2},  # 调整为1.2

        # 5. FiLM参数生成器，较大学习率
        {'params': [p for n, p in model.named_parameters()
                    if any(fp in n for fp in film_params)],
         'weight_decay': weight_decay,
         'lr': base_lr * 1.2}  # 动态调制参数使用略高学习率但不过于激进
    ]

    return optimizer_grouped_parameters


def save_training_config(save_dir, config_dict):
    """
    将训练配置保存到指定目录下的config.txt文件中

    Args:
        save_dir (str): 保存目录路径
        config_dict (dict): 包含所有要保存的参数的字典
    """
    import json
    import datetime
    from pathlib import Path
    import argparse

    # 创建配置字典的副本，避免修改原始配置
    config_copy = config_dict.copy()

    # 自动转换命名空间对象
    for key, value in config_copy.items():
        if isinstance(value, argparse.Namespace):
            config_copy[key] = vars(value)

    # 确保目录存在
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 配置文件路径
    config_path = os.path.join(save_dir, "training_config.txt")
    config_json_path = os.path.join(save_dir, "training_config.json")

    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 写入JSON格式(便于程序读取)
    with open(config_json_path, 'w', encoding='utf-8') as f:
        json.dump(config_copy, f, indent=4, ensure_ascii=False)

    # 写入人类可读的格式
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(f"训练配置 (记录时间: {current_time})\n")
        f.write("=" * 50 + "\n\n")

        # 写入命令行参数
        if 'args' in config_dict:
            f.write("命令行参数:\n")
            args_dict = vars(config_dict['args']) if hasattr(config_dict['args'], '__dict__') else config_dict['args']
            for k, v in args_dict.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        # 按类别组织参数
        categories = {
            "模型参数": ["num_classes", "latent_channels", "structure_dim", "sub_centers_k",
                     "arcface_s", "arcface_m", "latent_inversion_iters", "window_size"],
            "优化器参数": ["optimizer_type", "Init_lr", "weight_decay", "warmup_epochs",
                      "warmup_start_lr", "lr_decay_type", "Min_lr"],
            "训练参数": ["Epoch", "batch_size", "accumulation_steps", "grad_clip_norm",
                     "losses_weights", "use_warmup_simple", "fp16"],
            "数据参数": ["input_shape", "annotation_path", "num_workers", "val_split", "use_sam_model"],
            "保存参数": ["save_dir", "save_period", "visualize_freq"]
        }

        # 未分类的参数
        used_keys = []
        for category_keys in categories.values():
            used_keys.extend(category_keys)

        # 按类别写入配置
        for category, keys in categories.items():
            f.write(f"{category}:\n")
            for k in keys:
                if k in config_dict:
                    value = config_dict[k]
                    # 如果值是字典，进行格式化
                    if isinstance(value, dict):
                        f.write(f"  {k}:\n")
                        for sub_k, sub_v in value.items():
                            f.write(f"    {sub_k}: {sub_v}\n")
                    else:
                        f.write(f"  {k}: {value}\n")
                    used_keys.append(k)
            f.write("\n")

        # 写入未分类的参数
        other_params = [k for k in config_dict.keys() if k not in used_keys and k != 'args']
        if other_params:
            f.write("其他参数:\n")
            for k in other_params:
                value = config_dict[k]
                if isinstance(value, dict):
                    f.write(f"  {k}:\n")
                    for sub_k, sub_v in value.items():
                        f.write(f"    {sub_k}: {sub_v}\n")
                else:
                    f.write(f"  {k}: {value}\n")
            f.write("\n")

        f.write("=" * 50 + "\n")
        f.write(f"配置文件保存在 {config_json_path} (JSON格式)\n")

    print(f"训练配置已保存至 {config_path}")
    return config_path


if __name__ == "__main__":
    args = parse_args()
    debug_mode = args.debug
    log_freq = args.log_freq
    use_warmup_simple = args.warmup_simple
    window_size = args.window_size
    thesis_tuned = args.thesis_tuned
    dataset_root = args.dataset_root.strip() or None

    # 如果启用调试模式，修改参数
    if debug_mode:
        print("启用调试模式 - 使用简化训练设置")
        Epoch = 20
        batch_size = 8
        losses_weights = {'rec': 0.0, 'lsc': 0.0, 'g_disc': 0.0}
        latent_inversion_iters = 5

    # --- 1. 超参数和配置 ---
    seed = 3407
    seed_everything(seed)
    Cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and Cuda else 'cpu')
    fp16 = False

    annotation_path = "paperdata-train.txt"
    input_shape = [128, 128]
    structure_dim = 256
    lightcnn_embedding_dim = 512

    # Latent Inversion 参数
    latent_inversion_iters = 30
    latent_inversion_lr = 0.01

    # 训练参数
    batch_size = 16
    num_workers = 0
    Init_Epoch = 0
    Epoch =  200

    # 梯度累积步数
    accumulation_steps = 2

    use_sam_model = not args.disable_sam
    sam_model_path = r"E:\Mypaper2\conf\sam2.1_hiera_tiny.pt"
    sam_model_cfg = r"E:\Mypaper2\conf\sam2.1_hiera_t.yaml"
    # 模型参数 - 更保守的设置
    latent_channels = 4
    pds_lrm_layers = 2
    pds_lrm_heads = 4
    sd_model_path = "runwayml/stable-diffusion-v1-5"
    sub_centers_k = 2
    arcface_s = 12
    arcface_m = 0.15

    # 优化器参数
    Init_lr = 0.0001
    optimizer_type = "adamw"
    weight_decay = 1e-3

    # 预热参数
    warmup_epochs = 10
    # 注意：warmup_steps会在创建数据集后重新计算
    warmup_steps = None  # 临时占位，稍后会用真实的epoch_step计算
    warmup_start_lr = Init_lr * 0.01

    # 学习率调度器
    lr_decay_type = "cosine"
    Min_lr = Init_lr * 0.1

    # 损失权重 - 更保守的设置
    losses_weights = {
        'rec': 0.3,
        'lsc': 0.8,
        'g_disc': 0.1,
        'center': 0.03,
        'domain_adapt': 0.05
    }

    if thesis_tuned:
        losses_weights = {
            'rec': 0.15,
            'lsc': 0.10,
            'g_disc': 0.10,
            'center': 0.02,
            'domain_adapt': 0.0
        }

    # 梯度裁剪阈值
    grad_clip_norm = 2.0

    # 保存设置
    model_name_suffix = f"{'_sam' if use_sam_model else ''}_win{window_size}"
    if thesis_tuned:
        model_name_suffix += "_thesis_tuned"
    save_dir = f'newdata-logs/swin{model_name_suffix}_iters{latent_inversion_iters}_k{sub_centers_k}_s{arcface_s}_m{arcface_m}_rec{losses_weights["rec"]}_lsc{losses_weights["lsc"]}_gdisc{losses_weights["g_disc"]}'
    save_period = 2
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    visualize_freq = 100






    # --- 初始化SAM2模型 ---
    sam_predictor = None
    if use_sam_model:
        try:
            print(f"初始化SAM2模型 (checkpoint={sam_model_path}, config={sam_model_cfg})")
            from utils.sam2_integration import initialize_sam2_model

            sam_predictor = initialize_sam2_model(
                checkpoint_path=sam_model_path,
                model_cfg=sam_model_cfg,
                device=device
            )
            if sam_predictor is None:
                print("SAM2初始化失败，禁用SAM功能")
                use_sam_model = False
            else:
                print("SAM2初始化成功")
        except Exception as e:
            import traceback

            print(f"SAM2初始化失败: {e}")
            traceback.print_exc()
            use_sam_model = False

    # --- 3. 数据 ---
    num_classes = get_num_classes(annotation_path)
    print(f"检测到 {num_classes} 个类别")

    lines, rewrite_info = rewrite_annotation_lines(annotation_path, dataset_root=dataset_root)
    print(f"标注路径重写信息: {rewrite_info}")
    np.random.seed(seed)
    np.random.shuffle(lines)
    np.random.seed(None)
    val_split = 0.15
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # 使用新的参数初始化数据集
    train_dataset = SGPDDataset(input_shape, lines[:num_train], train_mode=True,
                                sam_predictor=sam_predictor if use_sam_model else None)
    val_dataset = SGPDDataset(input_shape, lines[num_train:], train_mode=False,
                              sam_predictor=sam_predictor if use_sam_model else None)

    if len(train_dataset) == 0: raise ValueError("Train dataset empty!")

    train_sampler = build_weighted_sampler(train_dataset.labels) if thesis_tuned else None
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=sgpd_dataset_collate,
        worker_init_fn=partial(worker_init_fn, seed=seed)
    )
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=True, drop_last=False, collate_fn=sgpd_dataset_collate,
                                worker_init_fn=partial(worker_init_fn, seed=seed))

    epoch_step = max(1, len(train_dataset) // batch_size)
    epoch_step_val = max(1, len(val_dataset) // batch_size) if len(val_dataset) > 0 else 0
    
    # 使用真实的训练步数计算warmup_steps
    warmup_steps = warmup_epochs * epoch_step
    print(f"训练数据集大小: {len(train_dataset)}, 每轮步数: {epoch_step}, Warmup步数: {warmup_steps}")

    # --- 4. 模型初始化，添加窗口大小参数 ---
    model = SGPDNet(
        num_classes=num_classes, latent_channels=latent_channels, structure_dim=structure_dim,
        lightcnn_embedding_dim=lightcnn_embedding_dim,
        pretrained=False, sub_centers_k=sub_centers_k, arcface_s=arcface_s, arcface_m=arcface_m,
        latent_inversion_iters=latent_inversion_iters, latent_inversion_lr=latent_inversion_lr,
        window_size=window_size
    ).to(device)

    # 初始化检查点恢复相关变量
    start_epoch = Init_Epoch
    best_metrics = {
        'val_loss': float('inf'),
        'val_acc': 0.0,
        'val_top1': 0.0,
        'val_top5': 0.0,
        'val_macro_f1': 0.0,
        'val_score': 0.0
    }

    # 加载检查点（如果提供）
    if args.resume and os.path.exists(args.resume):
        print(f'正在从 {args.resume} 加载检查点')

        # 加载检查点文件
        checkpoint = torch.load(args.resume, map_location=device)

        # 判断是否是旧格式检查点
        if args.legacy or not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
            print("检测到旧格式检查点（只有模型权重），只恢复模型参数")
            try:
                model.load_state_dict(checkpoint, strict=False)
                print("成功加载旧格式模型权重")
            except Exception as e:
                print(f"加载旧格式模型权重时出错: {e}")
                try:
                    # 尝试非严格加载
                    model.load_state_dict(checkpoint, strict=False)
                    print("以非严格模式成功加载旧格式模型权重（部分参数未加载）")
                except Exception as e2:
                    print(f"非严格加载也失败: {e2}")

            start_epoch = args.start_epoch if args.start_epoch > 0 else 0
        else:
            # 新格式检查点，恢复完整状态
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("成功加载新格式模型权重")
            except Exception as e:
                print(f"加载新格式模型权重时出错: {e}")
                try:
                    # 尝试非严格加载
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print("以非严格模式成功加载新格式模型权重（部分参数未加载）")
                except Exception as e2:
                    print(f"非严格加载也失败: {e2}")

            if args.start_epoch <= 0 and 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1  # 从下一轮开始
            else:
                start_epoch = args.start_epoch

            # 恢复最佳指标（如果有）
            if 'metrics' in checkpoint:
                best_metrics = checkpoint['metrics']
                best_metrics['val_score'] = compute_checkpoint_score(best_metrics)
                print(
                    f"已恢复指标: val_loss={best_metrics.get('val_loss', 'N/A')}, val_acc={best_metrics.get('val_acc', 'N/A')}")

            # 恢复随机状态（如果有）
            if 'random_state' in checkpoint:
                try:
                    torch.set_rng_state(checkpoint['random_state']['torch'])
                    if torch.cuda.is_available() and checkpoint['random_state']['cuda'] is not None:
                        if isinstance(checkpoint['random_state']['cuda'], list):
                            torch.cuda.set_rng_state_all(checkpoint['random_state']['cuda'])
                        else:
                            torch.cuda.set_rng_state(checkpoint['random_state']['cuda'])
                    np.random.set_state(checkpoint['random_state']['numpy'])
                    print("已恢复随机状态")
                except Exception as e:
                    print(f"警告: 无法恢复随机状态: {e}")

        # 设置为我们的新 Init_Epoch
        Init_Epoch = start_epoch
        print(f"将从第 {start_epoch} 轮恢复训练")
    elif args.resume:
        print(f"警告: 检查点文件 {args.resume} 未找到，将从头开始训练")

    show_config(
        latent_inversion_iters=latent_inversion_iters,
        latent_inversion_lr=latent_inversion_lr,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        Init_lr=Init_lr,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_start_lr=warmup_start_lr,
        losses_weights=losses_weights,
        grad_clip_norm=grad_clip_norm,
        sub_centers_k=sub_centers_k,
        arcface_s=arcface_s,
        arcface_m=arcface_m,
        use_warmup_simple=use_warmup_simple,
        thesis_tuned=thesis_tuned,
        dataset_root=dataset_root,
        window_size=window_size,
        use_sam_model=use_sam_model,

    )

    print("\n--------- 模型参数检查 ---------")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {total_params - trainable_params:,}")

    # 检查VAE冻结状态
    vae_params = sum(p.numel() for p in model.vae.parameters())
    vae_trainable = sum(p.numel() for p in model.vae.parameters() if p.requires_grad)
    print(f"VAE参数状态: 总数={vae_params:,}, 可训练={vae_trainable:,}")
    print(f"VAE是否正确冻结: {'是' if vae_trainable == 0 else '否'}")

    # 检查各模块参数数量
    sgclfa_params = sum(p.numel() for name, p in model.named_parameters() if 'sgclfa' in name)
    film_params = sum(p.numel() for name, p in model.named_parameters() if 'film_param_generator' in name)
    fusion_params = sum(p.numel() for name, p in model.named_parameters() if 'fusion_conv' in name)

    print(f"SGCLFA参数: {sgclfa_params:,}")
    print(f"FiLM参数生成器参数: {film_params:,}")
    print(f"融合层参数: {fusion_params:,}")

    # 检查SubCenterArcFaceLoss配置
    print(
        f"SubCenterArcFaceLoss配置: 类别数={model.arcface_loss_calc.out_features}, K={model.arcface_loss_calc.K}, s={model.arcface_loss_calc.s}, m={model.arcface_loss_calc.m}")
    print("--------------------------------\n")

    # 模型架构可视化
    visualize_model_architecture(model, save_dir)

    # --- 5. 优化器和学习率调度器 ---
    # 使用分组参数优化器
    optimizer_params = get_optimizer_grouped_parameters(model, Init_lr, weight_decay)

    if optimizer_type == "adamw":
        optimizer = optim.AdamW(optimizer_params, lr=Init_lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        optimizer = optim.Adam(optimizer_params, lr=Init_lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(optimizer_params, lr=Init_lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)

    # 如果从检查点恢复，且是新格式，加载优化器状态
    if args.resume and os.path.exists(args.resume) and isinstance(checkpoint,
                                                                  dict) and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("已恢复优化器状态")
        except Exception as e:
            print(f"警告: 无法加载优化器状态: {e}")

    # 使用余弦退火学习率调度器
    if lr_decay_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epoch - warmup_epochs, eta_min=Min_lr)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=(Epoch - warmup_epochs) // 3, gamma=0.5)

    # 如果从检查点恢复，且是新格式，加载调度器状态
    if args.resume and os.path.exists(args.resume) and isinstance(checkpoint,
                                                                  dict) and 'scheduler_state_dict' in checkpoint and \
            checkpoint['scheduler_state_dict']:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("已恢复调度器状态")
        except Exception as e:
            print(f"警告: 无法加载调度器状态: {e}")

    # --- 6. 训练准备 ---
    loss_history = LossHistory(save_dir, model, input_shape)
    scaler = torch.cuda.amp.GradScaler() if fp16 else None

    # --- 7. 训练循环 ---
    print("开始训练循环...")

    # 检查在反演过程中是否存在任何NaN（执行一个小批量）
    print("执行测试批次以检查模型运行...")
    test_batch = next(iter(train_dataloader))
    if test_batch is not None and test_batch[0] is not None:
        # 修改为仅解包4个值而非5个
        images1, images2, subject_masks, labels = test_batch
        images1 = images1.to(device)
        images2 = images2.to(device)
        subject_masks = subject_masks.to(device)
        labels = labels.to(device)

        # 检查标签分布
        print(f"测试批次标签分布: min={labels.min()}, max={labels.max()}, 唯一值数量={len(torch.unique(labels))}")

        with torch.no_grad():
            try:
                # 只测试原图的反演（images1）
                z0_base_star, _ = model._perform_latent_inversion(images1[:2])  # 只测试2个样本
                print(f"测试反演成功，形状: {z0_base_star.shape}")

                # 测试模型输出（使用两张图而不是三张）
                reid_logits = model(images1[:2], images2[:2], subject_masks[:2], labels[:2], mode="train")
                print(f"测试模型输出成功，logits形状: {reid_logits.shape}")
                print(f"测试logits统计: mean={reid_logits.mean():.4f}, std={reid_logits.std():.4f}")
            except Exception as e:
                print(f"测试模型失败: {e}")
                import traceback

                traceback.print_exc()

    training_config = {
        'args': vars(args),  # 将Namespace对象转换为字典
        'num_classes': num_classes,
        'latent_channels': latent_channels,
        'structure_dim': structure_dim,
        'lightcnn_embedding_dim': lightcnn_embedding_dim,
        'latent_inversion_iters': latent_inversion_iters,
        'latent_inversion_lr': latent_inversion_lr,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'Init_Epoch': Init_Epoch,
        'Epoch': Epoch,
        'accumulation_steps': accumulation_steps,
        'sub_centers_k': sub_centers_k,
        'arcface_s': arcface_s,
        'arcface_m': arcface_m,
        'Init_lr': Init_lr,
        'optimizer_type': optimizer_type,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'warmup_start_lr': warmup_start_lr,
        'lr_decay_type': lr_decay_type,
        'Min_lr': Min_lr,
        'losses_weights': losses_weights,
        'grad_clip_norm': grad_clip_norm,
        'save_dir': save_dir,
        'save_period': save_period,
        'visualize_freq': visualize_freq,
        'input_shape': input_shape,
        'annotation_path': annotation_path,
        'val_split': val_split,
        'seed': seed,
        'thesis_tuned': thesis_tuned,
        'dataset_root': dataset_root,
        'annotation_rewrite_info': rewrite_info,
        'fp16': fp16,
        'use_warmup_simple': use_warmup_simple,
        'device': str(device),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset) if val_dataset else 0,
        'steps_per_epoch': epoch_step,
        'steps_per_epoch_val': epoch_step_val,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'gpu_info': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only',
        'window_size': window_size,
        'use_sam_model': use_sam_model,
        'class_balanced_sampler': thesis_tuned,

        'sam_model_path': sam_model_path if use_sam_model else "None",
    }
    # 保存训练配置
    config_path = save_training_config(save_dir, training_config)

    # 训练循环
    for epoch in range(Init_Epoch, Epoch):
        epoch_start_time = time.time()

        # 动态调整损失权重 - 使用改进的策略
        current_losses_weights = get_dynamic_loss_weights(
            epoch, Epoch, use_warmup_simple, losses_weights, thesis_tuned=thesis_tuned
        )
        print(f"\n当前损失权重:")
        print(f"  rec={current_losses_weights['rec']:.3f}, lsc={current_losses_weights['lsc']:.3f}")
        print(f"  g_disc={current_losses_weights['g_disc']:.3f}, center={current_losses_weights['center']:.3f}")
        print(f"  domain_adapt={current_losses_weights['domain_adapt']:.3f}")

        # 确定是否使用简化训练
        use_simplified_training = use_warmup_simple and epoch < 10

        if use_simplified_training:
            print(f"\n第 {epoch + 1} 轮使用简化训练（仅分类损失）")

        # 训练
        train_metrics = fit_one_epoch(
            model, optimizer, scheduler,
            epoch, epoch_step, train_dataloader, device,
            current_losses_weights, loss_history,
            use_warmup=(epoch < warmup_epochs), warmup_steps=warmup_steps,
            base_lr=Init_lr, warmup_start_lr=warmup_start_lr,
            use_simplified_training=use_simplified_training,
            scaler=scaler, grad_clip_norm=grad_clip_norm,
            visualize_every_n_iters=visualize_freq,
            accumulation_steps=accumulation_steps
        )

        # 验证
        val_metrics = {
            'val_loss': 0.0,
            'val_acc': 0.0,
            'val_top1': 0.0,
            'val_top5': 0.0,
            'val_macro_f1': 0.0,
            'val_score': 0.0,
            'val_reid_l': 0.0
        }
        if epoch_step_val > 0:
            val_metrics = eval_one_epoch(model, epoch, epoch_step_val, val_dataloader, device)

        # 在预热阶段之后才更新学习率调度器
        if epoch >= warmup_epochs:
            scheduler.step()

        # 记录日志
        all_metrics = {**train_metrics, **val_metrics}
        loss_history.append_loss(epoch, all_metrics)

        if val_metrics['val_score'] > best_metrics.get('val_score', 0.0):
            best_metrics.update({
                'val_acc': val_metrics['val_acc'],
                'val_top1': val_metrics['val_top1'],
                'val_top5': val_metrics['val_top5'],
                'val_macro_f1': val_metrics['val_macro_f1'],
                'val_loss': val_metrics['val_loss'],
                'val_score': val_metrics['val_score']
            })
            best_checkpoint_path = os.path.join(save_dir, 'checkpoint_best.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, best_metrics, best_checkpoint_path)
            print(
                f"保存新的最佳模型，val_score = {best_metrics['val_score']:.4f}, "
                f"Top-1 = {best_metrics['val_top1']:.4f}, Macro-F1 = {best_metrics['val_macro_f1']:.4f}"
            )

        # 保存模型
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            # 保存普通权重文件（只有模型权重）
            weights_path = os.path.join(save_dir,
                                        f'ep{epoch + 1:03d}-loss{train_metrics["train_loss"]:.3f}-val_loss{val_metrics["val_loss"]:.3f}.pth')
            torch.save(model.state_dict(), weights_path)
            print(f"已保存模型权重到 {weights_path}")

            # 保存包含所有状态的完整检查点
            checkpoint_path = os.path.join(save_dir, f'checkpoint_ep{epoch + 1:03d}.pth')
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={**train_metrics, **val_metrics},
                save_path=checkpoint_path
            )

            # 保存一个"最新"检查点，每次都会被覆盖
            latest_checkpoint_path = os.path.join(save_dir, 'checkpoint_latest.pth')
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={**train_metrics, **val_metrics},
                save_path=latest_checkpoint_path
            )

        epoch_end_time = time.time()
        print(f"第 {epoch + 1} 轮训练完成，用时 {epoch_end_time - epoch_start_time:.2f} 秒。")

        # 如果连续多轮准确率为0，输出调试信息
        if epoch > 5 and train_metrics['train_acc'] == 0 and val_metrics['val_acc'] == 0:
            print(f"\n警告: 连续 {epoch - 4} 轮准确率为0，建议检查以下内容：")
            print("1. 标签是否正确映射")
            print("2. 模型输出的logits分布")
            print("3. ArcFace参数是否过于激进")
            print("4. 学习率是否过低")
            print("5. FiLM参数生成器是否正确初始化")
            print("6. 窗口注意力机制是否产生了预期的效果")

            # 尝试恢复训练的建议
            print("\n尝试使用以下参数恢复训练:")
            print("- 减小arcface_s和arcface_m参数")
            print("- 增大初始学习率或禁用预热")
            print("- 使用--warmup-simple以逐步引入复杂损失")
            print("- 尝试调整FiLM激活范围")

        # 监控内存使用情况
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_summary'):
                print("\nGPU内存使用情况:")
                print(torch.cuda.memory_summary(device=device, abbreviated=True))

    # 清理SAM资源
    if use_sam_model and sam_predictor is not None:
        try:
            # 释放SAM模型GPU内存
            sam_predictor.model.to('cpu')
            del sam_predictor
            torch.cuda.empty_cache()
            print("已释放SAM模型资源")
        except Exception as e:
            print(f"释放SAM资源时出错: {e}")

    loss_history.close_writer()
    print(
        f"训练完成，最佳验证 Top-1: {best_metrics.get('val_top1', best_metrics.get('val_acc', 0)):.4f}, "
        f"Macro-F1: {best_metrics.get('val_macro_f1', 0):.4f}, "
        f"Score: {best_metrics.get('val_score', 0):.4f}"
    )

    # 输出最终训练总结
    print("\n" + "=" * 50)
    print("训练总结")
    print("=" * 50)
    print(f"模型参数: 总计 {total_params:,} 参数，其中 {trainable_params:,} 可训练")
    print(f"训练样本: {len(train_dataset)} 个，验证样本: {len(val_dataset)} 个")
    print(f"训练轮次: {Epoch} 轮，批次大小: {batch_size}")
    print(
        f"最佳验证指标: Top-1 {best_metrics.get('val_top1', best_metrics.get('val_acc', 0)):.4f}, "
        f"Top-5 {best_metrics.get('val_top5', 0):.4f}, "
        f"Macro-F1 {best_metrics.get('val_macro_f1', 0):.4f}, "
        f"Score {best_metrics.get('val_score', 0):.4f}, "
        f"Loss {best_metrics.get('val_loss', 0):.4f}"
    )
    print(f"最佳模型保存在: {os.path.join(save_dir, 'checkpoint_best.pth')}")
    print(f"SAM模型: {'已启用' if use_sam_model else '未启用'}")
    print(f"窗口大小: {window_size}")
    print(f"训练配置保存在: {config_path}")
    print("=" * 50)
