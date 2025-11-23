# train_ablation_study.py - 修复版本
import os
import numpy as np
import torch
import torch.optim as optim
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import math
import argparse
import json
from datetime import datetime

# 早停机制类
class EarlyStopping:
    """早停机制，当验证指标不再改善时提前停止训练"""
    def __init__(self, patience=10, min_delta=0.0001, mode='max', verbose=True):
        """
        Args:
            patience: 容忍多少个epoch没有改善
            min_delta: 最小改善量，小于这个值不算改善
            mode: 'max'表示指标越大越好(如准确率)，'min'表示越小越好(如损失)
            verbose: 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_value, epoch):
        """
        检查是否应该早停
        Returns:
            bool: 是否应该早停
        """
        if self.best_value is None:
            # 第一次调用
            self.best_value = current_value
            self.best_epoch = epoch
            return False

        # 判断是否改善
        if self.mode == 'max':
            improved = (current_value - self.best_value) > self.min_delta
        else:
            improved = (self.best_value - current_value) > self.min_delta

        if improved:
            # 有改善，更新最佳值并重置计数器
            if self.verbose:
                print(f"验证指标改善: {self.best_value:.4f} -> {current_value:.4f}")
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
        else:
            # 无改善，增加计数器
            self.counter += 1
            if self.verbose:
                print(f"验证指标未改善，早停计数: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                if self.verbose:
                    print(f"\n早停触发！已连续 {self.patience} 个epoch无改善")
                    print(f"最佳epoch: {self.best_epoch}, 最佳指标: {self.best_value:.4f}")
                self.early_stop = True
                return True

        return False

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0

# 导入原始模块
from models.sgpd_net import SGPDNet
from models.ablation_models import *  # 我们将创建的消融模型
from utils.dataloader_sgpd import SGPDDataset, sgpd_dataset_collate
from utils.callback import LossHistory
from utils.utils import get_num_classes, seed_everything, show_config, worker_init_fn, get_lr
from utils.sam2_integration import initialize_sam2_model


def parse_args():
    parser = argparse.ArgumentParser(description='SGPD-Net消融实验')

    # 基本参数
    parser.add_argument('--experiment-name', type=str, default='ablation_study',
                        help='实验名称（用于保存目录）')
    parser.add_argument('--config', type=str, default='configs/ablation_experiments.json',
                        help='消融实验配置文件')
    parser.add_argument('--data-path', type=str, default='paperdata-train.txt',
                        help='数据标注文件路径')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--seed', type=int, default=3407, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备：auto/cuda/cpu')

    # SAM相关
    parser.add_argument('--sam-checkpoint', type=str,
                        default='E:\Mypaper2\conf\sam2.1_hiera_tiny.pt',
                        help='SAM模型检查点路径')
    parser.add_argument('--sam-config', type=str,
                        default='E:\Mypaper2\conf\sam2.1_hiera_t.yaml',
                        help='SAM模型配置文件路径')
    parser.add_argument('--disable-sam', action='store_true', help='禁用SAM模型')

    # 实验控制
    parser.add_argument('--run-all', action='store_true', help='运行所有预定义的消融实验', default=True)
    parser.add_argument('--skip-baseline', action='store_true', help='跳过完整的基线模型（baseline）')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--save-freq', type=int, default=10, help='模型保存频率')

    # 早停参数
    parser.add_argument('--early-stopping', action='store_true', default=True,
                        help='启用早停机制（默认启用）')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值，连续多少个epoch无改善则停止（默认15）')
    parser.add_argument('--min-delta', type=float, default=0.0001,
                        help='最小改善阈值，小于此值不算改善（默认0.0001）')

    return parser.parse_args()


class AblationExperimentManager:
    """消融实验管理器"""

    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.experiments = self._load_experiment_configs()

        # 创建主实验目录
        self.base_save_dir = f'ablation_results/{args.experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.base_save_dir, exist_ok=True)

        # 保存实验配置
        self._save_experiment_config()

    def _setup_device(self):
        if self.args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.args.device)
        print(f"使用设备: {device}")
        return device

    def _load_experiment_configs(self):
        """加载消融实验配置"""
        if os.path.exists(self.args.config):
            with open(self.args.config, 'r', encoding='utf-8') as f:
                configs = json.load(f)
        else:
            # 默认消融实验配置
            configs = self._get_default_ablation_configs()
            # 保存默认配置
            os.makedirs(os.path.dirname(self.args.config), exist_ok=True)
            with open(self.args.config, 'w', encoding='utf-8') as f:
                json.dump(configs, f, indent=4, ensure_ascii=False)
            print(f"已创建默认消融实验配置文件: {self.args.config}")

        return configs

    def _get_default_ablation_configs(self):
        """获取默认的消融实验配置"""
        return {
            "baseline": {
                "description": "完整SGPD-Net模型（基线）",
                "disabled_components": [],
                "model_type": "SGPDNet"
            },
            "no_pdslrm": {
                "description": "禁用PDSLRM模块",
                "disabled_components": ["pdslrm"],
                "model_type": "SGPDNet_NoPDSLRM"
            },
            "no_sgclfa": {
                "description": "禁用SGCLFA注意力机制",
                "disabled_components": ["sgclfa"],
                "model_type": "SGPDNet_NoSGCLFA"
            },
            "no_window_attention": {
                "description": "禁用窗口注意力",
                "disabled_components": ["window_attention"],
                "model_type": "SGPDNet_NoWindowAttention"
            },
            "no_cross_layer": {
                "description": "禁用跨层聚合",
                "disabled_components": ["cross_layer"],
                "model_type": "SGPDNet_NoCrossLayer"
            },
            "no_film": {
                "description": "禁用FiLM参数生成器",
                "disabled_components": ["film"],
                "model_type": "SGPDNet_NoFiLM"
            },
            "no_structure_guidance": {
                "description": "禁用结构引导（PDSLRM + 结构特征提取器）",
                "disabled_components": ["pdslrm", "structure_feature_extractor"],
                "model_type": "SGPDNet_NoStructureGuidance"
            },
            "simple_cnn_baseline": {
                "description": "简单CNN基线（仅LightCNN）",
                "disabled_components": ["all_generative_components"],
                "model_type": "SimpleLightCNNBaseline"
            }
        }

    def _save_experiment_config(self):
        """保存实验配置到结果目录"""
        config_save_path = os.path.join(self.base_save_dir, 'experiment_config.json')
        experiment_info = {
            'args': vars(self.args),
            'experiments': self.experiments,
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device)
        }
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_info, f, indent=4, ensure_ascii=False)

    def run_all_experiments(self):
        """运行所有消融实验"""
        results = {}

        for exp_name, exp_config in self.experiments.items():
            # 如果设置了跳过基线，则跳过baseline实验
            if self.args.skip_baseline and exp_name == 'baseline':
                print(f"\n{'=' * 60}")
                print(f"跳过实验: {exp_name} (--skip-baseline 已启用)")
                print(f"{'=' * 60}")
                continue

            print(f"\n{'=' * 60}")
            print(f"开始消融实验: {exp_name}")
            print(f"描述: {exp_config['description']}")
            print(f"禁用组件: {exp_config.get('disabled_components', [])}")
            print(f"{'=' * 60}")

            try:
                # 运行单个实验
                exp_result = self.run_single_experiment(exp_name, exp_config)
                results[exp_name] = exp_result

                print(f"实验 {exp_name} 完成")
                print(f"最佳验证准确率: {exp_result['best_val_acc']:.4f}")

            except Exception as e:
                print(f"实验 {exp_name} 失败: {e}")
                import traceback
                traceback.print_exc()
                results[exp_name] = {'error': str(e), 'best_val_acc': 0.0}

        # 保存所有实验结果
        self._save_experiment_results(results)
        self._generate_comparison_report(results)

        return results

    def run_single_experiment(self, exp_name, exp_config):
        """运行单个消融实验"""
        # 设置随机种子
        seed_everything(self.args.seed)

        # 创建实验专用目录
        exp_save_dir = os.path.join(self.base_save_dir, exp_name)
        os.makedirs(exp_save_dir, exist_ok=True)

        # 初始化数据加载器
        train_loader, val_loader, num_classes = self._setup_data_loaders()

        # 初始化模型
        model = self._create_ablation_model(exp_config, num_classes)
        model.to(self.device)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

        # 初始化优化器和调度器
        optimizer = self._setup_optimizer(model)
        scheduler = self._setup_scheduler(optimizer)

        # 初始化损失权重
        loss_weights = self._setup_loss_weights(exp_config)

        # 初始化记录器
        loss_history = LossHistory(exp_save_dir, model, [128, 128])

        # 初始化早停器
        early_stopping = None
        if self.args.early_stopping:
            early_stopping = EarlyStopping(
                patience=self.args.patience,
                min_delta=self.args.min_delta,
                mode='max',  # 监控验证准确率，越大越好
                verbose=True
            )
            print(f"早停机制已启用 - 耐心值: {self.args.patience}, 最小改善阈值: {self.args.min_delta}")
        else:
            print("早停机制未启用，将训练全部epoch")

        # 训练循环
        best_val_acc = 0.0
        best_metrics = {}

        for epoch in range(self.args.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.args.epochs} [{exp_name}] ---")

            # 训练一个epoch
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, epoch, loss_weights, exp_config
            )

            # 验证
            val_metrics = self._validate_epoch(model, val_loader, epoch)

            # 更新学习率
            scheduler.step()

            # 记录指标
            all_metrics = {**train_metrics, **val_metrics}
            loss_history.append_loss(epoch, all_metrics)

            # 保存最佳模型
            if val_metrics['val_acc'] > best_val_acc:
                best_val_acc = val_metrics['val_acc']
                best_metrics = all_metrics.copy()
                best_model_path = os.path.join(exp_save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'metrics': best_metrics
                }, best_model_path)

            # 定期保存检查点
            if (epoch + 1) % self.args.save_freq == 0:
                checkpoint_path = os.path.join(exp_save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': all_metrics
                }, checkpoint_path)

            print(f"Train Acc: {train_metrics['train_acc']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.4f}, "
                  f"Best Val Acc: {best_val_acc:.4f}")

            # 早停检查
            if early_stopping is not None:
                if early_stopping(val_metrics['val_acc'], epoch):
                    print(f"\n早停触发！训练在第 {epoch + 1} 轮停止")
                    print(f"最佳模型在第 {early_stopping.best_epoch + 1} 轮，验证准确率: {early_stopping.best_value:.4f}")
                    break

        loss_history.close_writer()

        # 计算实际训练的epoch数
        actual_epochs = epoch + 1  # epoch是0-based的

        # 保存实验结果
        exp_result = {
            'exp_name': exp_name,
            'description': exp_config['description'],
            'disabled_components': exp_config.get('disabled_components', []),
            'best_val_acc': best_val_acc,
            'best_metrics': best_metrics,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_epochs': self.args.epochs,  # 计划的总epoch数
            'actual_epochs': actual_epochs,  # 实际训练的epoch数
            'early_stopped': early_stopping.early_stop if early_stopping is not None else False,
            'best_epoch': early_stopping.best_epoch + 1 if early_stopping is not None else actual_epochs
        }

        result_file = os.path.join(exp_save_dir, 'experiment_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(exp_result, f, indent=4, ensure_ascii=False)

        return exp_result

    def _setup_data_loaders(self):
        """设置数据加载器"""
        # 初始化SAM2模型
        sam_predictor = None
        if not self.args.disable_sam:
            try:
                sam_predictor = initialize_sam2_model(
                    checkpoint_path=self.args.sam_checkpoint,
                    model_cfg=self.args.sam_config,
                    device=self.device
                )
                if sam_predictor is None:
                    print("SAM2初始化失败，禁用SAM功能")
            except Exception as e:
                print(f"SAM2初始化失败: {e}")

        # 获取类别数量
        num_classes = get_num_classes(self.args.data_path)
        print(f"检测到 {num_classes} 个类别")

        # 加载数据
        with open(self.args.data_path, "r", encoding='utf-8') as f:
            lines = f.readlines()

        np.random.seed(self.args.seed)
        np.random.shuffle(lines)
        np.random.seed(None)

        val_split = 0.15
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val

        # 创建数据集
        train_dataset = SGPDDataset(
            input_shape_hw=[128, 128],
            annotation_lines=lines[:num_train],
            train_mode=True,
            sam_predictor=sam_predictor
        )
        val_dataset = SGPDDataset(
            input_shape_hw=[128, 128],
            annotation_lines=lines[num_train:],
            train_mode=False,
            sam_predictor=sam_predictor
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=sgpd_dataset_collate,
            worker_init_fn=partial(worker_init_fn, seed=self.args.seed)
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=sgpd_dataset_collate,
            worker_init_fn=partial(worker_init_fn, seed=self.args.seed)
        )

        return train_loader, val_loader, num_classes

    def _create_ablation_model(self, exp_config, num_classes):
        """创建消融模型 - 支持从配置文件读取模型参数"""
        model_type = exp_config.get('model_type', 'SGPDNet')
        model_params = exp_config.get('model_params', {})

        # 默认参数
        default_params = {
            'latent_channels': 4,
            'structure_dim': 256,
            'lightcnn_embedding_dim': 512,
            'sub_centers_k': 2,
            'arcface_s': 12,
            'arcface_m': 0.15,
            'latent_inversion_iters': 30,
            'latent_inversion_lr': 0.01,
            'window_size': 7
        }

        # 用配置文件中的参数覆盖默认参数
        for key, value in model_params.items():
            default_params[key] = value

        if model_type == 'SGPDNet':
            # 完整模型
            model = SGPDNet(
                num_classes=num_classes,
                latent_channels=default_params['latent_channels'],
                structure_dim=default_params['structure_dim'],
                lightcnn_embedding_dim=default_params['lightcnn_embedding_dim'],
                pretrained=False,
                sub_centers_k=default_params['sub_centers_k'],
                arcface_s=default_params['arcface_s'],
                arcface_m=default_params['arcface_m'],
                latent_inversion_iters=default_params['latent_inversion_iters'],
                latent_inversion_lr=default_params['latent_inversion_lr'],
                window_size=default_params['window_size']
            )
        elif model_type == 'SGPDNet_NoPDSLRM':
            model = SGPDNet_NoPDSLRM(
                num_classes=num_classes,
                latent_channels=default_params['latent_channels'],
                structure_dim=default_params['structure_dim'],
                lightcnn_embedding_dim=default_params['lightcnn_embedding_dim'],
                window_size=default_params['window_size']
            )
        elif model_type == 'SGPDNet_NoSGCLFA':
            model = SGPDNet_NoSGCLFA(
                num_classes=num_classes,
                latent_channels=default_params['latent_channels'],
                structure_dim=default_params['structure_dim'],
                lightcnn_embedding_dim=default_params['lightcnn_embedding_dim']
            )
        elif model_type == 'SGPDNet_NoWindowAttention':
            model = SGPDNet_NoWindowAttention(
                num_classes=num_classes,
                latent_channels=default_params['latent_channels'],
                structure_dim=default_params['structure_dim'],
                lightcnn_embedding_dim=default_params['lightcnn_embedding_dim']
            )
        elif model_type == 'SGPDNet_NoCrossLayer':
            model = SGPDNet_NoCrossLayer(
                num_classes=num_classes,
                latent_channels=default_params['latent_channels'],
                structure_dim=default_params['structure_dim'],
                lightcnn_embedding_dim=default_params['lightcnn_embedding_dim'],
                window_size=default_params['window_size']
            )
        elif model_type == 'SGPDNet_NoFiLM':
            model = SGPDNet_NoFiLM(
                num_classes=num_classes,
                latent_channels=default_params['latent_channels'],
                structure_dim=default_params['structure_dim'],
                lightcnn_embedding_dim=default_params['lightcnn_embedding_dim'],
                window_size=default_params['window_size']
            )
        elif model_type == 'SGPDNet_NoStructureGuidance':
            model = SGPDNet_NoStructureGuidance(
                num_classes=num_classes,
                lightcnn_embedding_dim=default_params['lightcnn_embedding_dim']
            )
        elif model_type == 'SimpleLightCNNBaseline':
            model = SimpleLightCNNBaseline(
                num_classes=num_classes,
                lightcnn_embedding_dim=default_params['lightcnn_embedding_dim']
            )
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

        return model

    def _setup_optimizer(self, model):
        """设置优化器"""
        return optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=1e-3)

    def _setup_scheduler(self, optimizer):
        """设置学习率调度器"""
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=self.args.lr * 0.1)

    def _setup_loss_weights(self, exp_config):
        """设置损失权重"""
        default_weights = {
            'rec': 0.3,
            'lsc': 0.8,
            'g_disc': 0.1,
            'center': 0.03,
            'domain_adapt': 0.0
        }

        # 从实验配置中覆盖权重
        loss_config = exp_config.get('loss_config', {})
        for key, value in loss_config.items():
            weight_key = key.replace('_weight', '')
            if weight_key in default_weights:
                default_weights[weight_key] = value

        return default_weights

    def _train_epoch(self, model, train_loader, optimizer, epoch, loss_weights, exp_config):
        """训练一个epoch - 修复版本"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        valid_batches = 0  # 添加有效批次计数

        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}')

        for batch_idx, batch_data in enumerate(pbar):
            if batch_data is None or batch_data[0] is None:
                continue

            images1, images2, subject_masks, labels = batch_data
            images1 = images1.to(self.device)
            images2 = images2.to(self.device)
            subject_masks = subject_masks.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            try:
                # 修复：正确调用不同类型的模型
                model_type = type(model).__name__

                if model_type == 'SimpleLightCNNBaseline':
                    # 简单基线模型，直接传入图像
                    logits = model(images1)
                elif model_type == 'SGPDNet':
                    # 完整SGPD模型
                    logits = model(images1, images2, subject_masks, labels, mode="train")
                elif model_type in ['SGPDNet_NoStructureGuidance']:
                    # 无结构引导的模型
                    logits = model(images1, img2=images2, subject_mask=subject_masks, label=labels, mode="train")
                else:
                    # 其他SGPD系列消融模型
                    # 尝试标准SGPD调用方式
                    try:
                        logits = model(images1, img2=images2, subject_mask=subject_masks, label=labels, mode="train")
                    except TypeError as te:
                        # 如果上述调用失败，尝试简化的调用方式
                        print(f"调用失败，尝试简化调用: {te}")
                        try:
                            logits = model(images1, subject_mask=subject_masks, label=labels, mode="train")
                        except TypeError:
                            # 最后的尝试：只传入图像
                            logits = model(images1)

                # 检查logits是否为None
                if logits is None:
                    print(f"警告: 模型返回None logits，跳过此批次")
                    continue

                # 计算损失
                loss = self._compute_total_loss(
                    model, logits, labels, images1, loss_weights, exp_config
                )

                # 检查loss是否为None或NaN
                if loss is None or torch.isnan(loss):
                    print(f"警告: 损失为None或NaN，跳过此批次")
                    continue

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 统计
                total_loss += loss.item()
                with torch.no_grad():
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    valid_batches += 1

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%' if total > 0 else '0.00%'
                })

            except Exception as e:
                print(f"训练批次出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        pbar.close()

        # 修复：避免除零错误
        if valid_batches == 0 or total == 0:
            print("警告: 没有有效的训练批次")
            return {
                'train_loss': 0.0,
                'train_acc': 0.0,
                'train_reid_l': 0.0,
                'train_rec_l': 0.0,
                'train_lsc_l': 0.0,
                'train_gdisc_l': 0.0,
                'train_center_l': 0.0,
                'train_domain_l': 0.0
            }

        avg_loss = total_loss / valid_batches
        avg_acc = correct / total

        return {
            'train_loss': avg_loss,
            'train_acc': avg_acc,
            'train_reid_l': avg_loss,  # 简化
            'train_rec_l': 0.0,
            'train_lsc_l': 0.0,
            'train_gdisc_l': 0.0,
            'train_center_l': 0.0,
            'train_domain_l': 0.0
        }

    def _validate_epoch(self, model, val_loader, epoch):
        """验证一个epoch - 修复版本"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        valid_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}')

            for batch_data in pbar:
                if batch_data is None or batch_data[0] is None:
                    continue

                images1, images2, subject_masks, labels = batch_data
                images1 = images1.to(self.device)
                images2 = images2.to(self.device)
                subject_masks = subject_masks.to(self.device)
                labels = labels.to(self.device)

                try:
                    # 修复：正确调用不同类型的模型进行验证
                    model_type = type(model).__name__

                    if model_type == 'SimpleLightCNNBaseline':
                        # 简单基线模型，直接传入图像
                        logits = model(images1)
                    elif model_type == 'SGPDNet':
                        # 完整SGPD模型
                        logits = model(images1, images2, subject_masks, labels, mode="eval_logits")
                    elif model_type in ['SGPDNet_NoStructureGuidance']:
                        # 无结构引导的模型
                        logits = model(images1, img2=images2, subject_mask=subject_masks, label=labels, mode="eval_logits")
                    else:
                        # 其他SGPD系列消融模型
                        # 尝试标准SGPD调用方式
                        try:
                            logits = model(images1, img2=images2, subject_mask=subject_masks, label=labels, mode="eval_logits")
                        except TypeError:
                            # 如果上述调用失败，尝试简化的调用方式
                            try:
                                logits = model(images1, subject_mask=subject_masks, label=labels, mode="eval_logits")
                            except TypeError:
                                # 最后的尝试：只传入图像
                                logits = model(images1)

                    if logits is None:
                        continue

                    loss = F.cross_entropy(logits, labels)

                    total_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    valid_batches += 1

                    pbar.set_postfix({
                        'Val Loss': f'{loss.item():.4f}',
                        'Val Acc': f'{100. * correct / total:.2f}%' if total > 0 else '0.00%'
                    })

                except Exception as e:
                    print(f"验证批次出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            pbar.close()

        # 修复：避免除零错误
        if valid_batches == 0 or total == 0:
            print("警告: 没有有效的验证批次")
            return {
                'val_loss': 0.0,
                'val_acc': 0.0,
                'val_reid_l': 0.0
            }

        avg_loss = total_loss / valid_batches
        avg_acc = correct / total

        return {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'val_reid_l': avg_loss
        }

    def _compute_total_loss(self, model, logits, labels, images, loss_weights, exp_config):
        """计算总损失"""
        # 基础分类损失
        if logits is None:
            return None

        reid_loss = F.cross_entropy(logits, labels)
        total_loss = reid_loss

        # 检查是否禁用某些损失
        disabled_components = exp_config.get('disabled_components', [])

        # 重建损失
        if 'reconstruction_loss' not in disabled_components and hasattr(model, 'compute_reconstruction_loss'):
            try:
                rec_loss = model.compute_reconstruction_loss(images)
                if rec_loss is not None and not torch.isnan(rec_loss):
                    total_loss += loss_weights.get('rec', 0.0) * rec_loss
            except:
                pass

        # LSC损失
        if 'lsc_loss' not in disabled_components and hasattr(model, 'compute_lsc_loss'):
            try:
                lsc_loss = model.compute_lsc_loss()
                if lsc_loss is not None and not torch.isnan(lsc_loss):
                    total_loss += loss_weights.get('lsc', 0.0) * lsc_loss
            except:
                pass

        # 辅助损失
        if 'auxiliary_loss' not in disabled_components and hasattr(model, 'compute_auxiliary_loss'):
            try:
                aux_loss = model.compute_auxiliary_loss(labels)
                if aux_loss is not None and not torch.isnan(aux_loss):
                    total_loss += loss_weights.get('g_disc', 0.0) * aux_loss
            except:
                pass

        # Center Loss
        if 'center_loss' not in disabled_components and hasattr(model, 'compute_center_loss'):
            try:
                center_loss = model.compute_center_loss(labels)
                if center_loss is not None and not torch.isnan(center_loss):
                    total_loss += loss_weights.get('center', 0.0) * center_loss
            except:
                pass

        # 域适应损失 - 注意：当前消融模型中大部分没有实现域适应
        # 如果需要对比域适应的影响，需要在相应的消融模型中添加此功能
        if 'domain_adaptation_loss' not in disabled_components and hasattr(model, 'compute_domain_adaptation_loss'):
            try:
                # 这里仅使用原始特征，不构建增强分支
                if hasattr(model, 'fd') and model.fd is not None:
                    batch_size = model.fd.size(0)
                    domain_labels = torch.zeros(batch_size, dtype=torch.long, device=model.fd.device)
                    domain_loss = model.compute_domain_adaptation_loss(model.fd, domain_labels)
                    if domain_loss is not None and not torch.isnan(domain_loss):
                        total_loss += loss_weights.get('domain_adapt', 0.0) * domain_loss
            except:
                pass

        return total_loss

    def _save_experiment_results(self, results):
        """保存所有实验结果"""
        results_file = os.path.join(self.base_save_dir, 'all_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"所有实验结果已保存到: {results_file}")

    def _generate_comparison_report(self, results):
        """生成对比报告"""
        report_file = os.path.join(self.base_save_dir, 'comparison_report.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"SGPD-Net 消融实验对比报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 80 + "\n\n")

            # 按准确率排序
            sorted_results = sorted(results.items(),
                                    key=lambda x: x[1].get('best_val_acc', 0),
                                    reverse=True)

            f.write("实验结果排序（按验证准确率）:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'排名':<4} {'实验名称':<25} {'验证准确率':<12} {'参数量':<15} {'描述'}\n")
            f.write("-" * 80 + "\n")

            for rank, (exp_name, result) in enumerate(sorted_results, 1):
                if 'error' in result:
                    f.write(f"{rank:<4} {exp_name:<25} {'失败':<12} {'-':<15} 实验失败\n")
                else:
                    val_acc = result.get('best_val_acc', 0)
                    params = result.get('trainable_params', 0)
                    desc = result.get('description', '')[:30]
                    f.write(f"{rank:<4} {exp_name:<25} {val_acc:<12.4f} {params:<15,} {desc}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("详细分析:\n\n")

            # 基线对比
            baseline_acc = results.get('baseline', {}).get('best_val_acc', 0)
            f.write(f"基线模型准确率: {baseline_acc:.4f}\n\n")

            f.write("各消融实验相对基线的性能变化:\n")
            f.write("-" * 50 + "\n")

            for exp_name, result in sorted_results:
                if exp_name == 'baseline' or 'error' in result:
                    continue

                val_acc = result.get('best_val_acc', 0)
                diff = val_acc - baseline_acc
                diff_pct = (diff / baseline_acc * 100) if baseline_acc > 0 else 0

                f.write(f"{exp_name}: {diff:+.4f} ({diff_pct:+.2f}%)\n")

        print(f"对比报告已保存到: {report_file}")


def main():
    args = parse_args()

    # 设置随机种子
    seed_everything(args.seed)

    # 创建实验管理器
    manager = AblationExperimentManager(args)

    if args.run_all:
        # 运行所有实验
        results = manager.run_all_experiments()

        print("\n" + "=" * 60)
        print("所有消融实验完成！")
        print("结果摘要:")
        for exp_name, result in results.items():
            if 'error' in result:
                print(f"  {exp_name}: 失败 - {result['error']}")
            else:
                print(f"  {exp_name}: {result['best_val_acc']:.4f}")
        print(f"详细结果保存在: {manager.base_save_dir}")
    else:
        print("使用 --run-all 参数运行所有实验")
        print(f"可用的实验: {list(manager.experiments.keys())}")


if __name__ == "__main__":
    main()