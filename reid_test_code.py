# enhanced_ablation_reid_test.py
"""
增强版消融模型人员再识别测试代码
支持所有消融模型和完整SGPD-Net的测试评估
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score
import warnings
from collections import defaultdict, Counter
import inspect

warnings.filterwarnings('ignore')

# 导入模型相关
try:
    from models.sgpd_net import SGPDNet
    from models.ablation_models import *
    from utils.utils import cvtColor, resize_image
    from utils.sam2_integration import initialize_sam2_model
except ImportError as e:
    print(f"警告: 模块导入失败 {e}")
    print("请确保相关模块文件存在")


class UniversalModelWrapper:
    """通用模型包装器，适配不同的消融模型"""

    def __init__(self, model, model_type='unknown'):
        self.model = model
        self.model_type = model_type
        self.model_name = model.__class__.__name__
        self.device = next(model.parameters()).device

        # 检测模型能力
        self._detect_model_capabilities()

    def _detect_model_capabilities(self):
        """检测模型的能力和接口"""
        self.capabilities = {
            'has_predict_mode': False,
            'has_mask_support': False,
            'has_multiple_modes': False,
            'requires_label': False,
            'feature_dim': 512,  # 默认特征维度
            'is_simple_baseline': False
        }

        # 检查forward方法签名
        try:
            forward_sig = inspect.signature(self.model.forward)
            params = list(forward_sig.parameters.keys())

            self.capabilities['has_mask_support'] = 'subject_mask' in params
            self.capabilities['has_multiple_modes'] = 'mode' in params
            self.capabilities['requires_label'] = 'label' in params

            # 特殊处理简单基线模型
            if 'SimpleLightCNNBaseline' in self.model_name:
                self.capabilities['is_simple_baseline'] = True
                self.capabilities['requires_label'] = True

        except Exception as e:
            print(f"警告: 无法检测模型 {self.model_name} 的接口: {e}")

        print(f"模型 {self.model_name} 能力检测:")
        for key, value in self.capabilities.items():
            print(f"  {key}: {value}")

    def extract_features(self, images, masks=None, dummy_labels=None):
        """统一的特征提取接口"""
        self.model.eval()

        with torch.no_grad():
            try:
                if self.capabilities['is_simple_baseline']:
                    # 简单基线模型：需要标签，但我们只要特征
                    if dummy_labels is None:
                        dummy_labels = torch.zeros(images.size(0), dtype=torch.long, device=self.device)

                    # 直接访问特征而不是logits
                    if hasattr(self.model, 'lightcnn'):
                        # 转换输入范围并提取特征
                        img_input = (images + 1.0) / 2.0
                        features = self.model.lightcnn(img_input)
                        return F.normalize(features, p=2, dim=1)
                    else:
                        # 回退方案
                        logits = self.model(images, label=dummy_labels)
                        return F.normalize(logits, p=2, dim=1)

                elif self.capabilities['has_multiple_modes']:
                    # 支持多模式的模型（大部分消融模型）
                    try:
                        features = self.model(images, subject_mask=masks, mode="predict")
                        return F.normalize(features, p=2, dim=1)
                    except Exception as e:
                        print(f"predict模式失败，尝试其他方法: {e}")
                        # 尝试直接访问特征
                        if hasattr(self.model, 'lightcnn'):
                            img_input = (images + 1.0) / 2.0
                            if masks is not None:
                                features = self.model.lightcnn(img_input)
                            else:
                                features = self.model.lightcnn(img_input)
                            return F.normalize(features, p=2, dim=1)
                        else:
                            raise e

                else:
                    # 其他模型类型
                    if hasattr(self.model, 'lightcnn'):
                        img_input = (images + 1.0) / 2.0
                        features = self.model.lightcnn(img_input)
                        return F.normalize(features, p=2, dim=1)
                    else:
                        # 最后的回退方案
                        output = self.model(images)
                        if isinstance(output, tuple):
                            features = output[0]
                        else:
                            features = output
                        return F.normalize(features, p=2, dim=1)

            except Exception as e:
                print(f"特征提取失败: {e}")
                print(f"模型类型: {self.model_name}")
                print(f"输入形状: {images.shape}")

                # 应急回退：返回随机特征
                batch_size = images.size(0)
                features = torch.randn(batch_size, 512, device=self.device)
                return F.normalize(features, p=2, dim=1)


class NumpyJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyJSONEncoder, self).default(obj)


def load_training_identities(annotation_path):
    """从训练标注文件中加载训练时见过的身份ID映射"""
    if not os.path.exists(annotation_path):
        print(f"警告: 训练标注文件不存在: {annotation_path}")
        return {}, {}

    identity_to_classid = {}
    classid_to_identity = {}

    with open(annotation_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(';')
        if len(parts) >= 2:
            class_id = int(parts[0])
            image_path = parts[1]

            # 从路径中提取身份文件夹名称
            path_parts = image_path.replace('\\', '/').split('/')
            identity_folder = None

            # 找到 'train' 文件夹后的下一个文件夹
            for i, part in enumerate(path_parts):
                if part == 'train' and i + 1 < len(path_parts):
                    identity_folder = path_parts[i + 1]
                    break

            if identity_folder:
                identity_to_classid[identity_folder] = class_id
                classid_to_identity[class_id] = identity_folder

    print(f"从训练集中加载了 {len(identity_to_classid)} 个身份ID")
    return classid_to_identity, identity_to_classid


class ReIDTestDataset(Dataset):
    """人员再识别测试数据集"""

    def __init__(self, data_root, subset_type='probe', input_shape=(128, 128),
                 sam_predictor=None, transform=None):
        self.data_root = Path(data_root)
        self.subset_type = subset_type
        self.input_shape = input_shape
        self.sam_predictor = sam_predictor
        self.transform = transform

        # 构建数据列表
        self.data_list = self._build_data_list()
        print(f"加载 {subset_type} 数据集: {len(self.data_list)} 张图片")

    def _build_data_list(self):
        """构建数据列表"""
        data_list = []
        subset_dir = self.data_root / self.subset_type

        if not subset_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {subset_dir}")

        # 遍历每个身份文件夹
        identity_dirs = [d for d in subset_dir.iterdir() if d.is_dir()]
        identity_dirs.sort()

        for identity_dir in identity_dirs:
            identity_id = identity_dir.name

            # 遍历该身份下的所有图片
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(identity_dir.glob(ext))

            for img_path in image_files:
                data_list.append({
                    'image_path': str(img_path),
                    'identity_id': identity_id,
                    'image_name': img_path.name
                })

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_path = item['image_path']
        identity_id = item['identity_id']

        try:
            # 读取并预处理图像
            image_pil = Image.open(image_path)
            image_pil = cvtColor(image_pil)
            image_resized = resize_image(image_pil,
                                         (self.input_shape[1], self.input_shape[0]),
                                         letterbox_image=True)

            # 转换为numpy数组
            image_np = np.array(image_resized)

            # 获取SAM掩码（如果可用）
            subject_mask = None
            if self.sam_predictor is not None:
                try:
                    subject_mask = self._get_sam_mask(image_resized)
                except Exception as e:
                    print(f"SAM分割失败 {image_path}: {e}")
                    h, w = image_np.shape[:2]
                    subject_mask = np.ones((h, w), dtype=np.uint8) * 255

            # 标准化到[-1, 1]
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            image_tensor = (image_tensor - 0.5) / 0.5

            if subject_mask is not None:
                mask_tensor = torch.from_numpy(subject_mask.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                mask_tensor = torch.ones(1, self.input_shape[0], self.input_shape[1])

            return {
                'image': image_tensor,
                'mask': mask_tensor,
                'identity_id': identity_id,
                'image_path': image_path,
                'image_name': item['image_name']
            }

        except Exception as e:
            print(f"处理图像失败 {image_path}: {e}")
            # 返回空白数据
            return {
                'image': torch.zeros(3, self.input_shape[0], self.input_shape[1]),
                'mask': torch.ones(1, self.input_shape[0], self.input_shape[1]),
                'identity_id': identity_id,
                'image_path': image_path,
                'image_name': item['image_name']
            }

    def _get_sam_mask(self, image_pil):
        """获取SAM分割掩码"""
        from utils.dataloader_sgpd import get_sam_segmentation_mask
        mask_pil = get_sam_segmentation_mask(image_pil, self.sam_predictor)
        return np.array(mask_pil)


class EnhancedReIDEvaluator:
    """增强版人员再识别评估器，支持所有消融模型"""

    def __init__(self, model, device='cuda', training_identities=None, model_type='unknown'):
        self.model_wrapper = UniversalModelWrapper(model, model_type)
        self.device = device
        self.training_identities = training_identities or {}

    def extract_features(self, dataloader, desc="提取特征"):
        """提取特征"""
        features = []
        identities = []
        image_paths = []
        image_names = []

        for batch in tqdm(dataloader, desc=desc):
            if batch is None:
                continue

            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device) if 'mask' in batch else None

            # 使用通用包装器提取特征
            try:
                # 为简单基线模型创建虚拟标签
                dummy_labels = None
                if self.model_wrapper.capabilities['requires_label']:
                    dummy_labels = torch.zeros(images.size(0), dtype=torch.long, device=self.device)

                batch_features = self.model_wrapper.extract_features(images, masks, dummy_labels)
                features.append(batch_features.cpu())

            except Exception as e:
                print(f"特征提取失败: {e}")
                # 使用零特征作为备选
                batch_features = torch.zeros(images.size(0), 512)
                features.append(batch_features)

            identities.extend(batch['identity_id'])
            image_paths.extend(batch['image_path'])
            image_names.extend(batch['image_name'])

        if features:
            features = torch.cat(features, dim=0)
        else:
            features = torch.empty(0, 512)

        return features, identities, image_paths, image_names

    def classify_identities(self, identities):
        """将身份分类为闭集（训练时见过）和开集（新身份）"""
        closed_set_ids = []
        open_set_ids = []

        for identity in set(identities):
            if identity in self.training_identities:
                closed_set_ids.append(identity)
            else:
                open_set_ids.append(identity)

        return {
            'closed_set': closed_set_ids,
            'open_set': open_set_ids,
            'total_closed': len(closed_set_ids),
            'total_open': len(open_set_ids)
        }

    def compute_distance_matrix(self, probe_features, gallery_features, metric='cosine'):
        """计算距离矩阵"""
        if metric == 'cosine':
            # 余弦距离
            probe_norm = F.normalize(probe_features, p=2, dim=1)
            gallery_norm = F.normalize(gallery_features, p=2, dim=1)
            similarity = torch.mm(probe_norm, gallery_norm.t())
            distance = 1 - similarity
        elif metric == 'euclidean':
            # 欧氏距离
            probe_features = probe_features.unsqueeze(1)  # (N, 1, D)
            gallery_features = gallery_features.unsqueeze(0)  # (1, M, D)
            distance = torch.sqrt(torch.sum((probe_features - gallery_features) ** 2, dim=2))
        else:
            raise ValueError(f"未知距离度量: {metric}")

        return distance.numpy()

    def evaluate_ranking(self, distance_matrix, probe_ids, gallery_ids, max_rank=10):
        """评估排序性能"""
        num_probe = distance_matrix.shape[0]
        num_gallery = distance_matrix.shape[1]

        if num_probe == 0 or num_gallery == 0:
            return {
                'cmc': np.zeros(max_rank),
                'mAP': 0.0,
                'rank1': 0.0,
                'rank5': 0.0,
                'rank10': 0.0,
                'num_valid_queries': 0
            }

        # 转换身份ID为数字
        unique_ids = list(set(probe_ids + gallery_ids))
        id_to_num = {id_str: idx for idx, id_str in enumerate(unique_ids)}

        probe_ids_num = [id_to_num[pid] for pid in probe_ids]
        gallery_ids_num = [id_to_num[gid] for gid in gallery_ids]

        # CMC计算
        cmc = np.zeros(max_rank)
        ap_list = []

        for i in range(num_probe):
            distances = distance_matrix[i]
            sorted_indices = np.argsort(distances)

            probe_id = probe_ids_num[i]
            matches = np.array([gallery_ids_num[idx] == probe_id for idx in sorted_indices])

            if np.sum(matches) > 0:
                ap = average_precision_score(matches.astype(int), 1 - distances[sorted_indices])
                ap_list.append(ap)

                # 更新CMC
                first_match_idx = np.where(matches)[0]
                if len(first_match_idx) > 0:
                    first_match = first_match_idx[0]
                    for k in range(first_match, max_rank):
                        cmc[k] += 1

        # 计算最终指标
        cmc = cmc / num_probe if num_probe > 0 else cmc
        mAP = np.mean(ap_list) if ap_list else 0.0

        return {
            'cmc': cmc,
            'mAP': mAP,
            'rank1': cmc[0],
            'rank5': cmc[4] if max_rank > 4 else cmc[-1],
            'rank10': cmc[9] if max_rank > 9 else cmc[-1],
            'num_valid_queries': len(ap_list)
        }

    def evaluate_by_set_type(self, distance_matrix, probe_ids, gallery_ids, max_rank=10):
        """分别评估闭集和开集性能"""
        results = {}

        # 获取各个子集的索引
        closed_probe_indices = [i for i, pid in enumerate(probe_ids) if pid in self.training_identities]
        open_probe_indices = [i for i, pid in enumerate(probe_ids) if pid not in self.training_identities]

        closed_gallery_indices = [i for i, gid in enumerate(gallery_ids) if gid in self.training_identities]
        open_gallery_indices = [i for i, gid in enumerate(gallery_ids) if gid not in self.training_identities]

        # 1. 闭集对闭集
        if closed_probe_indices and closed_gallery_indices:
            c2c_distance = distance_matrix[np.ix_(closed_probe_indices, closed_gallery_indices)]
            c2c_probe_ids = [probe_ids[i] for i in closed_probe_indices]
            c2c_gallery_ids = [gallery_ids[i] for i in closed_gallery_indices]

            results['closed_to_closed'] = self.evaluate_ranking(
                c2c_distance, c2c_probe_ids, c2c_gallery_ids, max_rank
            )
            results['closed_to_closed']['sample_counts'] = {
                'probe_samples': len(closed_probe_indices),
                'gallery_samples': len(closed_gallery_indices)
            }

        # 2. 开集对开集
        if open_probe_indices and open_gallery_indices:
            o2o_distance = distance_matrix[np.ix_(open_probe_indices, open_gallery_indices)]
            o2o_probe_ids = [probe_ids[i] for i in open_probe_indices]
            o2o_gallery_ids = [gallery_ids[i] for i in open_gallery_indices]

            results['open_to_open'] = self.evaluate_ranking(
                o2o_distance, o2o_probe_ids, o2o_gallery_ids, max_rank
            )
            results['open_to_open']['sample_counts'] = {
                'probe_samples': len(open_probe_indices),
                'gallery_samples': len(open_gallery_indices)
            }

        # 3. 全集评估
        results['all_to_all'] = self.evaluate_ranking(
            distance_matrix, probe_ids, gallery_ids, max_rank
        )
        results['all_to_all']['sample_counts'] = {
            'probe_samples': len(probe_ids),
            'gallery_samples': len(gallery_ids)
        }

        return results


def load_ablation_model(model_path, model_type='auto', device='cuda'):
    """加载消融模型"""

    # 自动检测模型类型
    if model_type == 'auto':
        if 'simple_cnn' in model_path.lower() or 'baseline' in model_path.lower():
            model_type = 'SimpleLightCNNBaseline'
        elif 'no_pdslrm' in model_path.lower():
            model_type = 'SGPDNet_NoPDSLRM'
        elif 'no_sgclfa' in model_path.lower():
            model_type = 'SGPDNet_NoSGCLFA'
        elif 'no_film' in model_path.lower():
            model_type = 'SGPDNet_NoFiLM'
        elif 'no_structure' in model_path.lower():
            model_type = 'SGPDNet_NoStructureGuidance'
        else:
            model_type = 'SGPDNet'  # 默认完整模型

    print(f"检测到模型类型: {model_type}")

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 获取类别数量（从检查点或默认值）
    if isinstance(checkpoint, dict):
        num_classes = checkpoint.get('num_classes', 45)  # 默认45个类别
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        num_classes = 45  # 默认值

    # 创建模型
    try:
        if model_type == 'SimpleLightCNNBaseline':
            model = SimpleLightCNNBaseline(num_classes=num_classes)
        elif model_type == 'SGPDNet_NoPDSLRM':
            model = SGPDNet_NoPDSLRM(num_classes=num_classes)
        elif model_type == 'SGPDNet_NoSGCLFA':
            model = SGPDNet_NoSGCLFA(num_classes=num_classes)
        elif model_type == 'SGPDNet_NoFiLM':
            model = SGPDNet_NoFiLM(num_classes=num_classes)
        elif model_type == 'SGPDNet_NoStructureGuidance':
            model = SGPDNet_NoStructureGuidance(num_classes=num_classes)
        elif model_type == 'SGPDNet_NoWindowAttention':
            model = SGPDNet_NoWindowAttention(num_classes=num_classes)
        elif model_type == 'SGPDNet_NoCrossLayer':
            model = SGPDNet_NoCrossLayer(num_classes=num_classes)
        else:
            # 完整SGPDNet或其他
            model = SGPDNet(num_classes=num_classes)

    except Exception as e:
        print(f"创建模型失败: {e}")
        print("回退到简单基线模型...")
        model = SimpleLightCNNBaseline(num_classes=num_classes)
        model_type = 'SimpleLightCNNBaseline'

    # 加载权重
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f"模型权重加载成功")
    except Exception as e:
        print(f"权重加载警告: {e}")
        print("继续使用部分加载的权重...")

    model.to(device)
    model.eval()

    print(f"模型加载完成: {model_type}, 设备: {device}")
    return model, model_type


def collate_fn(batch):
    """自定义数据整理函数"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])

    return {
        'image': images,
        'mask': masks,
        'identity_id': [item['identity_id'] for item in batch],
        'image_path': [item['image_path'] for item in batch],
        'image_name': [item['image_name'] for item in batch]
    }


def main():
    parser = argparse.ArgumentParser(description='增强版消融模型人员再识别测试')
    parser.add_argument('--model-path', type=str, help='模型权重路径',default='logs/swin_sam_win4_iters4_k2_s12_m0.15_rec0.3_lsc0.8_gdisc0.1/checkpoint_best.pth')
    parser.add_argument('--model-type', type=str, default='auto',
                        choices=['auto', 'SGPDNet', 'SimpleLightCNNBaseline', 'SGPDNet_NoPDSLRM',
                                 'SGPDNet_NoSGCLFA', 'SGPDNet_NoFiLM', 'SGPDNet_NoStructureGuidance',
                                 'SGPDNet_NoWindowAttention', 'SGPDNet_NoCrossLayer'],
                        help='模型类型（auto为自动检测）')
    parser.add_argument('--annotation-path', type=str, default="paperdata-train.txt",
                        help='训练集标注文件路径')
    parser.add_argument('--data-root', type=str, default="test-data", help='测试数据根目录')
    parser.add_argument('--save-dir', type=str, default='ablation_test_results', help='结果保存目录')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--distance-metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean'], help='距离度量方法')
    parser.add_argument('--max-rank', type=int, default=20, help='最大排序位置')
    parser.add_argument('--disable-sam', action='store_true', help='禁用SAM分割')
    parser.add_argument('--sam-checkpoint', type=str, default="E:\Mypaper2\conf\sam2.1_hiera_tiny.pt",
                        help='SAM模型检查点路径')
    parser.add_argument('--sam-config', type=str, default="E:\Mypaper2\conf\sam2.1_hiera_t.yaml",
                        help='SAM模型配置文件路径')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载训练集身份信息
    print("加载训练集身份信息...")
    classid_to_identity, identity_to_classid = load_training_identities(args.annotation_path)

    # 初始化SAM模型（如果需要）
    sam_predictor = None
    if not args.disable_sam and os.path.exists(args.sam_checkpoint):
        print("初始化SAM模型...")
        sam_predictor = initialize_sam2_model(
            checkpoint_path=args.sam_checkpoint,
            model_cfg=args.sam_config,
            device=device
        )
        if sam_predictor is None:
            print("SAM模型初始化失败，继续使用无掩码模式")

    # 加载模型
    print("加载消融模型...")
    model, detected_model_type = load_ablation_model(args.model_path, args.model_type, device)

    # 创建数据集
    print("创建测试数据集...")
    probe_dataset = ReIDTestDataset(
        data_root=args.data_root,
        subset_type='probe',
        sam_predictor=sam_predictor
    )

    gallery_dataset = ReIDTestDataset(
        data_root=args.data_root,
        subset_type='gallery',
        sam_predictor=sam_predictor
    )

    # 创建数据加载器
    probe_loader = DataLoader(
        probe_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )

    gallery_loader = DataLoader(
        gallery_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )

    # 创建评估器
    evaluator = EnhancedReIDEvaluator(model, device, identity_to_classid, detected_model_type)

    # 提取特征
    print("提取probe特征...")
    probe_features, probe_ids, probe_paths, probe_names = evaluator.extract_features(
        probe_loader, "提取probe特征"
    )

    print("提取gallery特征...")
    gallery_features, gallery_ids, gallery_paths, gallery_names = evaluator.extract_features(
        gallery_loader, "提取gallery特征"
    )

    print(f"probe特征形状: {probe_features.shape}")
    print(f"gallery特征形状: {gallery_features.shape}")

    if probe_features.shape[0] == 0 or gallery_features.shape[0] == 0:
        print("错误: 特征提取失败，没有有效特征")
        return

    # 计算距离矩阵
    print(f"计算距离矩阵 (使用{args.distance_metric}距离)...")
    distance_matrix = evaluator.compute_distance_matrix(
        probe_features, gallery_features, metric=args.distance_metric
    )

    # 评估性能
    print("评估整体性能...")
    overall_results = evaluator.evaluate_ranking(
        distance_matrix, probe_ids, gallery_ids, max_rank=args.max_rank
    )

    # 评估开集/闭集性能
    print("评估开集/闭集性能...")
    set_results = evaluator.evaluate_by_set_type(
        distance_matrix, probe_ids, gallery_ids, max_rank=args.max_rank
    )

    # 分类统计
    probe_classification = evaluator.classify_identities(probe_ids)
    gallery_classification = evaluator.classify_identities(gallery_ids)

    # 打印结果
    print("\n" + "=" * 70)
    print(f"消融模型测试结果 - {detected_model_type}")
    print("=" * 70)

    # 数据集统计
    print(f"数据集统计:")
    print(f"  Probe: {len(probe_ids)} 样本")
    print(f"    闭集: {probe_classification['total_closed']} 个身份")
    print(f"    开集: {probe_classification['total_open']} 个身份")
    print(f"  Gallery: {len(gallery_ids)} 样本")
    print(f"    闭集: {gallery_classification['total_closed']} 个身份")
    print(f"    开集: {gallery_classification['total_open']} 个身份")
    print()

    # 整体性能
    print(f"整体性能:")
    print(f"  mAP: {overall_results['mAP']:.4f}")
    print(f"  Rank-1: {overall_results['rank1']:.4f}")
    print(f"  Rank-5: {overall_results['rank5']:.4f}")
    print(f"  Rank-10: {overall_results['rank10']:.4f}")
    print(f"  有效查询数: {overall_results['num_valid_queries']}")
    print()

    # 分集合性能
    for set_name, results in set_results.items():
        if results['num_valid_queries'] > 0:
            set_display_name = {
                'closed_to_closed': '闭集对闭集',
                'open_to_open': '开集对开集',
                'all_to_all': '全集'
            }.get(set_name, set_name)

            print(f"{set_display_name}性能:")
            print(f"  mAP: {results['mAP']:.4f}")
            print(f"  Rank-1: {results['rank1']:.4f}")
            print(f"  Rank-5: {results['rank5']:.4f}")
            print(f"  Rank-10: {results['rank10']:.4f}")
            print(f"  样本数: Probe {results['sample_counts']['probe_samples']}, "
                  f"Gallery {results['sample_counts']['gallery_samples']}")
            print()

    # 生成可视化
    print("生成可视化结果...")
    generate_visualization(set_results, probe_classification, gallery_classification,
                           detected_model_type, save_dir)

    # 保存详细结果
    detailed_results = {
        'model_info': {
            'model_path': args.model_path,
            'model_type': detected_model_type,
            'model_capabilities': evaluator.model_wrapper.capabilities
        },
        'test_config': vars(args),
        'dataset_statistics': {
            'probe_classification': probe_classification,
            'gallery_classification': gallery_classification,
            'total_probe_samples': len(probe_ids),
            'total_gallery_samples': len(gallery_ids)
        },
        'overall_results': overall_results,
        'set_results': set_results,
        'feature_statistics': {
            'probe_feature_shape': list(probe_features.shape),
            'gallery_feature_shape': list(gallery_features.shape),
            'feature_mean': float(probe_features.mean()),
            'feature_std': float(probe_features.std())
        }
    }

    # 保存结果
    with open(save_dir / 'test_results.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)

    # 保存距离矩阵
    np.save(save_dir / 'distance_matrix.npy', distance_matrix)

    # 生成性能报告
    generate_performance_report(detailed_results, save_dir)

    print(f"\n所有结果已保存到: {save_dir}")
    print("消融模型测试完成!")


def generate_visualization(set_results, probe_classification, gallery_classification,
                           model_type, save_dir):
    """生成可视化图表"""

    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 性能对比柱状图
    ax1 = axes[0, 0]
    metrics = ['mAP', 'Rank-1', 'Rank-5', 'Rank-10']

    # 选择有效的测试集
    valid_sets = {k: v for k, v in set_results.items() if v['num_valid_queries'] > 0}

    if valid_sets:
        set_names = list(valid_sets.keys())
        set_display_names = [
            'Closed-to-Closed' if name == 'closed_to_closed' else
            'Open-to-Open' if name == 'open_to_open' else
            'All-to-All' if name == 'all_to_all' else name.replace('_', '-')
            for name in set_names
        ]

        x = np.arange(len(set_names))
        width = 0.2

        map_scores = [valid_sets[name]['mAP'] for name in set_names]
        rank1_scores = [valid_sets[name]['rank1'] for name in set_names]
        rank5_scores = [valid_sets[name]['rank5'] for name in set_names]
        rank10_scores = [valid_sets[name]['rank10'] for name in set_names]

        ax1.bar(x - 1.5 * width, map_scores, width, label='mAP', alpha=0.8)
        ax1.bar(x - 0.5 * width, rank1_scores, width, label='Rank-1', alpha=0.8)
        ax1.bar(x + 0.5 * width, rank5_scores, width, label='Rank-5', alpha=0.8)
        ax1.bar(x + 1.5 * width, rank10_scores, width, label='Rank-10', alpha=0.8)

        ax1.set_xlabel('Test Sets')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Performance Comparison - {model_type}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(set_display_names, rotation=15)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. 数据集分布饼图
    ax2 = axes[0, 1]
    probe_sizes = [probe_classification['total_closed'], probe_classification['total_open']]
    probe_labels = ['Closed Set', 'Open Set']
    colors = ['lightblue', 'lightcoral']

    if sum(probe_sizes) > 0:
        ax2.pie(probe_sizes, labels=probe_labels, colors=colors, autopct='%1.1f%%',
                startangle=90, shadow=True)
        ax2.set_title('Probe Identity Distribution')

    # 3. CMC曲线
    ax3 = axes[1, 0]
    for set_name, results in valid_sets.items():
        if 'cmc' in results and len(results['cmc']) > 0:
            ranks = np.arange(1, len(results['cmc']) + 1)
            display_name = {
                'closed_to_closed': 'Closed-to-Closed',
                'open_to_open': 'Open-to-Open',
                'all_to_all': 'All-to-All'
            }.get(set_name, set_name.replace('_', '-'))

            ax3.plot(ranks, results['cmc'], marker='o', linewidth=2,
                     label=f"{display_name} (mAP: {results['mAP']:.3f})")

    ax3.set_xlabel('Rank')
    ax3.set_ylabel('Matching Rate')
    ax3.set_title('CMC Curves')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(1, min(20, len(results['cmc']) if 'cmc' in results else 20))
    ax3.set_ylim(0, 1)

    # 4. 模型能力雷达图（如果有多个指标）
    ax4 = axes[1, 1]
    if len(valid_sets) >= 2:
        # 选择两个主要的测试集进行对比
        main_sets = ['all_to_all', 'closed_to_closed', 'open_to_open']
        available_sets = [s for s in main_sets if s in valid_sets][:2]

        if len(available_sets) >= 2:
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合圆形

            for set_name in available_sets:
                values = [
                    valid_sets[set_name]['mAP'],
                    valid_sets[set_name]['rank1'],
                    valid_sets[set_name]['rank5'],
                    valid_sets[set_name]['rank10']
                ]
                values += values[:1]  # 闭合圆形

                display_name = {
                    'closed_to_closed': 'Closed-to-Closed',
                    'open_to_open': 'Open-to-Open',
                    'all_to_all': 'All-to-All'
                }.get(set_name, set_name)

                ax4.plot(angles, values, 'o-', linewidth=2, label=display_name)
                ax4.fill(angles, values, alpha=0.1)

            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics)
            ax4.set_ylim(0, 1)
            ax4.set_title('Performance Radar Chart')
            ax4.legend()
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor radar chart',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Performance Radar Chart')
    else:
        ax4.text(0.5, 0.5, 'Insufficient test sets\nfor comparison',
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Performance Radar Chart')

    plt.tight_layout()
    plt.savefig(save_dir / f'{model_type}_performance_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 重置matplotlib参数
    plt.rcParams.update(plt.rcParamsDefault)


def generate_performance_report(results, save_dir):
    """生成性能报告"""

    report_lines = [
        f"消融模型测试报告 - {results['model_info']['model_type']}",
        "=" * 60,
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "模型信息:",
        f"  模型类型: {results['model_info']['model_type']}",
        f"  模型路径: {results['model_info']['model_path']}",
        "",
        "模型能力:",
    ]

    for capability, value in results['model_info']['model_capabilities'].items():
        report_lines.append(f"  {capability}: {value}")

    report_lines.extend([
        "",
        "数据集统计:",
        f"  Probe样本总数: {results['dataset_statistics']['total_probe_samples']}",
        f"  Gallery样本总数: {results['dataset_statistics']['total_gallery_samples']}",
        f"  Probe闭集身份: {results['dataset_statistics']['probe_classification']['total_closed']}",
        f"  Probe开集身份: {results['dataset_statistics']['probe_classification']['total_open']}",
        f"  Gallery闭集身份: {results['dataset_statistics']['gallery_classification']['total_closed']}",
        f"  Gallery开集身份: {results['dataset_statistics']['gallery_classification']['total_open']}",
        "",
        "性能结果:",
        "-" * 30,
    ])

    # 添加各项测试结果
    overall = results['overall_results']
    report_lines.extend([
        f"整体性能:",
        f"  mAP: {overall['mAP']:.4f}",
        f"  Rank-1: {overall['rank1']:.4f}",
        f"  Rank-5: {overall['rank5']:.4f}",
        f"  Rank-10: {overall['rank10']:.4f}",
        f"  有效查询数: {overall['num_valid_queries']}",
        ""
    ])

    for set_name, set_results in results['set_results'].items():
        if set_results['num_valid_queries'] > 0:
            set_display_name = {
                'closed_to_closed': '闭集对闭集',
                'open_to_open': '开集对开集',
                'all_to_all': '全集'
            }.get(set_name, set_name)

            report_lines.extend([
                f"{set_display_name}性能:",
                f"  mAP: {set_results['mAP']:.4f}",
                f"  Rank-1: {set_results['rank1']:.4f}",
                f"  Rank-5: {set_results['rank5']:.4f}",
                f"  Rank-10: {set_results['rank10']:.4f}",
                f"  Probe样本: {set_results['sample_counts']['probe_samples']}",
                f"  Gallery样本: {set_results['sample_counts']['gallery_samples']}",
                ""
            ])

    # 特征统计
    feature_stats = results['feature_statistics']
    report_lines.extend([
        "特征统计:",
        f"  Probe特征形状: {feature_stats['probe_feature_shape']}",
        f"  Gallery特征形状: {feature_stats['gallery_feature_shape']}",
        f"  特征均值: {feature_stats['feature_mean']:.6f}",
        f"  特征标准差: {feature_stats['feature_std']:.6f}",
        "",
        "=" * 60
    ])

    # 保存报告
    with open(save_dir / 'performance_report.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


if __name__ == "__main__":
    main()