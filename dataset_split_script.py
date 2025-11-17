# dataset_split.py - Re-ID数据集划分脚本
# 支持训练/测试划分，以及query/gallery划分，支持开集识别

import os
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import json
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Re-ID数据集划分脚本')
    parser.add_argument('--data-root', type=str,default='C:\\Users\PC\Desktop\Paper-Data-Copy\merged_dataset',
                        help='原始数据集根目录路径')
    parser.add_argument('--output-root', type=str, default="C:\\Users\PC\Desktop\Paper-Data-Copy\splited_dataset",
                        help='输出数据集根目录路径')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集个体比例 (默认: 0.7)')
    parser.add_argument('--query-ratio', type=float, default=0.5,
                        help='测试集中query的图片比例 (默认: 0.5)')
    parser.add_argument('--min-images-per-identity', type=int, default=2,
                        help='每个个体最少图片数量 (默认: 2)')
    parser.add_argument('--min-gallery-images', type=int, default=1,
                        help='gallery中每个个体最少图片数量 (默认: 1)')
    parser.add_argument('--min-query-images', type=int, default=1,
                        help='query中每个个体最少图片数量 (默认: 1)')
    parser.add_argument('--seed', type=int, default=3407,
                        help='随机种子 (默认: 3407)')
    parser.add_argument('--copy-files', action='store_true',
                        help='是否复制文件（默认创建软链接以节省空间）')
    parser.add_argument('--generate-annotation', action='store_true', default=False,
                        help='是否生成标注文件')
    parser.add_argument('--preview-only', action='store_true',
                        help='仅预览划分结果，不实际创建文件')

    return parser.parse_args()


class ReIDDatasetSplitter:
    def __init__(self, args):
        self.args = args
        self.data_root = Path(args.data_root)
        self.output_root = Path(args.output_root)

        # 设置随机种子
        random.seed(args.seed)

        # 创建输出目录结构
        self.train_dir = self.output_root / "train"
        self.test_dir = self.output_root / "test"
        self.query_dir = self.test_dir / "query"
        self.gallery_dir = self.test_dir / "gallery"

        # 统计信息
        self.stats = {
            'total_identities': 0,
            'total_images': 0,
            'train_identities': 0,
            'train_images': 0,
            'test_identities': 0,
            'gallery_images': 0,
            'query_images': 0,
            'discarded_identities': 0,
            'discarded_images': 0
        }

    def scan_dataset(self):
        """扫描数据集，获取所有个体和图片信息"""
        print("正在扫描数据集...")

        identity_images = defaultdict(list)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        # 扫描所有个体文件夹
        for identity_dir in self.data_root.iterdir():
            if not identity_dir.is_dir():
                continue

            identity_name = identity_dir.name
            images = []

            # 扫描该个体的所有图片
            for img_file in identity_dir.iterdir():
                if img_file.suffix.lower() in valid_extensions:
                    images.append(img_file)

            if len(images) >= self.args.min_images_per_identity:
                identity_images[identity_name] = images
            else:
                print(f"警告: 个体 {identity_name} 只有 {len(images)} 张图片，少于最小要求 {self.args.min_images_per_identity}，将被跳过")
                self.stats['discarded_identities'] += 1
                self.stats['discarded_images'] += len(images)

        # 更新统计信息
        self.stats['total_identities'] = len(identity_images)
        self.stats['total_images'] = sum(len(images) for images in identity_images.values())

        print(f"扫描完成:")
        print(f"  - 有效个体数量: {self.stats['total_identities']}")
        print(f"  - 有效图片数量: {self.stats['total_images']}")
        print(f"  - 跳过个体数量: {self.stats['discarded_identities']}")
        print(f"  - 跳过图片数量: {self.stats['discarded_images']}")

        return identity_images

    def split_identities(self, identity_images):
        """将个体划分为训练集和测试集（个体级别划分，确保不交叉）"""
        print("\n正在随机划分个体...")

        all_identities = list(identity_images.keys())
        # 随机打乱所有个体ID
        random.shuffle(all_identities)

        # 计算训练集个体数量
        num_train_identities = int(len(all_identities) * self.args.train_ratio)

        # 确保至少有一个测试集个体
        if num_train_identities >= len(all_identities):
            num_train_identities = len(all_identities) - 1

        # 随机选取的前num_train_identities个作为训练集
        train_identities = all_identities[:num_train_identities]
        # 剩余的作为测试集
        test_identities = all_identities[num_train_identities:]

        # 更新统计信息
        self.stats['train_identities'] = len(train_identities)
        self.stats['test_identities'] = len(test_identities)

        print(f"个体划分完成 (随机种子: {self.args.seed}):")
        print(f"  - 训练集个体数量: {len(train_identities)} ({len(train_identities) / len(all_identities):.1%})")
        print(f"  - 测试集个体数量: {len(test_identities)} ({len(test_identities) / len(all_identities):.1%})")
        print(f"  - 个体完全不交叉: 训练集和测试集没有共同的个体")
        print(f"  - 选取方式: 随机选取")

        return train_identities, test_identities

    def split_test_images(self, identity_images, test_identities):
        """将测试集个体的图片随机划分为query和gallery"""
        print("\n正在随机划分测试集图片...")

        gallery_data = {}
        query_data = {}

        for identity in test_identities:
            images = identity_images[identity].copy()
            # 随机打乱每个个体的所有图片
            random.shuffle(images)

            total_images = len(images)

            # 确保gallery中至少有最小数量的图片
            min_required = self.args.min_gallery_images + self.args.min_query_images
            if total_images < min_required:
                print(f"警告: 个体 {identity} 只有 {total_images} 张图片，不足以同时满足query和gallery的最小要求")
                # 至少保证gallery有一张图片
                gallery_data[identity] = images[:1]
                query_data[identity] = images[1:]
                continue

            # 计算query的图片数量
            query_count = max(
                self.args.min_query_images,
                int(total_images * self.args.query_ratio)
            )

            # 确保gallery至少有最小数量的图片
            gallery_count = total_images - query_count
            if gallery_count < self.args.min_gallery_images:
                gallery_count = self.args.min_gallery_images
                query_count = total_images - gallery_count

            # 随机分配图片（由于已经shuffle过，直接切片即可）
            query_images = images[:query_count]
            gallery_images = images[query_count:]

            gallery_data[identity] = gallery_images
            query_data[identity] = query_images

            self.stats['gallery_images'] += len(gallery_images)
            self.stats['query_images'] += len(query_images)

        print(f"测试集图片划分完成 (随机划分):")
        print(f"  - Query图片数量: {self.stats['query_images']}")
        print(f"  - Gallery图片数量: {self.stats['gallery_images']}")
        print(
            f"  - Query/Gallery比例: {self.stats['query_images'] / (self.stats['query_images'] + self.stats['gallery_images']):.1%}/{self.stats['gallery_images'] / (self.stats['query_images'] + self.stats['gallery_images']):.1%}")
        print(f"  - 图片选取方式: 随机选取")

        # 验证每个query中的个体在gallery中都有图片
        valid_check = True
        for identity in query_data:
            if identity not in gallery_data or len(gallery_data[identity]) == 0:
                print(f"错误: Query中的个体 {identity} 在Gallery中没有图片!")
                valid_check = False

        if valid_check:
            print(f"  - 验证通过: 每个Query个体在Gallery中都有对应图片 ✓")

        return gallery_data, query_data

    def create_directory_structure(self):
        """创建输出目录结构"""
        if not self.args.preview_only:
            self.output_root.mkdir(parents=True, exist_ok=True)
            self.train_dir.mkdir(parents=True, exist_ok=True)
            self.query_dir.mkdir(parents=True, exist_ok=True)
            self.gallery_dir.mkdir(parents=True, exist_ok=True)

    def copy_or_link_file(self, src_path, dst_path):
        """复制或创建软链接"""
        if self.args.preview_only:
            return

        # 确保目标目录存在
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if self.args.copy_files:
            shutil.copy2(src_path, dst_path)
        else:
            # 创建软链接（在Windows上可能需要管理员权限）
            try:
                if dst_path.exists():
                    dst_path.unlink()
                dst_path.symlink_to(src_path.absolute())
            except OSError:
                # 如果创建软链接失败，则复制文件
                shutil.copy2(src_path, dst_path)

    def create_train_set(self, identity_images, train_identities):
        """创建训练集"""
        print(f"\n{'预览' if self.args.preview_only else '创建'}训练集...")

        train_images_count = 0

        for identity in train_identities:
            images = identity_images[identity]
            identity_dir = self.train_dir / identity

            if not self.args.preview_only:
                identity_dir.mkdir(parents=True, exist_ok=True)

            for img_path in images:
                dst_path = identity_dir / img_path.name
                self.copy_or_link_file(img_path, dst_path)
                train_images_count += 1

        self.stats['train_images'] = train_images_count
        print(f"训练集创建完成: {train_images_count} 张图片")

    def create_test_set(self, gallery_data, query_data):
        """创建测试集(query和gallery)"""
        print(f"\n{'预览' if self.args.preview_only else '创建'}测试集...")

        # 创建gallery
        print("创建Gallery集...")
        gallery_count = 0
        for identity, images in gallery_data.items():
            identity_dir = self.gallery_dir / identity

            if not self.args.preview_only:
                identity_dir.mkdir(parents=True, exist_ok=True)

            for img_path in images:
                dst_path = identity_dir / img_path.name
                self.copy_or_link_file(img_path, dst_path)
                gallery_count += 1

        # 创建query
        print("创建Query集...")
        query_count = 0
        for identity, images in query_data.items():
            identity_dir = self.query_dir / identity

            if not self.args.preview_only:
                identity_dir.mkdir(parents=True, exist_ok=True)

            for img_path in images:
                dst_path = identity_dir / img_path.name
                self.copy_or_link_file(img_path, dst_path)
                query_count += 1

        print(f"测试集创建完成: Query {query_count} 张, Gallery {gallery_count} 张")

    def generate_annotation_files(self, identity_images, train_identities, gallery_data, query_data):
        """生成标注文件"""
        if not self.args.generate_annotation:
            return

        print(f"\n{'预览' if self.args.preview_only else '生成'}标注文件...")

        # 创建ID映射 (训练集个体从0开始编号)
        train_id_mapping = {identity: idx for idx, identity in enumerate(sorted(train_identities))}

        # 测试集个体编号从训练集最大ID+1开始 (用于开集识别)
        test_identities = list(gallery_data.keys())
        max_train_id = len(train_identities) - 1 if train_identities else -1
        test_id_mapping = {identity: max_train_id + 1 + idx for idx, identity in enumerate(sorted(test_identities))}

        # 合并映射
        all_id_mapping = {**train_id_mapping, **test_id_mapping}

        if not self.args.preview_only:
            # 生成训练集标注文件
            train_annotation_path = self.output_root / "train_annotation.txt"
            with open(train_annotation_path, 'w', encoding='utf-8') as f:
                for identity in sorted(train_identities):
                    class_id = train_id_mapping[identity]
                    for img_path in identity_images[identity]:
                        relative_path = self.train_dir / identity / img_path.name
                        f.write(f"{class_id};{relative_path.absolute()}\n")

            # 生成Gallery标注文件
            gallery_annotation_path = self.output_root / "gallery_annotation.txt"
            with open(gallery_annotation_path, 'w', encoding='utf-8') as f:
                for identity in sorted(gallery_data.keys()):
                    class_id = all_id_mapping[identity]
                    for img_path in gallery_data[identity]:
                        relative_path = self.gallery_dir / identity / img_path.name
                        f.write(f"{class_id};{relative_path.absolute()}\n")

            # 生成Query标注文件
            query_annotation_path = self.output_root / "query_annotation.txt"
            with open(query_annotation_path, 'w', encoding='utf-8') as f:
                for identity in sorted(query_data.keys()):
                    class_id = all_id_mapping[identity]
                    for img_path in query_data[identity]:
                        relative_path = self.query_dir / identity / img_path.name
                        f.write(f"{class_id};{relative_path.absolute()}\n")

            # 生成ID映射文件
            mapping_path = self.output_root / "identity_mapping.json"
            mapping_info = {
                'train_identities': train_id_mapping,
                'test_identities': test_id_mapping,
                'num_train_classes': len(train_identities),
                'num_test_classes': len(test_identities),
                'total_classes': len(all_id_mapping),
                'train_id_range': [0, len(train_identities) - 1] if train_identities else [],
                'test_id_range': [len(train_identities), len(all_id_mapping) - 1] if test_identities else [],
                'notes': {
                    'train_test_split': '训练集和测试集的个体完全不交叉',
                    'query_gallery': '每个Query中的个体在Gallery中都有对应的图片',
                    'open_set': '支持开集识别，测试集个体ID从训练集最大ID+1开始'
                }
            }

            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(mapping_info, f, indent=2, ensure_ascii=False)

            print(f"标注文件生成完成:")
            print(f"  - 训练集标注: {train_annotation_path}")
            print(f"  - Gallery标注: {gallery_annotation_path}")
            print(f"  - Query标注: {query_annotation_path}")
            print(f"  - ID映射文件: {mapping_path}")

        return all_id_mapping

    def generate_dataset_info(self, all_id_mapping):
        """生成数据集信息文件"""
        if self.args.preview_only:
            return

        info = {
            'dataset_info': {
                'name': f"Re-ID Split Dataset from {self.data_root.name}",
                'creation_time': str(datetime.datetime.now()),
                'parameters': vars(self.args),
                'description': 'Re-ID数据集，训练集和测试集个体完全不交叉'
            },
            'statistics': self.stats,
            'split_info': {
                'train_ratio': self.args.train_ratio,
                'query_ratio': self.args.query_ratio,
                'closed_set_identities': len(
                    [k for k in all_id_mapping.keys() if all_id_mapping[k] < self.stats['train_identities']]),
                'open_set_identities': len(
                    [k for k in all_id_mapping.keys() if all_id_mapping[k] >= self.stats['train_identities']]),
                'train_test_overlap': False,
                'query_gallery_overlap': True  # Query中的个体在Gallery中有对应图片
            }
        }

        info_path = self.output_root / "dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print(f"数据集信息文件: {info_path}")

    def print_summary(self):
        """打印划分总结"""
        print("\n" + "=" * 60)
        print("Re-ID数据集划分总结")
        print("=" * 60)
        print(f"原始数据集: {self.data_root}")
        print(f"输出目录: {self.output_root}")
        print(f"随机种子: {self.args.seed}")
        print()

        print("个体统计:")
        print(f"  总个体数量: {self.stats['total_identities']}")
        print(
            f"  训练集个体: {self.stats['train_identities']} ({self.stats['train_identities'] / self.stats['total_identities']:.1%})")
        print(
            f"  测试集个体: {self.stats['test_identities']} ({self.stats['test_identities'] / self.stats['total_identities']:.1%})")
        print(f"  跳过个体: {self.stats['discarded_identities']}")
        print()

        print("图片统计:")
        print(f"  总图片数量: {self.stats['total_images']}")
        print(f"  训练集图片: {self.stats['train_images']} ({self.stats['train_images'] / self.stats['total_images']:.1%})")
        print(
            f"  Query图片: {self.stats['query_images']} ({self.stats['query_images'] / self.stats['total_images']:.1%})")
        print(
            f"  Gallery图片: {self.stats['gallery_images']} ({self.stats['gallery_images'] / self.stats['total_images']:.1%})")
        print(f"  跳过图片: {self.stats['discarded_images']}")
        print()

        print("测试集Query/Gallery划分:")
        test_total = self.stats['query_images'] + self.stats['gallery_images']
        if test_total > 0:
            print(f"  Query比例: {self.stats['query_images'] / test_total:.1%}")
            print(f"  Gallery比例: {self.stats['gallery_images'] / test_total:.1%}")
        print()

        print("Re-ID任务支持:")
        print(f"  训练集和测试集个体完全不交叉: ✓")
        print(f"  每个Query个体在Gallery中都有对应图片: ✓")
        print(f"  ID和图片选取方式: 完全随机 (种子: {self.args.seed})")
        print(f"  支持闭集识别: ✓")
        print(f"  支持开集识别: ✓")
        print(f"  闭集个体ID范围: 0 - {self.stats['train_identities'] - 1}")
        print(
            f"  开集个体ID范围: {self.stats['train_identities']} - {self.stats['train_identities'] + self.stats['test_identities'] - 1}")
        print("=" * 60)

    def run(self):
        """运行数据集划分"""
        print("=" * 60)
        print("Re-ID数据集随机划分工具")
        print("=" * 60)
        print(f"输入数据集: {self.data_root}")
        print(f"输出目录: {self.output_root}")
        print(f"随机种子: {self.args.seed}")
        print(f"预览模式: {self.args.preview_only}")
        print("\n划分策略:")
        print(f"  - ID选取: 随机选取")
        print(f"  - 图片选取: 随机选取")
        print(f"  - 训练/测试比例: {self.args.train_ratio:.0%}/{1 - self.args.train_ratio:.0%}")
        print(f"  - Query/Gallery比例: {self.args.query_ratio:.0%}/{1 - self.args.query_ratio:.0%}")
        print("=" * 60)

        # 扫描数据集
        identity_images = self.scan_dataset()
        if not identity_images:
            print("错误: 没有找到有效的个体数据!")
            return

        # 随机划分个体（训练集和测试集个体完全不交叉）
        train_identities, test_identities = self.split_identities(identity_images)

        # 随机划分测试集图片为Query和Gallery
        gallery_data, query_data = self.split_test_images(identity_images, test_identities)

        # 创建目录结构
        self.create_directory_structure()

        # 创建训练集
        self.create_train_set(identity_images, train_identities)

        # 创建测试集
        self.create_test_set(gallery_data, query_data)

        # 生成标注文件
        all_id_mapping = self.generate_annotation_files(
            identity_images, train_identities, gallery_data, query_data)

        # 生成数据集信息
        if not self.args.preview_only and all_id_mapping:
            self.generate_dataset_info(all_id_mapping)

        # 打印总结
        self.print_summary()

        if self.args.preview_only:
            print("\n这是预览模式，未实际创建文件。")
            print("如需实际执行，请移除 --preview-only 参数。")


def main():
    args = parse_args()

    # 验证输入参数
    if not os.path.exists(args.data_root):
        print(f"错误: 数据根目录不存在: {args.data_root}")
        return

    if args.train_ratio <= 0 or args.train_ratio >= 1:
        print("错误: train_ratio必须在0和1之间")
        return

    if args.query_ratio <= 0 or args.query_ratio >= 1:
        print("错误: query_ratio必须在0和1之间")
        return

    # 运行数据集划分
    splitter = ReIDDatasetSplitter(args)
    splitter.run()


if __name__ == "__main__":
    main()