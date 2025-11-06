# utils/utils.py
import random
import numpy as np
import torch
from PIL import Image, ImageFile # Import ImageFile
import os

# 允许加载可能损坏的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 将图像转换为RGB
def cvtColor(image):
    if image.mode == 'RGB':
        return image
    else:
        image = image.convert('RGB')
        return image

# 使用Letterbox调整图像大小
def resize_image(image, size, letterbox_image):
    """
    调整图像大小，可选使用letterbox保持宽高比。
    size: tuple (目标宽度, 目标高度)
    """
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        # 使用 ANTIALIAS 以获得更好的缩放质量
        image = image.resize((nw, nh), Image.Resampling.LANCZOS) # 或者 BICUBIC
        # 创建灰色背景的新图像
        new_image = Image.new('RGB', size, (128, 128, 128))
        # 将缩放后的图像粘贴到中心
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.Resampling.LANCZOS) # 或者 BICUBIC
    return new_image

# 从标注文件获取类别（ID）数量

# 在 utils/utils.py 中修改 get_num_classes 函数

def get_num_classes(annotation_path):
    """返回标注文件中唯一ID的数量，而不是最大ID+1"""
    unique_ids = set()

    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"未找到标注文件: {annotation_path}")

    with open(annotation_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(';')
        if len(parts) >= 2 and parts[0].isdigit():
            unique_ids.add(int(parts[0]))
        else:
            print(f"警告: 在标注文件中跳过无效行: {line.strip()}")

    if not unique_ids:
        raise ValueError("在标注文件中未找到有效的数字ID。")

    # 返回不同ID的数量，而不是最大ID+1
    num_classes = len(unique_ids)
    print(f"检测到{num_classes}个唯一ID/类别。")
    return num_classes
# 获取优化器的当前学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# 设置全局随机种子以保证可复现性
def seed_everything(seed=11):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 设置Python哈希种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU.
    # 设置CUDNN的确定性，可能会牺牲一些性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Benchmark为False确保完全确定性
    print(f"Global seed set to {seed}")

# Dataloader worker的初始化函数，确保多线程加载时种子不同
def worker_init_fn(worker_id, rank=0, seed=11):
    # 使用 rank, seed, 和 worker_id 来为每个worker创建唯一的种子
    worker_seed = rank + seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    print(f"Worker {worker_id} initialized with seed {worker_seed}")


# 显示配置信息
def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)