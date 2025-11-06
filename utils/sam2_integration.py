# utils/sam2_integration.py
import os
import torch
import numpy as np
import cv2
from PIL import Image


def initialize_sam2_model(checkpoint_path, model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml", device="cuda"):
    """初始化 SAM2 图像预测器

    Args:
        checkpoint_path: SAM2 模型权重路径
        model_cfg: SAM2 模型配置文件路径
        device: 设备类型 ('cuda' 或 'cpu')

    Returns:
        SAM2ImagePredictor 实例或 None（如果初始化失败）
    """
    try:
        # 导入 SAM2 模块
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # 检查文件存在
        if not os.path.exists(checkpoint_path):
            print(f"错误: SAM2 模型权重文件不存在: {checkpoint_path}")
            return None

        if not os.path.exists(model_cfg):
            print(f"错误: SAM2 模型配置文件不存在: {model_cfg}")
            return None

        # 构建 SAM2 模型
        model = build_sam2(model_cfg, checkpoint_path)
        model.to(device)

        # 创建预测器
        predictor = SAM2ImagePredictor(model)
        print(f"SAM2 模型初始化成功: {os.path.basename(checkpoint_path)}")
        return predictor

    except Exception as e:
        import traceback
        print(f"SAM2 模型初始化失败: {e}")
        traceback.print_exc()
        return None