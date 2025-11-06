# utils/dataloader_sgpd.py - 修复版本
import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import random
from utils.utils import cvtColor, resize_image, worker_init_fn
import os

# Albumentations imports
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 允许加载可能损坏的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_sam_segmentation_mask(image_pil, sam_predictor):
    """使用SAM分割主体，返回二值掩码"""
    try:
        image_np = np.array(image_pil)
        h, w = image_np.shape[:2]

        if image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_available() and
                                                                                    torch.cuda.get_device_capability()[
                                                                                        0] >= 8 else torch.float32):
            sam_predictor.set_image(image_np)

            center_x, center_y = w // 2, h // 2
            input_point = np.array([[center_x, center_y],
                                    [center_x, center_y - h // 4],
                                    [center_x, center_y + h // 4]])
            input_label = np.array([1, 1, 1])

            margin_x = w // 6
            margin_y = h // 12
            box = np.array([margin_x, margin_y, w - margin_x, h - margin_y])

            masks, _, _ = sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=box.reshape(1, 4) if hasattr(sam_predictor, 'predict_torch') else box,
                multimask_output=True
            )

        if masks is not None and len(masks) > 0:
            subject_mask = masks[0]
            if isinstance(subject_mask, torch.Tensor):
                subject_mask = subject_mask.cpu().numpy()

            if subject_mask.ndim == 3 and subject_mask.shape[0] == 1:
                subject_mask = subject_mask.squeeze(0)
            elif subject_mask.ndim != 2:
                raise ValueError(f"Unexpected mask dimension: {subject_mask.shape}")

            kernel = np.ones((5, 5), np.uint8)
            closed_mask = cv2.morphologyEx(subject_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            refined_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

            mask_area_ratio = np.sum(refined_mask) / (h * w)
            if mask_area_ratio < 0.05:
                backup_mask = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = [int(c) for c in box]
                backup_mask[y1:y2, x1:x2] = 1
                final_mask_np = backup_mask
            else:
                final_mask_np = refined_mask

            mask_pil = Image.fromarray((final_mask_np * 255).astype(np.uint8))
            return mask_pil
        else:
            mask_pil = Image.fromarray(np.ones((h, w), dtype=np.uint8) * 255)
            return mask_pil

    except Exception as e:
        import traceback
        print(f"SAM分割出错: {e}")
        traceback.print_exc()
        h_orig, w_orig = np.array(image_pil).shape[:2]
        mask_pil = Image.fromarray(np.ones((h_orig, w_orig), dtype=np.uint8) * 255)
        return mask_pil


class SGPDDataset(Dataset):
    def __init__(self, input_shape_hw, annotation_lines, train_mode=True, sam_predictor=None):
        super(SGPDDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape_hw = tuple(input_shape_hw)
        self.train_mode = train_mode
        self.sam_predictor = sam_predictor

        # --- 修复的数据增强策略 ---
        # 1. 基础几何和颜色变换（对所有图像应用）
        self.base_transform_for_image1_and_mask = A.Compose([
            A.HorizontalFlip(p=0.5 if train_mode else 0),
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.02,
                rotate_limit=3,
                p=0.4 if train_mode else 0,
                border_mode=cv2.BORDER_REFLECT_101
            ),
            A.ColorJitter(
                brightness=0.03,
                contrast=0.03,
                saturation=0.03,
                hue=0.01,
                p=0.3 if train_mode else 0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.03,
                contrast_limit=0.03,
                p=0.2 if train_mode else 0
            ),
        ])

        # 2. 增强版几何变换（专门用于image2）
        self.enhanced_geometric_transform_for_img2 = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.08,
                rotate_limit=10,
                p=0.7 if train_mode else 0,
                border_mode=cv2.BORDER_REFLECT_101
            ),
            A.HorizontalFlip(p=0.5 if train_mode else 0),
            A.ElasticTransform(
                alpha=20,
                sigma=3,
                p=0.1 if train_mode else 0
            ),
            A.GridDistortion(
                num_steps=3,
                distort_limit=0.1,
                p=0.1 if train_mode else 0
            ),
        ])

        # 3. 多样化的主体内容变换
        self.diverse_subject_content_transform = A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.4),
                A.MotionBlur(blur_limit=(3, 5), p=0.4),
                A.MedianBlur(blur_limit=3, p=0.2),
            ], p=0.5 if train_mode else 0),

            A.OneOf([
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.1,
                    hue=0.03,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=0.3
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.2),
            ], p=0.6 if train_mode else 0),

            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3 if train_mode else 0
            ),

            # 修复：使用正确的GaussNoise参数
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 30.0), mean=0, p=0.3),  # 添加mean参数
                A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.1, 0.3), p=0.2),
            ], p=0.2 if train_mode else 0),
        ])

        # 4. 背景变换
        self.diverse_background_content_transform = A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(5, 9), p=0.6),
                A.MotionBlur(blur_limit=(5, 9), p=0.4),
            ], p=0.8 if train_mode else 0),

            A.ColorJitter(
                brightness=0.08,
                contrast=0.08,
                saturation=0.05,
                hue=0.02,
                p=0.4 if train_mode else 0
            ),

            # 修复：使用正确的GaussNoise参数
            A.GaussNoise(var_limit=(5.0, 15.0), mean=0, p=0.3 if train_mode else 0),
        ])

        # 5. 主体遮挡概率
        self.subject_occlusion_probability = 0.05 if train_mode else 0

        # 6. 最终标准化
        self.final_tensor_transform = A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])

        # --- 解析类别ID映射 ---
        self.image_paths = []
        self.labels = []
        unique_class_ids = set()
        for line in annotation_lines:
            parts = line.strip().split(';')
            if len(parts) >= 2 and parts[0].isdigit():
                unique_class_ids.add(int(parts[0]))

        self.class_ids = sorted(list(unique_class_ids))
        self.class_to_idx = {class_id: idx for idx, class_id in enumerate(self.class_ids)}

        valid_lines = 0
        for i, line in enumerate(annotation_lines):
            parts = line.strip().split(';')
            if len(parts) >= 2:
                image_path = parts[1].replace('\\', '/')
                label_str = parts[0]
                if os.path.exists(image_path) and label_str.isdigit():
                    self.image_paths.append(image_path)
                    label_id = int(label_str)
                    if label_id in self.class_to_idx:
                        self.labels.append(self.class_to_idx[label_id])
                        valid_lines += 1

        self.num_classes = len(self.class_to_idx)
        if valid_lines == 0:
            raise ValueError("No valid data found in annotation file!")
        print(f"Dataset: {valid_lines} valid samples, {self.num_classes} unique classes.")

    def _apply_subject_occlusion(self, image, mask, p=0.05, fill_value=128):
        """改进的主体遮挡"""
        if random.random() > p:
            return image.copy()

        occluded_image = image.copy()
        subject_pixels_bool = (mask / 255.0) > 0.5

        y_indices, x_indices = np.where(subject_pixels_bool)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return occluded_image

        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)

        subject_height = y_max - y_min
        subject_width = x_max - x_min

        num_holes = random.randint(1, 1)

        for _ in range(num_holes):
            hole_h = random.randint(
                max(3, int(subject_height * 0.015)),
                min(6, int(subject_height * 0.06))
            )
            hole_w = random.randint(
                max(3, int(subject_width * 0.015)),
                min(6, int(subject_width * 0.06))
            )

            center_y = random.randint(y_min, y_max)
            center_x = random.randint(x_min, x_max)

            y1 = max(0, center_y - hole_h // 2)
            y2 = min(image.shape[0], center_y + hole_h // 2)
            x1 = max(0, center_x - hole_w // 2)
            x2 = min(image.shape[1], center_x + hole_w // 2)

            occlusion_mask = np.zeros_like(subject_pixels_bool)
            occlusion_mask[y1:y2, x1:x2] = True

            overlap = np.logical_and(occlusion_mask, subject_pixels_bool)
            overlap_ratio = np.sum(overlap) / np.sum(occlusion_mask) if np.sum(occlusion_mask) > 0 else 0

            if overlap_ratio >= 0.3:
                if random.random() < 0.7:
                    fill_val = fill_value
                else:
                    fill_val = random.randint(0, 255)

                occluded_image[y1:y2, x1:x2] = fill_val

        return occluded_image

    def _apply_random_patch_swap(self, image, mask, p=0.1):
        """随机补丁交换"""
        if random.random() > p:
            return image.copy()

        swapped_image = image.copy()
        subject_pixels_bool = (mask / 255.0) > 0.5

        y_indices, x_indices = np.where(subject_pixels_bool)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return swapped_image

        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)

        patch_size = min(8, (y_max - y_min) // 4, (x_max - x_min) // 4)
        if patch_size < 4:
            return swapped_image

        for _ in range(2):
            try:
                y1 = random.randint(y_min, y_max - patch_size)
                x1 = random.randint(x_min, x_max - patch_size)
                y2 = random.randint(y_min, y_max - patch_size)
                x2 = random.randint(x_min, x_max - patch_size)

                patch1 = swapped_image[y1:y1 + patch_size, x1:x1 + patch_size].copy()
                patch2 = swapped_image[y2:y2 + patch_size, x2:x2 + patch_size].copy()

                swapped_image[y1:y1 + patch_size, x1:x1 + patch_size] = patch2
                swapped_image[y2:y2 + patch_size, x2:x2 + patch_size] = patch1
            except:
                continue

        return swapped_image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        try:
            # 1. 图像加载与初始尺寸调整
            image_pil = Image.open(image_path)
            image_pil = cvtColor(image_pil)
            image_resized_pil = resize_image(image_pil, (self.input_shape_hw[1], self.input_shape_hw[0]), True)

            # 转换为NumPy数组
            image_np_orig = np.array(image_resized_pil)

            # 2. 获取SAM主体掩码
            if self.sam_predictor:
                try:
                    subject_mask_orig_pil = get_sam_segmentation_mask(image_resized_pil, self.sam_predictor)
                except Exception as e:
                    print(f"SAM分割失败，使用默认掩码: {e}")
                    h, w = image_np_orig.shape[:2]
                    subject_mask_orig_pil = Image.fromarray(np.ones((h, w), dtype=np.uint8) * 255)
            else:
                h, w = image_np_orig.shape[:2]
                subject_mask_orig_pil = Image.fromarray(np.ones((h, w), dtype=np.uint8) * 255)

            mask_np_orig = np.array(subject_mask_orig_pil)
            if mask_np_orig.ndim == 3:
                mask_np_orig = mask_np_orig[:, :, 0]

            # 3. 为image1和掩码应用基础变换
            try:
                transformed = self.base_transform_for_image1_and_mask(
                    image=image_np_orig.copy(),
                    mask=mask_np_orig.copy()
                )
                image1_np_after_base = transformed['image']
                mask1_np_aligned_clean = transformed['mask']
            except Exception as e:
                print(f"基础变换失败，使用原始图像: {e}")
                image1_np_after_base = image_np_orig.copy()
                mask1_np_aligned_clean = mask_np_orig.copy()

            # 4. 应用轻微的主体遮挡
            try:
                image1_np_after_occlusion = self._apply_subject_occlusion(
                    image=image1_np_after_base,
                    mask=mask1_np_aligned_clean,
                    p=self.subject_occlusion_probability,
                    fill_value=128
                )
            except Exception as e:
                print(f"主体遮挡失败，跳过: {e}")
                image1_np_after_occlusion = image1_np_after_base.copy()

            # 5. 应用补丁交换
            try:
                image1_np_final = self._apply_random_patch_swap(
                    image=image1_np_after_occlusion,
                    mask=mask1_np_aligned_clean,
                    p=0.05 if self.train_mode else 0
                )
            except Exception as e:
                print(f"补丁交换失败，跳过: {e}")
                image1_np_final = image1_np_after_occlusion.copy()

            # 6. 生成image2
            if self.train_mode:
                try:
                    # 几何变换
                    transformed_for_img2 = self.enhanced_geometric_transform_for_img2(
                        image=image1_np_after_base.copy(),
                        mask=mask1_np_aligned_clean.copy()
                    )
                    image_base_for_img2 = transformed_for_img2['image']
                    mask_for_img2 = transformed_for_img2['mask']

                    # 布尔掩码
                    bool_mask_for_img2 = (mask_for_img2 / 255.0) > 0.5
                    bool_mask_for_img2_3ch = np.stack([bool_mask_for_img2] * 3, axis=-1)

                    # 主体和背景内容变换
                    try:
                        subj_content_augmented = self.diverse_subject_content_transform(
                            image=image_base_for_img2.copy()
                        )['image']
                    except Exception as e:
                        print(f"主体增强失败，使用原图: {e}")
                        subj_content_augmented = image_base_for_img2.copy()

                    try:
                        bg_content_augmented = self.diverse_background_content_transform(
                            image=image_base_for_img2.copy()
                        )['image']
                    except Exception as e:
                        print(f"背景增强失败，使用原图: {e}")
                        bg_content_augmented = image_base_for_img2.copy()

                    # 混合主体和背景
                    image2_np = np.where(
                        bool_mask_for_img2_3ch,
                        subj_content_augmented,
                        bg_content_augmented
                    )

                    # 补丁交换
                    try:
                        image2_np = self._apply_random_patch_swap(
                            image=image2_np,
                            mask=mask_for_img2,
                            p=0.08 if self.train_mode else 0
                        )
                    except Exception as e:
                        print(f"image2补丁交换失败，跳过: {e}")

                except Exception as e:
                    print(f"image2生成失败，使用image1: {e}")
                    image2_np = image1_np_after_base.copy()
            else:
                # 验证模式
                image2_np = image1_np_after_base.copy()

            # 7. 转换为Tensor
            try:
                image1_tensor = self.final_tensor_transform(image=image1_np_final)['image']
                image2_tensor = self.final_tensor_transform(image=image2_np)['image']
            except Exception as e:
                print(f"张量转换失败: {e}")
                return None

            # 8. 准备掩码tensor
            subject_mask_tensor = torch.from_numpy(
                mask1_np_aligned_clean.astype(np.float32) / 255.0
            ).unsqueeze(0)

            return image1_tensor, image2_tensor, subject_mask_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error processing item at index {index} (path: {image_path}): {e}")
            import traceback
            traceback.print_exc()
            return None


def sgpd_dataset_collate(batch):
    """修复的数据收集函数"""
    batch = [item for item in batch if item is not None]
    if not batch:
        print("Warning: Collate function received an empty batch after filtering Nones.")
        return None, None, None, None

    try:
        images1 = torch.stack([item[0] for item in batch], 0)
        images2 = torch.stack([item[1] for item in batch], 0)
        subject_masks = torch.stack([item[2] for item in batch], 0)
        labels = torch.tensor([item[3] for item in batch], dtype=torch.long)
        return images1, images2, subject_masks, labels
    except Exception as e:
        print(f"Error during collate_fn: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None