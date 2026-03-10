import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from models.ablation_models import SGPDNet_NoStructureGuidance, SimpleLightCNNBaseline
from models.ch4_baselines import (
    DenseNet121SubCenterArcFace,
    LightCNNSubCenterArcFace,
    ResNet50SubCenterArcFace,
)
from models.sgpd_net import SGPDNet
from utils.dataloader_sgpd import SGPDDataset, sgpd_dataset_collate
from utils.utils import cvtColor, resize_image


TEXT_ENCODINGS = ("utf-8", "gbk", "gb18030")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_text_lines(path, encodings=TEXT_ENCODINGS):
    path = Path(path)
    last_error = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding).splitlines()
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return path.read_text().splitlines()


def parse_annotation_lines(annotation_path):
    records = []
    for line in read_text_lines(annotation_path):
        if not line.strip():
            continue
        parts = line.strip().split(";", 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        records.append({"label": int(parts[0]), "path": parts[1]})
    return records


def extract_identity_name(image_path):
    return Path(image_path.replace("\\", "/")).parent.name


def collect_training_identity_names(annotation_path):
    return sorted({extract_identity_name(record["path"]) for record in parse_annotation_lines(annotation_path)})


def guess_dataset_root():
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "Paper-Data-Copy",
        repo_root / "paper-data-copy",
        repo_root / "splited_dataset",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_image_path(raw_path, dataset_root=None):
    normalized = raw_path.replace("\\", "/")
    candidates = [Path(normalized)]

    if dataset_root:
        dataset_root = Path(dataset_root)
        if "Paper-Data-Copy/" in normalized:
            suffix = normalized.split("Paper-Data-Copy/", 1)[1]
            candidates.append(dataset_root / suffix)
        if "splited_dataset/" in normalized:
            suffix = normalized.split("splited_dataset/", 1)[1]
            if dataset_root.name.lower() == "splited_dataset":
                candidates.append(dataset_root / suffix)
            else:
                candidates.append(dataset_root / "splited_dataset" / suffix)
        candidates.append(dataset_root / Path(normalized).name)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return normalized


def rewrite_annotation_lines(annotation_path, dataset_root=None):
    if dataset_root is None:
        dataset_root = guess_dataset_root()
    records = parse_annotation_lines(annotation_path)
    rewritten = []
    rewrites = 0
    for record in records:
        resolved = resolve_image_path(record["path"], dataset_root=dataset_root)
        original = record["path"].replace("\\", "/")
        if resolved.replace("\\", "/") != original:
            rewrites += 1
        rewritten.append(f"{record['label']};{resolved}\n")
    return rewritten, {"dataset_root": str(dataset_root) if dataset_root else None, "rewritten_paths": rewrites}


def build_closedset_annotation_splits(annotation_path, seed=3407, val_split=0.15, dataset_root=None):
    lines, rewrite_info = rewrite_annotation_lines(annotation_path, dataset_root=dataset_root)
    rng = np.random.RandomState(seed)
    shuffled = list(lines)
    rng.shuffle(shuffled)
    num_val = int(len(shuffled) * val_split)
    num_train = len(shuffled) - num_val
    return shuffled[:num_train], shuffled[num_train:], rewrite_info


def create_closedset_datasets(annotation_path, input_shape, seed=3407, val_split=0.15, dataset_root=None):
    train_lines, val_lines, rewrite_info = build_closedset_annotation_splits(
        annotation_path=annotation_path,
        seed=seed,
        val_split=val_split,
        dataset_root=dataset_root,
    )
    train_dataset = SGPDDataset(input_shape, train_lines, train_mode=True, sam_predictor=None)
    val_dataset = SGPDDataset(input_shape, val_lines, train_mode=False, sam_predictor=None)
    metadata = {
        "seed": seed,
        "val_split": val_split,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "num_classes": train_dataset.num_classes,
        "rewrite_info": rewrite_info,
    }
    return train_dataset, val_dataset, metadata


def compute_topk_accuracies(logits, labels, topk=(1, 5)):
    max_k = min(max(topk), logits.size(1))
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    results = {}
    batch_size = labels.size(0)
    for k in topk:
        k = min(k, logits.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results[k] = float(correct_k.item() / batch_size) if batch_size else 0.0
    return results


def compute_macro_f1(predictions, labels, num_classes=None):
    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return 0.0
    if num_classes is None:
        pred_max = int(predictions.max()) if predictions.size else 0
        label_max = int(labels.max()) if labels.size else 0
        num_classes = max(pred_max, label_max) + 1
    f1_values = []
    for class_id in range(num_classes):
        tp = np.sum((predictions == class_id) & (labels == class_id))
        fp = np.sum((predictions == class_id) & (labels != class_id))
        fn = np.sum((predictions != class_id) & (labels == class_id))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1_values)) if f1_values else 0.0


def count_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_json(path, content):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, ensure_ascii=False)


def write_single_row_csv(path, content):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(content.keys()))
        writer.writeheader()
        writer.writerow(content)


def format_metric_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_latex_row(method_name, metrics, columns):
    values = [method_name]
    for column in columns:
        values.append(format_metric_value(metrics.get(column, "")))
    return " & ".join(values) + r" \\"


def save_metric_bundle(output_dir, stem, metrics, latex_columns=None):
    ensure_dir(output_dir)
    json_path = Path(output_dir) / f"{stem}.json"
    csv_path = Path(output_dir) / f"{stem}.csv"
    write_json(json_path, metrics)
    write_single_row_csv(csv_path, metrics)
    latex_path = None
    if latex_columns:
        latex_path = Path(output_dir) / f"{stem}_latex.txt"
        latex_path.write_text(
            format_latex_row(metrics.get("method_name", stem), metrics, latex_columns),
            encoding="utf-8",
        )
    return {"json": str(json_path), "csv": str(csv_path), "latex": str(latex_path) if latex_path else None}


class IdentityImageFolderDataset(Dataset):
    def __init__(self, root_dir, input_shape=(128, 128)):
        self.root_dir = Path(root_dir)
        self.input_shape = tuple(input_shape)
        self.samples = []
        for identity_dir in sorted(self.root_dir.iterdir()):
            if not identity_dir.is_dir():
                continue
            for image_path in sorted(identity_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS and image_path.is_file():
                    self.samples.append((str(image_path), identity_dir.name))
        if not self.samples:
            raise ValueError(f"No images found under {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, identity = self.samples[index]
        image = Image.open(image_path)
        image = cvtColor(image)
        image = resize_image(image, (self.input_shape[1], self.input_shape[0]), True)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))
        image_tensor = (image_tensor - 0.5) / 0.5
        mask_tensor = torch.ones((1, self.input_shape[0], self.input_shape[1]), dtype=torch.float32)
        return image_tensor.float(), mask_tensor, identity, image_path


def identity_folder_collate(batch):
    images, masks, identities, paths = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(masks, dim=0), list(identities), list(paths)


def compute_distance_matrix(query_features, gallery_features, metric="cosine"):
    if metric == "cosine":
        query_features = F.normalize(query_features, p=2, dim=1)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)
        similarity = torch.mm(query_features, gallery_features.t())
        return 1.0 - similarity
    if metric == "euclidean":
        return torch.cdist(query_features, gallery_features, p=2)
    raise ValueError(f"Unsupported metric: {metric}")


def evaluate_retrieval(distance_matrix, query_ids, gallery_ids, max_rank=5):
    distance_matrix = distance_matrix.detach().cpu().numpy()
    gallery_ids = np.asarray(gallery_ids)
    max_rank = min(max_rank, distance_matrix.shape[1])
    cmc = np.zeros(max_rank, dtype=np.float64)
    average_precisions = []
    valid_queries = 0

    for row_index, query_id in enumerate(query_ids):
        order = np.argsort(distance_matrix[row_index])
        matches = (gallery_ids[order] == query_id).astype(np.int32)
        if matches.sum() == 0:
            continue
        valid_queries += 1
        first_hit = np.where(matches == 1)[0][0]
        cmc[first_hit:] += 1

        cumulative_hits = np.cumsum(matches)
        precision = cumulative_hits / (np.arange(len(matches)) + 1.0)
        average_precisions.append(float((precision * matches).sum() / matches.sum()))

    if valid_queries == 0:
        return {"rank1": 0.0, "rank5": 0.0, "mAP": 0.0, "valid_queries": 0}

    cmc = cmc / valid_queries
    return {
        "rank1": float(cmc[0]) if len(cmc) >= 1 else 0.0,
        "rank5": float(cmc[min(4, len(cmc) - 1)]) if len(cmc) >= 1 else 0.0,
        "mAP": float(np.mean(average_precisions)) if average_precisions else 0.0,
        "valid_queries": int(valid_queries),
    }


def build_model(model_type, num_classes, device, pretrained=False, window_size=7):
    model_type = model_type.lower()
    common_kwargs = {"num_classes": num_classes}
    if model_type == "sgpd_net":
        model = SGPDNet(
            num_classes=num_classes,
            latent_channels=4,
            structure_dim=256,
            lightcnn_embedding_dim=512,
            pretrained=False,
            sub_centers_k=2,
            arcface_s=12,
            arcface_m=0.15,
            latent_inversion_iters=30,
            latent_inversion_lr=0.01,
            window_size=window_size,
        )
    elif model_type == "sgpd_no_structure":
        model = SGPDNet_NoStructureGuidance(**common_kwargs)
    elif model_type == "lightcnn_ce":
        model = SimpleLightCNNBaseline(**common_kwargs)
    elif model_type == "lightcnn_arcface":
        model = LightCNNSubCenterArcFace(
            **common_kwargs,
            sub_centers_k=2,
            arcface_s=16.0,
            arcface_m=0.2,
        )
    elif model_type == "resnet50_arcface":
        model = ResNet50SubCenterArcFace(
            **common_kwargs,
            sub_centers_k=2,
            arcface_s=16.0,
            arcface_m=0.2,
            pretrained=pretrained,
        )
    elif model_type == "densenet121_arcface":
        model = DenseNet121SubCenterArcFace(
            **common_kwargs,
            sub_centers_k=2,
            arcface_s=16.0,
            arcface_m=0.2,
            pretrained=pretrained,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model.to(device)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint
    metadata = {}
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    model.load_state_dict(state_dict, strict=False)
    return metadata


def predict_logits(model, images, masks=None, labels=None):
    if masks is None:
        return model(images, label=labels, mode="eval_logits")
    return model(images, subject_mask=masks, label=labels, mode="eval_logits")


def extract_embeddings(model, images, masks=None):
    if masks is None:
        return model(images, mode="predict")
    return model(images, subject_mask=masks, mode="predict")


def measure_inference_time(model, dataloader, device, max_batches=30):
    model.eval()
    warmup_batches = 3
    timings = []
    seen = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            images = batch[0].to(device)
            masks = batch[2].to(device) if len(batch) >= 3 and torch.is_tensor(batch[2]) else None
            labels = batch[3].to(device) if len(batch) >= 4 and torch.is_tensor(batch[3]) else None

            synchronize_if_needed(device)
            start = time.perf_counter()
            _ = predict_logits(model, images, masks=masks, labels=labels)
            synchronize_if_needed(device)
            elapsed = time.perf_counter() - start

            if batch_index >= warmup_batches:
                timings.append(elapsed / max(images.size(0), 1))
            seen += 1
            if seen >= max_batches:
                break

    if not timings:
        return {"time_per_image_ms": 0.0, "fps": 0.0}
    avg_seconds = sum(timings) / len(timings)
    return {
        "time_per_image_ms": avg_seconds * 1000.0,
        "fps": 1.0 / avg_seconds if avg_seconds > 0 else 0.0,
    }


def create_identity_manifest(root_dir):
    root_dir = Path(root_dir)
    manifest = {}
    for identity_dir in sorted(root_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        manifest[identity_dir.name] = sorted(
            [str(path.resolve()) for path in identity_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        )
    return manifest
