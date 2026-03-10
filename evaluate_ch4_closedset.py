import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.ch4_protocol import (
    build_model,
    compute_macro_f1,
    compute_topk_accuracies,
    count_parameters,
    create_closedset_datasets,
    load_checkpoint,
    measure_inference_time,
    save_metric_bundle,
)
from utils.dataloader_sgpd import sgpd_dataset_collate
from utils.utils import seed_everything, worker_init_fn


def parse_args():
    parser = argparse.ArgumentParser(description="第四章闭集分类评估脚本")
    parser.add_argument("--model-type", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--annotation-path", type=str, default="paperdata-train.txt")
    parser.add_argument("--dataset-root", type=str, default="", help="本地图像根目录，用于重写标注路径")
    parser.add_argument("--output-dir", type=str, default="chapter4_reports/closedset")
    parser.add_argument("--method-name", type=str, default="", help="表格中的方法名")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_dataset, metadata = create_closedset_datasets(
        annotation_path=args.annotation_path,
        input_shape=[128, 128],
        seed=args.seed,
        val_split=args.val_split,
        dataset_root=args.dataset_root or None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=sgpd_dataset_collate,
        worker_init_fn=partial(worker_init_fn, seed=args.seed),
    )

    model = build_model(
        model_type=args.model_type,
        num_classes=metadata["num_classes"],
        device=device,
        pretrained=args.pretrained,
    )
    checkpoint_meta = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    top1_correct = 0.0
    top5_correct = 0.0
    predictions = []
    labels_all = []

    with torch.no_grad():
        for images1, _, subject_masks, labels in tqdm(val_loader, desc="Closed-set Eval", mininterval=0.3):
            images1 = images1.to(device)
            subject_masks = subject_masks.to(device)
            labels = labels.to(device)

            logits = model(images1, subject_mask=subject_masks, label=labels, mode="eval_logits")
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            predictions.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
            topk = compute_topk_accuracies(logits, labels, topk=(1, 5))
            top1_correct += topk[1] * batch_size
            top5_correct += topk[5] * batch_size

    macro_f1 = compute_macro_f1(predictions, labels_all, num_classes=metadata["num_classes"])
    total_params, trainable_params = count_parameters(model)
    timing = measure_inference_time(model, val_loader, device=device)

    method_name = args.method_name or args.model_type
    metrics = {
        "method_name": method_name,
        "model_type": args.model_type,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "annotation_path": str(Path(args.annotation_path).resolve()),
        "split_seed": args.seed,
        "val_split": args.val_split,
        "num_classes": metadata["num_classes"],
        "val_samples": metadata["val_samples"],
        "loss": total_loss / total_samples if total_samples else 0.0,
        "top1": top1_correct / total_samples if total_samples else 0.0,
        "top5": top5_correct / total_samples if total_samples else 0.0,
        "macro_f1": macro_f1,
        "params_total": total_params,
        "params_trainable": trainable_params,
        "time_per_image_ms": timing["time_per_image_ms"],
        "fps": timing["fps"],
        "checkpoint_metrics": checkpoint_meta.get("metrics", {}) if isinstance(checkpoint_meta, dict) else {},
        "rewrite_info": metadata["rewrite_info"],
    }

    output_paths = save_metric_bundle(
        output_dir=args.output_dir,
        stem=method_name.replace(" ", "_").lower(),
        metrics=metrics,
        latex_columns=["top1", "top5", "macro_f1", "params_total", "time_per_image_ms"],
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Results saved to: {output_paths}")


if __name__ == "__main__":
    main()
