import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from utils.ch4_protocol import (
    build_model,
    compute_checkpoint_score,
    compute_macro_f1,
    compute_topk_accuracies,
    create_closedset_datasets,
    ensure_dir,
)
from utils.dataloader_sgpd import sgpd_dataset_collate
from utils.utils import seed_everything, worker_init_fn


def parse_args():
    parser = argparse.ArgumentParser(description="第四章公平基线训练脚本")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["lightcnn_arcface", "resnet50_arcface", "densenet121_arcface"],
        help="基线模型类型",
    )
    parser.add_argument("--annotation-path", type=str, default="paperdata-train.txt")
    parser.add_argument("--dataset-root", type=str, default="", help="本地图像根目录，用于重写标注路径")
    parser.add_argument("--output-dir", type=str, default="chapter4_runs")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument("--pretrained", action="store_true", help="为 ResNet/DenseNet 使用 torchvision 预训练权重")
    parser.add_argument("--class-balanced-sampler", action="store_true", help="按类别频次倒数采样")
    return parser.parse_args()


def create_weighted_sampler(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    pbar = tqdm(dataloader, desc="Train", mininterval=0.3)

    for images1, _, subject_masks, labels in pbar:
        images1 = images1.to(device)
        subject_masks = subject_masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images1, subject_mask=subject_masks, label=labels, mode="train")
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        pbar.set_postfix(loss=total_loss / total_samples, acc=correct / total_samples)

    return {
        "train_loss": total_loss / total_samples if total_samples else 0.0,
        "train_acc": correct / total_samples if total_samples else 0.0,
    }


def evaluate_one_epoch(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_top1 = 0.0
    correct_top5 = 0.0
    predictions = []
    labels_all = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Val", mininterval=0.3)
        for images1, _, subject_masks, labels in pbar:
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
            correct_top1 += topk[1] * batch_size
            correct_top5 += topk[5] * batch_size
            pbar.set_postfix(
                val_loss=total_loss / total_samples,
                top1=correct_top1 / total_samples,
                top5=correct_top5 / total_samples,
            )

    macro_f1 = compute_macro_f1(predictions, labels_all, num_classes=num_classes)
    top1 = correct_top1 / total_samples if total_samples else 0.0
    top5 = correct_top5 / total_samples if total_samples else 0.0
    return {
        "val_loss": total_loss / total_samples if total_samples else 0.0,
        "val_acc": top1,
        "val_top1": top1,
        "val_top5": top5,
        "val_macro_f1": macro_f1,
        "val_score": compute_checkpoint_score({"val_top1": top1, "val_macro_f1": macro_f1}),
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
        },
        save_path,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, metadata = create_closedset_datasets(
        annotation_path=args.annotation_path,
        input_shape=[128, 128],
        seed=args.seed,
        val_split=args.val_split,
        dataset_root=args.dataset_root or None,
    )

    train_sampler = create_weighted_sampler(train_dataset.labels) if args.class_balanced_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=sgpd_dataset_collate,
        worker_init_fn=partial(worker_init_fn, seed=args.seed),
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
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    run_dir = Path(args.output_dir) / args.model_type
    ensure_dir(run_dir)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "metadata": metadata,
                "device": str(device),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    history = []
    best_metrics = {"val_score": 0.0}
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_one_epoch(model, val_loader, criterion, device, metadata["num_classes"])
        scheduler.step()

        metrics = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **val_metrics,
        }
        history.append(metrics)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

        latest_path = run_dir / "checkpoint_latest.pth"
        save_checkpoint(model, optimizer, scheduler, epoch, metrics, latest_path)

        if (epoch + 1) % args.save_period == 0 or epoch + 1 == args.epochs:
            torch.save(model.state_dict(), run_dir / f"epoch_{epoch + 1:03d}.pth")

        if metrics["val_score"] > best_metrics.get("val_score", 0.0):
            best_metrics = metrics
            save_checkpoint(model, optimizer, scheduler, epoch, best_metrics, run_dir / "checkpoint_best.pth")
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            print(
                f"Best checkpoint updated: score={best_metrics['val_score']:.4f}, "
                f"top1={best_metrics['val_top1']:.4f}, macro_f1={best_metrics['val_macro_f1']:.4f}"
            )

    print("\nTraining complete.")
    print(
        f"Best metrics: Top-1={best_metrics.get('val_top1', 0.0):.4f}, "
        f"Top-5={best_metrics.get('val_top5', 0.0):.4f}, "
        f"Macro-F1={best_metrics.get('val_macro_f1', 0.0):.4f}"
    )
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
