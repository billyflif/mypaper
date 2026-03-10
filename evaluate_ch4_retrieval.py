import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.ch4_protocol import (
    IdentityImageFolderDataset,
    build_model,
    compute_distance_matrix,
    count_parameters,
    evaluate_retrieval,
    extract_embeddings,
    identity_folder_collate,
    load_checkpoint,
    save_metric_bundle,
)
from utils.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="第四章补充检索评估脚本")
    parser.add_argument("--model-type", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--query-root", required=True, type=str)
    parser.add_argument("--gallery-root", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default="chapter4_reports/retrieval")
    parser.add_argument("--method-name", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--distance-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--pretrained", action="store_true")
    return parser.parse_args()


def extract_all_embeddings(model, dataloader, device):
    features = []
    identities = []
    paths = []
    model.eval()
    with torch.no_grad():
        for images, masks, batch_ids, batch_paths in tqdm(dataloader, desc="Embedding", mininterval=0.3):
            images = images.to(device)
            masks = masks.to(device)
            batch_features = extract_embeddings(model, images, masks=masks)
            features.append(batch_features.cpu())
            identities.extend(batch_ids)
            paths.extend(batch_paths)
    return torch.cat(features, dim=0), identities, paths


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_dataset = IdentityImageFolderDataset(args.query_root, input_shape=(128, 128))
    gallery_dataset = IdentityImageFolderDataset(args.gallery_root, input_shape=(128, 128))
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=identity_folder_collate,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=identity_folder_collate,
    )

    unique_ids = sorted({identity for _, identity in query_dataset.samples} | {identity for _, identity in gallery_dataset.samples})
    model = build_model(
        model_type=args.model_type,
        num_classes=max(len(unique_ids), 1),
        device=device,
        pretrained=args.pretrained,
    )
    load_checkpoint(model, args.checkpoint, device)

    query_features, query_ids, query_paths = extract_all_embeddings(model, query_loader, device)
    gallery_features, gallery_ids, gallery_paths = extract_all_embeddings(model, gallery_loader, device)
    distance_matrix = compute_distance_matrix(query_features, gallery_features, metric=args.distance_metric)
    retrieval_metrics = evaluate_retrieval(distance_matrix, query_ids, gallery_ids, max_rank=5)
    total_params, trainable_params = count_parameters(model)

    method_name = args.method_name or args.model_type
    metrics = {
        "method_name": method_name,
        "model_type": args.model_type,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "query_root": str(Path(args.query_root).resolve()),
        "gallery_root": str(Path(args.gallery_root).resolve()),
        "distance_metric": args.distance_metric,
        "num_query_images": len(query_dataset),
        "num_gallery_images": len(gallery_dataset),
        "num_identities": len(unique_ids),
        "rank1": retrieval_metrics["rank1"],
        "rank5": retrieval_metrics["rank5"],
        "mAP": retrieval_metrics["mAP"],
        "valid_queries": retrieval_metrics["valid_queries"],
        "params_total": total_params,
        "params_trainable": trainable_params,
        "query_feature_dim": query_features.shape[1],
        "gallery_feature_dim": gallery_features.shape[1],
    }

    output_paths = save_metric_bundle(
        output_dir=args.output_dir,
        stem=method_name.replace(" ", "_").lower(),
        metrics=metrics,
        latex_columns=["rank1", "rank5", "mAP"],
    )

    feature_manifest = {
        "query_paths": query_paths,
        "gallery_paths": gallery_paths,
    }
    feature_manifest_path = Path(args.output_dir) / f"{method_name.replace(' ', '_').lower()}_pairs.json"
    feature_manifest_path.write_text(json.dumps(feature_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Results saved to: {output_paths}")


if __name__ == "__main__":
    main()
