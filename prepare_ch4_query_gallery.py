import argparse
import json
import random
import shutil
from pathlib import Path

from utils.ch4_io import (
    IMAGE_EXTENSIONS,
    collect_training_identity_names,
    ensure_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="第四章 query/gallery 划分脚本")
    parser.add_argument("--test-root", type=str, default="Paper-Data-Copy/test")
    parser.add_argument("--train-annotation", type=str, default="paperdata-train.txt")
    parser.add_argument("--output-root", type=str, default="chapter4_retrieval_split")
    parser.add_argument("--query-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--clear-output", action="store_true", help="运行前清空既有输出目录")
    return parser.parse_args()


def safe_transfer(src_path, dst_path, copy_mode):
    ensure_dir(dst_path.parent)
    if dst_path.exists():
        return
    if copy_mode == "hardlink":
        try:
            dst_path.hardlink_to(src_path)
            return
        except OSError:
            pass
    shutil.copy2(src_path, dst_path)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    test_root = Path(args.test_root)
    output_root = Path(args.output_root)
    query_root = output_root / "query"
    gallery_root = output_root / "gallery"
    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)
    ensure_dir(query_root)
    ensure_dir(gallery_root)

    training_identities = set(collect_training_identity_names(args.train_annotation))
    manifest = {
        "seed": args.seed,
        "query_ratio": args.query_ratio,
        "test_root": str(test_root.resolve()),
        "training_identities": sorted(training_identities),
        "query": {},
        "gallery": {},
    }

    total_query = 0
    total_gallery = 0
    for identity_dir in sorted(test_root.iterdir()):
        if not identity_dir.is_dir() or identity_dir.name not in training_identities:
            continue

        images = [
            path for path in sorted(identity_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(images) < 2:
            continue

        rng.shuffle(images)
        query_count = max(1, min(len(images) - 1, int(len(images) * args.query_ratio)))
        query_images = images[:query_count]
        gallery_images = images[query_count:]

        manifest["query"][identity_dir.name] = [str(path.resolve()) for path in query_images]
        manifest["gallery"][identity_dir.name] = [str(path.resolve()) for path in gallery_images]

        for src_path in query_images:
            safe_transfer(src_path, query_root / identity_dir.name / src_path.name, args.copy_mode)
        for src_path in gallery_images:
            safe_transfer(src_path, gallery_root / identity_dir.name / src_path.name, args.copy_mode)

        total_query += len(query_images)
        total_gallery += len(gallery_images)

    manifest["statistics"] = {
        "num_identities": len(manifest["query"]),
        "num_query_images": total_query,
        "num_gallery_images": total_gallery,
    }
    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(manifest["statistics"], indent=2, ensure_ascii=False))
    print(f"Query root: {query_root}")
    print(f"Gallery root: {gallery_root}")


if __name__ == "__main__":
    main()
