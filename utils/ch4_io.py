import os
from pathlib import Path


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


def format_metric_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_latex_row(method_name, metrics, columns):
    values = [method_name]
    for column in columns:
        values.append(format_metric_value(metrics.get(column, "")))
    return " & ".join(values) + r" \\"
