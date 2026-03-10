import argparse
import csv
import json
from pathlib import Path

from utils.ch4_io import ensure_dir, format_latex_row


def parse_args():
    parser = argparse.ArgumentParser(description="第四章结果表导出脚本")
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help="结果项，格式为 方法名=metrics.json",
    )
    parser.add_argument("--columns", nargs="+", required=True, help="导出的列名")
    parser.add_argument("--output-prefix", type=str, default="chapter4_reports/table_export")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.result:
        raise ValueError("至少提供一个 --result 方法名=metrics.json")

    rows = []
    for item in args.result:
        if "=" not in item:
            raise ValueError(f"无效结果项: {item}")
        method_name, metrics_path = item.split("=", 1)
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        metrics["method_name"] = method_name
        rows.append(metrics)

    output_prefix = Path(args.output_prefix)
    ensure_dir(output_prefix.parent)

    csv_path = output_prefix.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method_name", *args.columns])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ["method_name", *args.columns]})

    latex_lines = [format_latex_row(row["method_name"], row, args.columns) for row in rows]
    latex_path = output_prefix.with_suffix(".tex")
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")

    print(f"CSV exported to: {csv_path}")
    print(f"LaTeX rows exported to: {latex_path}")


if __name__ == "__main__":
    main()
