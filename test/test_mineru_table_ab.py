"""MinerU 表格识别 A/B 对比：pipeline vs vlm（领克 03 配置表）。

目的：验证 model_version=vlm 对配置表符号矩阵的识别质量，
      不触发 qwen-vl 二次转录（兜底会掩盖 MinerU 原始差异）。

用法：
    /Users/wanghao/code/ai_project/rag/.venv-car/bin/python test/test_mineru_table_ab.py
    /Users/wanghao/code/ai_project/rag/.venv-car/bin/python test/test_mineru_table_ab.py \
        --pdf docs/领克_2027_03Plus_配置表.pdf

输出（都在 test/output/ 下）：
    mineru_raw/<pdf-stem>/<model_version>/   原始解析结果（含 content_list.json）
    <时间戳>/compare.md                      指标对比 + 每个表格的明细
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.core.loaders import mineru_loader as ml

logger = logging.getLogger("mineru")

# 表格单元格内出现的配置符号
SYMBOL_PATTERN = re.compile(r">\s*([●○■□])\s*<")
SYMBOL_KEYS = ["●", "○", "■", "□"]


def fetch_content_lists(
    pdf: Path,
    raw_dir: Path,
    model_version: str,
    api_key: str,
) -> tuple[list[Path], float]:
    """调用 MinerU 解析（命中本地缓存则跳过 API），返回 (content_list 路径列表, 耗时秒)。"""
    start = time.time()

    if raw_dir.exists():
        cached = sorted(raw_dir.rglob("*_content_list.json"))
        if cached:
            logger.info("[%s] 命中本地缓存（%d 个分片），跳过 API 调用", model_version, len(cached))
            return cached, time.time() - start

    parts = ml.split_pdf(pdf, output_dir=raw_dir)
    part_paths = [p for p, _, _ in parts]
    batch_id, upload_urls = ml.create_batch(api_key, part_paths, model_version=model_version)
    logger.info("[%s] batch_id: %s", model_version, batch_id)
    ml.upload_files(part_paths, upload_urls)
    extract_dirs = ml.download_results(api_key, batch_id, raw_dir)

    found = []
    for extract_dir in extract_dirs.values():
        got = sorted(extract_dir.rglob("*_content_list.json"))
        if not got:
            got = sorted(extract_dir.rglob("content_list.json"))
        if got:
            found.append(got[0])
    if not found:
        raise RuntimeError(f"[{model_version}] 解析结果中没有 content_list.json: {raw_dir}")
    return found, time.time() - start


def load_items(paths: list[Path]) -> list[dict]:
    """合并多个分片的 content_list。"""
    items = []
    for path in paths:
        items.extend(json.loads(path.read_text(encoding="utf-8")))
    return items


def analyze_tables(items: list[dict]) -> list[dict]:
    """逐个表格统计可用性、行列结构、符号保留情况。"""
    rows = []
    for item in items:
        if item.get("type") != "table":
            continue
        body = item.get("table_body") or ""
        cells = re.findall(r"<td[^>]*>([^<]*)</td>", body)
        parsed = ml.serialize_table_rows(body)
        rows.append({
            "caption": " ".join(item.get("table_caption") or []),
            "has_img": bool(item.get("img_path")),
            "body_len": len(body),
            "cell_count": len(cells),
            "empty_body": not any(c.strip() for c in cells),
            "grid_squash": any(ml.GRID_SQUASH_PATTERN.search(c) for c in cells),
            "usable": ml.table_body_is_usable(body),
            "parsed_rows": len(parsed),
            "value_pairs": sum(len(r["values"]) for r in parsed),
            "symbols": Counter(m.group(1) for m in SYMBOL_PATTERN.finditer(body)),
            "preview": re.sub(r"\s+", " ", body)[:180],
        })
    return rows


def summarize(rows: list[dict], item_count: int, cost: float) -> dict:
    """汇总指标（用于总览表）。"""
    symbols = Counter()
    for row in rows:
        symbols.update(row["symbols"])
    return {
        "解析条目数": item_count,
        "表格数": len(rows),
        "可用表体": sum(r["usable"] for r in rows),
        "空表体": sum(r["empty_body"] for r in rows),
        "网格挤压": sum(r["grid_squash"] for r in rows),
        "行级解析出行数": sum(r["parsed_rows"] for r in rows),
        "配置值对": sum(r["value_pairs"] for r in rows),
        "● 标配": symbols.get("●", 0),
        "○ 选装": symbols.get("○", 0),
        "■ 标配": symbols.get("■", 0),
        "□ 选装": symbols.get("□", 0),
        "耗时(秒)": f"{cost:.1f}",
    }


def write_report(out_path: Path, pdf: Path, versions: list[str], results: dict) -> None:
    """生成 markdown 对比报告。"""
    lines = [
        "# MinerU 表格识别对比：pipeline vs vlm",
        "",
        f"- PDF：`{pdf.name}`",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 说明：未触发 qwen-vl 二次转录，指标反映 MinerU 原始输出",
        "",
        "## 总览",
        "",
    ]

    metrics = list(results[versions[0]]["summary"].keys())
    header = "| 指标 | " + " | ".join(versions) + " |"
    sep = "| --- | " + " | ".join("---" for _ in versions) + " |"
    lines += [header, sep]
    for metric in metrics:
        cells = [str(results[v]["summary"][metric]) for v in versions]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## 表格明细", ""]
    for version in versions:
        lines.append(f"### {version}")
        lines.append("")
        rows = results[version]["rows"]
        if not rows:
            lines += ["（没有识别到表格）", ""]
            continue
        for i, row in enumerate(rows, 1):
            symbols = "  ".join(f"{k}x{row['symbols'].get(k, 0)}" for k in SYMBOL_KEYS)
            lines += [
                f"#### 表 {i}：{row['caption'] or '(无标题)'}",
                "",
                f"- 可用表体：{'是' if row['usable'] else '否'}"
                f"；空表体：{'是' if row['empty_body'] else '否'}"
                f"；网格挤压：{'是' if row['grid_squash'] else '否'}",
                f"- 单元格数：{row['cell_count']}；行级解析行数：{row['parsed_rows']}"
                f"；配置值对：{row['value_pairs']}",
                f"- 符号：{symbols}",
                f"- body 预览：`{row['preview']}`",
                "",
            ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="MinerU 表格识别 A/B 对比")
    parser.add_argument(
        "--pdf",
        default=str(ROOT / "docs" / "领克_2027_03_配置表.pdf"),
        help="待解析 PDF，默认领克 2027 款 03 配置表",
    )
    parser.add_argument(
        "--versions",
        default="pipeline,vlm",
        help="对比的 model_version，逗号分隔",
    )
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"PDF 不存在: {pdf}")

    api_key = os.environ.get("MINERU_API_KEY")
    if not api_key:
        raise SystemExit("未配置 MINERU_API_KEY（请在 .env 中填写）")

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "test" / "output" / f"{stamp}_{pdf.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for version in versions:
        raw_dir = ROOT / "test" / "output" / "mineru_raw" / pdf.stem / version
        raw_dir.mkdir(parents=True, exist_ok=True)
        content_list_paths, cost = fetch_content_lists(pdf, raw_dir, version, api_key)
        items = load_items(content_list_paths)
        rows = analyze_tables(items)
        results[version] = {
            "rows": rows,
            "summary": summarize(rows, len(items), cost),
        }
        logger.info(
            "[%s] 条目 %d，表格 %d，可用 %d，行级行数 %d，符号 %s",
            version,
            len(items),
            len(rows),
            sum(r["usable"] for r in rows),
            sum(r["parsed_rows"] for r in rows),
            dict(Counter({k: v for k, v in results[version]["summary"].items() if k in SYMBOL_KEYS})),
        )

    report_path = out_dir / "compare.md"
    write_report(report_path, pdf, versions, results)
    print(f"\n对比报告: {report_path}")
    print(f"原始结果: {ROOT / 'test' / 'output' / 'mineru_raw' / pdf.stem}")


if __name__ == "__main__":
    main()
