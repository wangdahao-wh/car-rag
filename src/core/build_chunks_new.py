"""建 chunk 入口：按 PDF_LOADER 配置选择加载器解析 PDF，输出统一 chunks.json。

用法：
    python src/core/build_chunks_new.py                 # 默认 pypdf 加载器
    PDF_LOADER=mineru python src/core/build_chunks_new.py   # MinerU 在线解析
"""
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.loaders import get_loader
from src.core.utils import setup_logging

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"

logger = logging.getLogger(__name__)


def print_stats(chunks, logger=None):
    """打印统计"""
    log = logger.info if logger else print
    
    if not chunks:
        log("警告：没有生成任何 chunk")
        return
    
    lens = [len(c["text"]) for c in chunks]
    total_pages = max(c["page"] for c in chunks) if chunks else 0
    
    log(f"总页数:       {total_pages}（来自 chunk 最大页码）")
    log(f"生成 chunk:   {len(chunks)}")
    log(f"总字数:       {sum(lens)}")
    log(f"平均字数:     {sum(lens) // len(lens)}")
    log(f"中位字数:     {int(median(lens))}")
    log(f"最长/最短:    {max(lens)} / {min(lens)}")

    log(f"\n字数分布:")
    for lo, hi, name in [(0,100,'<100'), (100,300,'100-300'), (300,600,'300-600'), (600,99999,'>600')]:
        n = sum(1 for x in lens if lo <= x < hi)
        log(f"  {name:<10} {n:>4} 个")

    log(f"\n章节分布（前 10）:")
    counter = Counter(c["path"][0] if c["path"] else "(无)" for c in chunks)
    for name, n in counter.most_common(10):
        log(f"  {name:<20} {n:>4} 个")


def save_chunks(chunks, path, logger=None):
    """保存成 JSON"""
    log = logger.info if logger else print
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    log(f"\n已保存 {len(chunks)} 个 chunk 到 {path}")


def get_output_path(pdf_path: Path, chunks: list[dict]) -> Path:
    """根据 PDF 文件名和实际解析加载器生成输出路径。

    输出目录按 chunk 里的 loader 字段决定（pypdf / mineru），
    AutoLoader 路由后各文件落到各自目录，不会统一落 mineru/。

    开启 TABLE_ROW_SERIALIZATION（.env）时输出到 chunks_rowwise/，
    与默认的 chunks/ 并存，便于两套方案做检索效果对比。

    文件名带构建时间戳（如 领克_2027_03_配置表_20261120_1806.json），
    用于区分新旧产物；同时清理同 stem 的旧版本（无时间戳老文件 +
    历史时间戳版本），避免下游 build_index 重复加载。

    :param pdf_path: PDF 文件路径
    :param chunks: 该 PDF 解析出的 chunk 列表（取 loader 字段）
    :return: chunks JSON 输出路径
    """
    pdf_stem = pdf_path.stem  # 文件名（不含 .pdf）
    loader_type = chunks[0]["loader"] if chunks else "mineru"
    chunks_dir = "chunks_rowwise" if os.environ.get("TABLE_ROW_SERIALIZATION", "0") == "1" else "chunks"
    out_dir = ROOT / "processed_docs" / loader_type / chunks_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧版本：未加时间戳的老文件 + 带时间戳的历史版本
    for old in list(out_dir.glob(f"{pdf_stem}.json")) + list(out_dir.glob(f"{pdf_stem}_2*.json")):
        old.unlink()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{pdf_stem}_{timestamp}.json"


if __name__ == "__main__":
    script_start = time.time()
    
    loader = get_loader()
    loader_name = type(loader).__name__
    
    # 配置日志
    log_file = setup_logging("build_chunks", loader_name.lower())
    
    logger.info("=" * 60)
    logger.info(f"加载器: {loader_name}")
    logger.info(f"扫描目录: {DOCS_DIR}")
    logger.info("=" * 60)
    logger.info("")
    
    # 获取所有 PDF 文件
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"错误：{DOCS_DIR} 目录下没有找到 PDF 文件")
        sys.exit(1)
    
    logger.info(f"找到 {len(pdf_files)} 个 PDF 文件\n")
    
    # 统计信息
    success_count = 0
    fail_count = 0
    total_chunks = 0
    file_times = []
    
    # 逐个处理
    for idx, pdf_path in enumerate(pdf_files, 1):
        logger.info("=" * 60)
        logger.info(f"[{idx}/{len(pdf_files)}] 处理: {pdf_path.name}")
        logger.info("=" * 60)
        
        file_start = time.time()
        try:
            # 加载 + 切分
            chunks = loader.load(pdf_path)
            
            # 统计
            print_stats(chunks, logger)
            
            # 保存（输出目录按 chunk 实际的 loader 字段决定）
            output_path = get_output_path(pdf_path, chunks)
            save_chunks(chunks, output_path, logger)
            
            # 抽样展示
            logger.info("\n" + "-" * 60)
            sample_indices = [0, len(chunks) // 2, len(chunks) - 1] if len(chunks) >= 3 else range(len(chunks))
            for i in sample_indices:
                c = chunks[i]
                logger.info(f"\nchunk[{i}] 第{c['page']}页:")
                logger.info(c["text"][:150] + ("..." if len(c["text"]) > 150 else ""))
            
            # 记录成功
            file_elapsed = time.time() - file_start
            file_times.append((pdf_path.name, file_elapsed, len(chunks)))
            total_chunks += len(chunks)
            success_count += 1
            logger.info(f"\n✅ 处理成功，耗时 {file_elapsed:.1f} 秒")
        
        except Exception as e:
            fail_count += 1
            logger.error(f"\n❌ 处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info("")  # 空行分隔
    
    # 总结报告
    total_elapsed = time.time() - script_start
    logger.info("=" * 60)
    logger.info("批量处理完成")
    logger.info("=" * 60)
    logger.info(f"加载器:       {loader_name}")
    logger.info(f"成功:         {success_count}/{len(pdf_files)} 个文件")
    logger.info(f"失败:         {fail_count} 个文件")
    logger.info(f"生成 chunk:   {total_chunks} 个")
    logger.info(f"总耗时:       {total_elapsed:.1f} 秒")
    
    if file_times:
        avg_time = sum(t for _, t, _ in file_times) / len(file_times)
        logger.info(f"平均耗时:     {avg_time:.1f} 秒/文件")
        logger.info(f"\n各文件处理时间:")
        for name, elapsed, chunks in file_times:
            logger.info(f"  {name:<40} {elapsed:>6.1f} 秒  ({chunks} chunks)")
    
    logger.info("=" * 60)
    logger.info(f"日志已保存: {log_file}")
