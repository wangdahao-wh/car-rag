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

ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = ROOT / "docs"
LOGS_DIR = ROOT / "logs"


def setup_logging(loader_name: str) -> Path:
    """配置日志（实时写入，异常时不丢失）。
    
    Returns:
        日志文件路径
    """
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"build_chunks_{loader_name.lower()}_{timestamp}.log"
    
    # 自定义 FileHandler：每次写入立即刷新
    class ImmediateFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()  # 每条日志立即刷新
    
    # 清除旧的 handler（避免重复）
    logging.root.handlers.clear()
    
    # 配置 root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            ImmediateFileHandler(log_file, encoding="utf-8"),  # 文件（立即刷新）
            logging.StreamHandler(sys.stdout),                  # 控制台
        ],
        force=True,  # 强制重新配置
    )
    
    return log_file


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


def get_output_path(pdf_path: Path, loader_name: str) -> Path:
    """根据加载器类型和 PDF 文件名生成输出路径。

    开启 TABLE_ROW_SERIALIZATION（.env）时输出到 chunks_rowwise/，
    与默认的 chunks/ 并存，便于两套方案做检索效果对比。

    :param pdf_path: PDF 文件路径
    :param loader_name: 加载器名称（PyPDFLoader / MinerULoader）
    :return: chunks JSON 输出路径
    """
    pdf_stem = pdf_path.stem  # 文件名（不含 .pdf）
    loader_type = "pypdf" if "PyPDF" in loader_name else "mineru"
    chunks_dir = "chunks_rowwise" if os.environ.get("TABLE_ROW_SERIALIZATION", "0") == "1" else "chunks"
    return ROOT / "processed_docs" / loader_type / chunks_dir / f"{pdf_stem}.json"


if __name__ == "__main__":
    script_start = time.time()
    
    loader = get_loader()
    loader_name = type(loader).__name__
    
    # 配置日志
    log_file = setup_logging(loader_name)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info(f"加载器: {loader_name}")
    logger.info(f"扫描目录: {DOCS_DIR}")
    logger.info(f"日志文件: {log_file}")
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
            
            # 保存
            output_path = get_output_path(pdf_path, loader_name)
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
