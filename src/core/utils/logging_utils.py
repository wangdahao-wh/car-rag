"""日志配置工具：控制台 + 文件双输出。"""
import logging
import sys
from datetime import datetime
from pathlib import Path


class ImmediateFileHandler(logging.FileHandler):
    """每条日志立即刷盘，避免进程异常退出时丢日志。"""
    
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(script_name: str, suffix: str = "") -> Path:
    """配置日志：控制台 + logs/{script_name}_{suffix}_{时间}.log。
    
    Args:
        script_name: 脚本名（如 build_index / test_retrieval）
        suffix: 可选后缀（如向量库名、模型名）
    
    Returns:
        日志文件路径
    """
    root = Path(__file__).resolve().parents[3]
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [script_name]
    if suffix:
        parts.append(suffix)
    parts.append(timestamp)
    log_file = logs_dir / f"{'_'.join(parts)}.log"
    
    logging.root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            ImmediateFileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("日志文件: %s", log_file)
    return log_file
