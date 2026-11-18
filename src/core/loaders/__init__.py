"""PDF 加载器工厂：按环境变量 PDF_LOADER 选择实现。

配置（.env）：
    PDF_LOADER=auto     # 自动检测（推荐）：根据 PDF 质量智能选择加载器
    PDF_LOADER=pypdf    # 强制 pypdf 本地解析（字体层级切分，快速）
    PDF_LOADER=mineru   # 强制 MinerU 在线解析（OCR + 表格识别，慢但质量高）
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .base import PDFLoader
from .pypdf_loader import PyPDFLoader, detect_pdf_quality

# 加载项目根目录 .env（MINERU_API_KEY / PDF_LOADER）
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

logger = logging.getLogger(__name__)


class AutoLoader(PDFLoader):
    """自动加载器：根据 PDF 质量智能选择 pypdf 或 mineru。
    
    策略：
        - structured PDF（字体层级完整）→ pypdf（快速）
        - plain_text / scanned PDF（字体信息缺失）→ mineru（质量优先）
    """
    
    def load(self, pdf_path: Path) -> list[dict]:
        """自动选择最佳加载器。"""
        pdf_path = Path(pdf_path)
        quality = detect_pdf_quality(pdf_path)
        
        if quality == "structured":
            logger.info(f"📄 {pdf_path.name}: 检测到字体层级 → 使用 pypdf（快速本地解析）")
            return PyPDFLoader().load(pdf_path)
        else:
            logger.info(f"🔍 {pdf_path.name}: 字体信息缺失 → 使用 mineru（在线 OCR，产生费用）")
            from .mineru_loader import MinerULoader
            return MinerULoader().load(pdf_path)


def get_loader(name: str | None = None) -> PDFLoader:
    """按名称获取加载器实例；不传则读环境变量 PDF_LOADER，默认 auto。"""
    name = name or os.environ.get("PDF_LOADER", "auto")
    
    if name == "auto":
        return AutoLoader()
    elif name == "pypdf":
        return PyPDFLoader()
    elif name == "mineru":
        # 延迟导入：未配置 MINERU_API_KEY 时 import 不报错，只在实例化时校验
        from .mineru_loader import MinerULoader
        return MinerULoader()
    else:
        raise ValueError(f"未知的 PDF_LOADER: {name}（可选：auto / pypdf / mineru）")


__all__ = ["PDFLoader", "PyPDFLoader", "AutoLoader", "get_loader"]
