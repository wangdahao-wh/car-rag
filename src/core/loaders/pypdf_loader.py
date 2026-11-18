"""pypdf 加载器：基于字体层级的结构化切分（本项目原有方案，迁移自 build_chunks_new.py）。

原理：用 pypdf 的 visitor_text 回调拿到每段文字的字号，
字号 >= 11.0 视为切分点（子标题），>= 10.0 视为标题（进路径），
其余为正文。切分后每个 chunk 带完整标题路径。

适用场景：手册类 PDF（字体层级规范、本地解析、无需联网）。

降级模式：当 PDF 字体信息缺失（扫描件）时，自动切换到段落切分模式。
"""
import logging
import re
from pathlib import Path

from pypdf import PdfReader

from .base import PDFLoader

MIN_CHARS = 30          # 少于这个字数的 chunk 丢弃
SPLIT_SIZE = 11.0       # >= 这个字号 → 切分成新小节
HEADING_SIZE = 10.0     # >= 这个字号 → 算标题（进路径）

logger = logging.getLogger(__name__)


def detect_pdf_quality(pdf_path: Path) -> str:
    """检测 PDF 质量（字体信息是否完整）。
    
    采样前3页，分析字号分布：
    - structured：字号种类 >= 4，有清晰层级
    - plain_text：字号单一（全是1.0）或种类 <= 2，字体信息缺失
    - scanned：无文字层（需 OCR）
    
    Returns:
        "structured" / "plain_text" / "scanned"
    """
    reader = PdfReader(str(pdf_path))
    sample_pages = min(3, len(reader.pages))
    
    font_sizes = []
    for i in range(sample_pages):
        page = reader.pages[i]
        def visitor(text, cm, tm, font_dict, font_size):
            if text.strip():
                font_sizes.append(round(font_size, 1))
        try:
            page.extract_text(visitor_text=visitor)
        except Exception:
            pass
    
    if not font_sizes:
        return "scanned"
    
    unique_sizes = set(font_sizes)
    # 字号种类 >= 4 种 → 有层级结构
    if len(unique_sizes) >= 4:
        return "structured"
    # 字号几乎全是 1.0 → 字体信息缺失
    if len(unique_sizes) <= 2 and 1.0 in unique_sizes:
        return "plain_text"
    
    return "plain_text"


def is_index_page(text: str) -> bool:
    """判断是否是索引页：大量「中文+数字」配对（关键字 + 页码）。"""
    return len(re.findall(r"[一-鿿]{2,}\s*\d{2,4}", text)) > 8


def extract_blocks(page) -> list[tuple[float, str]]:
    """用字体信息抽取片段，并剔除页脚（从后往前找 "Seite " 锚点）。

    Returns:
        list[tuple]: [(字号, 文本), ...]
    """
    parts = []

    def visitor(text, cm, tm, font_dict, font_size):
        t = text.strip()
        if t:
            parts.append((round(font_size, 1), t))

    page.extract_text(visitor_text=visitor)

    # 剔除页脚：找到最后一个 "Seite " 开头的片段，其后全部丢弃
    for i in range(len(parts) - 1, -1, -1):
        if parts[i][1].startswith("Seite "):
            parts = parts[:i]
            break
    return parts


def join_lines(lines: list[str]) -> str:
    """把 PDF 拆行的片段拼回完整的句子。

    规则：上一片段末尾不是句末标点 → 说明话没说完 → 直接接上。
    """
    out = ""
    for line in lines:
        if out and out[-1] not in "。！？；：":
            out += line
        else:
            out += ("\n" if out else "") + line
    return out


def build_sections(parts: list[tuple[float, str]]) -> list[tuple[list, list]]:
    """按标题切分，每节带【完整标题路径】。

    标题栈规则：遇到同级或更高级标题时，弹出栈中同级/低级标题再入栈。
    Returns:
        list[tuple]: [([(字号, 标题), ...], [正文行, ...]), ...]
    """
    sections = []
    titles = []  # 标题栈：[(字号, 标题), ...]
    lines = []

    for size, text in parts:
        if size >= HEADING_SIZE:
            # 切分点：当前节已有正文 → 收为一个小节
            if size >= SPLIT_SIZE and lines:
                sections.append((list(titles), lines))
                lines = []
            # 维护标题栈：去掉 >= 当前字号的（同级/低级标题出栈），再压入当前标题
            titles = [t for t in titles if t[0] > size] + [(size, text)]
        else:
            lines.append(text)

    if lines:
        sections.append((list(titles), lines))

    return sections


def section_text(titles: list[tuple[float, str]], lines: list[str]) -> str:
    """把标题路径 + 正文拼成 chunk 文本，形如 "[车灯 > 自动车灯控制]\n正文..."。"""
    path = " > ".join(t for _, t in titles)
    body = join_lines(lines)
    if not path:
        return body
    return f"[{path}]\n{body}"


class PyPDFLoader(PDFLoader):
    """pypdf 加载器实现（智能降级）。"""

    def load(self, pdf_path: Path) -> list[dict]:
        """加载 PDF，根据质量选择切分策略。
        
        - structured PDF → 字体层级切分
        - plain_text PDF → 段落切分（降级）
        - scanned PDF → 抛异常（需用 MinerU）
        """
        pdf_path = Path(pdf_path)
        quality = detect_pdf_quality(pdf_path)
        
        if quality == "structured":
            logger.info(f"{pdf_path.name}: 检测到字体层级，使用结构化切分")
            return self._load_structured(pdf_path)
        elif quality == "plain_text":
            logger.warning(f"{pdf_path.name}: 字体信息缺失，降级为段落切分（path 为空）")
            return self._load_plain_text(pdf_path)
        else:
            raise RuntimeError(
                f"{pdf_path.name} 是扫描件或无文字层，pypdf 无法处理。"
                f"请使用 MinerU: PDF_LOADER=mineru python src/core/build_chunks_new.py"
            )
    
    def _load_structured(self, pdf_path: Path) -> list[dict]:
        """结构化切分（原有逻辑）：基于字体层级。"""
        reader = PdfReader(str(pdf_path))
        chunks = []

        for pno, page in enumerate(reader.pages, 1):
            parts = extract_blocks(page)
            if not parts:
                continue

            for titles, lines in build_sections(parts):
                text = section_text(titles, lines)

                if len(text) < MIN_CHARS:   # 过短的丢弃
                    continue
                if text.count("...") > 5:   # 目录页丢弃
                    continue
                if is_index_page(text):     # 索引页丢弃
                    continue

                path = [t for _, t in titles]
                chunks.append({
                    "text": text,
                    "page": pno,
                    "chapter": path[0] if len(path) > 0 else "",
                    "section": path[1] if len(path) > 1 else "",
                    "path": path,
                })

        return chunks
    
    def _load_plain_text(self, pdf_path: Path) -> list[dict]:
        """降级方案：按段落切分（字体信息缺失时）。
        
        策略：按双换行分段，过滤过短段落。
        注意：chapter/section/path 为空，无结构信息。
        """
        reader = PdfReader(str(pdf_path))
        chunks = []
        
        for page_num, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text()
            except Exception:
                continue
            
            if not text.strip():
                continue
            
            # 按双换行分段
            paragraphs = re.split(r'\n\n+', text)
            
            for para in paragraphs:
                para = para.strip()
                # 过滤：过短、纯数字、纯符号
                if len(para) < MIN_CHARS:
                    continue
                if para.isdigit():
                    continue
                if all(c in '.-_=[]()' for c in para):
                    continue
                
                chunks.append({
                    "text": para,
                    "page": page_num,
                    "chapter": "",
                    "section": "",
                    "path": [],
                })
        
        return chunks
