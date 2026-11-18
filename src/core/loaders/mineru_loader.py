"""MinerU 加载器：调用 MinerU 在线解析服务，按标题层级切分。

参考：知乎大模型课程 23期/35/RAG-cy/src/pdf_mineru.py（v4 文件上传模式）。
流程：本地 PDF →（超 200 页先切分片）→ 申请上传地址 → 上传 → 轮询 → 下载 zip 解压
     → 读 content_list.json（含 text_level / page_idx）→ 组装成统一 chunk。

切分策略（参考 mineru-structured-chunking skill）：
  1. 标题分节：遇 text_level=N 开新节
  2. 内容类型：text/table/image/chart 按规则序列化
  3. 小节合并：< 50 token 并入下一节
  4. 长文本细分：> 300 token 递归切分
  5. 表格行级序列化：大表格分组复读表头

与 pypdf 加载器的差异：解析质量更高（OCR、表格、复杂排版），但依赖网络和 API Key。
API Key 从环境变量 MINERU_API_KEY 读取（写入 .env）。
"""
import base64
import json
import logging
import os
import re
import time
import zipfile
from pathlib import Path

import requests
from html.parser import HTMLParser
from pypdf import PdfReader, PdfWriter

from .base import PDFLoader

MAX_PAGES = 200          # MinerU 单文件页数上限
POLL_INTERVAL = 10       # 轮询间隔（秒）
VL_MODEL = "qwen3-vl-plus"  # 空表体二次提取用的视觉模型（qwen-vl-ocr 会丢弃符号单元格，不能用）

# 表格行级序列化开关（.env: TABLE_ROW_SERIALIZATION=1 开启，默认关）
# 开启后 serialize_table 把 HTML 表体转成「配置项 | 列名: 值」的自解释文本，
# 用于和默认的原始 HTML 方案做检索效果对比
TABLE_ROW_SERIALIZATION = os.environ.get("TABLE_ROW_SERIALIZATION", "0") == "1"

TABLE_HEADER_ROWS = 2  # 配置表通常两级表头：第一行车型名 + 第二行动力总成

# 配置表符号 → 语义文字（单元格内容精确匹配整个符号才替换）
_TABLE_SYMBOL_MAP = {
    "●": "标配",
    "○": "选装",
    "O": "选装",
    "o": "选装",
    "-": "不含",
    "—": "不含",
}

# 空表体转录提示词：图片型/符号型配置表转 HTML
TABLE_TRANSCRIBE_PROMPT = (
    "把图片中的表格完整转录为 HTML <table>，只输出表格本身，不要输出任何解释。"
    "最重要的要求：表格中的符号单元格（实心圆点、空心圆点、短横线）必须逐格转录，"
    "绝对不能省略——每个配置项所在行，在每个车型列下都必须有对应的单元格，"
    "内容是●（标配）、O（选装）或-（不含）其中一个。"
    "其他要求：1. 单元格内容原样转录；2. 合并单元格用 rowspan/colspan 表示；"
    "3. 表头和所有行列都要完整，不要遗漏。"
)

logger = logging.getLogger("mineru")
if not logger.handlers:
    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S", level=logging.INFO)


def split_pdf(pdf_path: Path, output_dir: Path, max_pages: int = MAX_PAGES) -> list[tuple[Path, int, int]]:
    """把超过页数上限的 PDF 切成多个分片，写入 output_dir（不污染源文档目录）。

    Returns:
        [(分片路径, 起始页, 结束页), ...]；未超限时返回 [(原路径, 1, 总页数)]（页码 1 起）
    """
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    if total <= max_pages:
        return [(pdf_path, 1, total)]

    output_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for start in range(0, total, max_pages):
        writer = PdfWriter()
        for page in reader.pages[start:start + max_pages]:
            writer.add_page(page)
        end = min(start + max_pages, total)
        part_path = output_dir / f"{pdf_path.stem}_part{start // max_pages + 1}.pdf"
        with open(part_path, "wb") as f:
            writer.write(f)
        parts.append((part_path, start + 1, end))
        logger.info("已切分: %s（第 %d-%d 页）", part_path.name, start + 1, end)
    return parts


def create_batch(api_key: str, file_paths: list[Path]) -> tuple[str, list[str]]:
    """申请批量上传，返回 (batch_id, 预签名上传地址列表)。"""
    res = requests.post(
        "https://mineru.net/api/v4/file-urls/batch",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json={
            "enable_formula": False,
            "language": "ch",
            "files": [{"name": p.name} for p in file_paths],
        },
        timeout=30,
    )
    payload = res.json()
    return payload["data"]["batch_id"], payload["data"]["file_urls"]


def upload_files(file_paths: list[Path], upload_urls: list[str]) -> None:
    """把本地 PDF 逐个 PUT 到预签名地址（地址自带权限，无需鉴权头）。"""
    for path, url in zip(file_paths, upload_urls):
        with open(path, "rb") as f:
            res = requests.put(url, data=f, timeout=300)
        logger.info("上传 %s: %d", path.name, res.status_code)


def download_results(api_key: str, batch_id: str, output_dir: Path) -> dict[str, Path]:
    """轮询批量任务直到全部进入终态，下载 zip 并解压。

    Returns:
        {文件名: 解压目录}；失败的任务不在结果里
    
    注：zip 压缩包保留在 output_dir，解压目录为 output_dir/{文件名}/
    """
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    # 轮询直到所有文件 done 或 failed
    while True:
        results = requests.get(url, headers=headers, timeout=30).json()["data"]["extract_result"]
        for item in results:
            logger.info("%s: %s", item.get("file_name"), item.get("state"))
        if all(item.get("state") in ("done", "failed") for item in results):
            break
        logger.info("任务未完成，%d 秒后重试...", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)

    output_dir.mkdir(parents=True, exist_ok=True)
    extract_dirs = {}
    for item in results:
        if item.get("state") != "done":
            continue
        name = item["file_name"]
        zip_path = output_dir / f"{name}.zip"
        
        # 下载 zip 压缩包
        r = requests.get(item["full_zip_url"], stream=True, timeout=300)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info("已下载: %s", zip_path)
        
        # 解压到同名目录
        extract_dir = output_dir / Path(name).stem
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        logger.info("已解压: %s", extract_dir)
        
        extract_dirs[name] = extract_dir
    return extract_dirs


def estimate_tokens(text: str) -> int:
    """估算 token 数（中文按字符数，英文按空格分词数）。"""
    chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
    english = len(re.findall(r'\b\w+\b', text))
    return chinese + english


class _TableRowsParser(HTMLParser):
    """把 HTML 表格解析为行列表（colspan 展开为重复值，rowspan 暂不处理）。

    self.rows: [[单元格文本, ...], ...]，展开 colspan 保证各行列对齐。
    """

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._colspan = 1

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._colspan = int(dict(attrs).get("colspan") or 1)

    def handle_endtag(self, tag):
        if tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None:
            text = "".join(self._cell).strip()
            self._row.extend([text] * self._colspan)
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def serialize_table_rowwise(html: str, header_rows: int = TABLE_HEADER_ROWS) -> str:
    """表格行级序列化：每行展开为「配置项 | 列名: 值」的自解释文本。

    纯本地解析（html.parser 标准库），不调用任何模型，零 token 消耗。
    规则：
        1. 前 header_rows 行作为表头，逐列拼成完整列名（两级表头如 纯Lite(1.5TD-6MT)）
        2. 符号单元格替换为语义文字（●→标配、○/O→选装、-→不含）
        3. colspan 展开后，连续同值列合并显示（列A/列B: 值）
        4. 分组行（首列有值、其余全空）输出为 [分组名]
        5. 每行独立成行且自带列名，后续按行切分 chunk 不会丢失列语义
    """
    parser = _TableRowsParser()
    parser.feed(html)
    # 丢弃整行为空的行
    rows = [r for r in parser.rows if any(r)]
    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    header = rows[:header_rows]

    # 逐列拼接多级表头：某一级为空时只取非空的层级
    col_names = []
    for i in range(col_count):
        parts = [h[i] for h in header if i < len(h) and h[i]]
        if len(parts) > 1:
            col_names.append("(".join(parts) + ")")
        else:
            col_names.append(parts[0] if parts else "")

    lines = []
    for row in rows[header_rows:]:
        # 行内单元格不足列数时补空
        cells = [row[i] if i < len(row) else "" for i in range(col_count)]
        # 分组行：首列有值、其余全空 → 小节标题
        if cells[0] and not any(cells[1:]):
            lines.append(f"[{cells[0]}]")
            continue
        # 数据行：首列是配置项名，其余列合并连续同值
        label = cells[0]
        pairs = []
        i = 1
        while i < col_count:
            val = _TABLE_SYMBOL_MAP.get(cells[i], cells[i])
            j = i + 1
            while j < col_count and _TABLE_SYMBOL_MAP.get(cells[j], cells[j]) == val:
                j += 1
            cols = "/".join(n for n in col_names[i:j] if n)
            if val:
                pairs.append(f"{cols}: {val}" if cols else val)
            i = j
        if label or pairs:
            lines.append(" | ".join([label] + pairs))
    return "\n".join(lines)


def serialize_table(item: dict) -> str:
    """序列化表格条目。

    MinerU schema：table_caption / table_footnote 是字符串列表，table_body 是 HTML 字符串。
    默认返回：caption + 原始 HTML 表体 + footnote。
    开启 TABLE_ROW_SERIALIZATION（.env）后，表体改为行级序列化文本，
    每个 chunk 自带列名和符号语义，便于 embedding 检索对比。
    """
    caption = "\n".join(item.get("table_caption") or [])
    body = item.get("table_body") or ""
    footnote = "\n".join(item.get("table_footnote") or [])
    if TABLE_ROW_SERIALIZATION and body:
        body = serialize_table_rowwise(body) or body
    return "\n".join(p for p in (caption, body, footnote) if p)


def serialize_chart(item: dict) -> str:
    """序列化图表条目（只取标题，内容通常为空）。"""
    return "\n".join(item.get("chart_caption") or [])


def serialize_image(item: dict) -> str:
    """序列化图片条目（只取标题）。"""
    return "\n".join(item.get("img_caption") or [])


def table_body_is_empty(body: str | None) -> bool:
    """判断表格 HTML 是否为空表体（单元格全部无内容）。"""
    if not body:
        return True
    cells = re.findall(r"<td[^>]*>([^<]*)</td>", body)
    return not any(c.strip() for c in cells)


def transcribe_table_image(image_path: Path, api_key: str, vl_url: str) -> str:
    """调用 qwen-vl 把表格图片转录为 HTML 表格（空表体二次提取）。

    MinerU 对图片型/符号型表格（如配置表的实心/空心圆点矩阵）只输出空表体，
    表格内容只在裁剪图片里，需要视觉模型转录。
    """
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    res = requests.post(
        vl_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": VL_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": TABLE_TRANSCRIBE_PROMPT},
                    ],
                }
            ],
        },
        timeout=300,
    )
    res.raise_for_status()
    html = res.json()["choices"][0]["message"]["content"].strip()
    # 去掉模型可能包裹的 markdown 代码块
    html = re.sub(r"^```(?:html)?\s*|\s*```$", "", html)
    if "<table" not in html:
        raise RuntimeError(f"qwen-vl 转录结果不是表格: {html[:200]}")
    return html


def recursive_split(text: str, max_tokens: int = 300, overlap: int = 50) -> list[str]:
    """递归切分长文本（简化版，按句子切分）。
    
    策略：按句号/问号/感叹号切句，累积到 max_tokens 后输出一个 chunk。
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]
    
    sentences = re.split(r'([。？！\n]+)', text)
    chunks = []
    current = []
    current_tokens = 0
    
    for i in range(0, len(sentences), 2):
        sent = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
        sent_tokens = estimate_tokens(sent)
        
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append("".join(current))
            # 重叠：保留最后 overlap token 的句子
            overlap_text = []
            overlap_count = 0
            for s in reversed(current):
                overlap_count += estimate_tokens(s)
                overlap_text.insert(0, s)
                if overlap_count >= overlap:
                    break
            current = overlap_text
            current_tokens = overlap_count
        
        current.append(sent)
        current_tokens += sent_tokens
    
    if current:
        chunks.append("".join(current))
    
    return chunks


def chunks_from_content_list(
    content_list: list[dict],
    page_offset: int,
    min_section_tokens: int = 50,
    max_chunk_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[dict]:
    """把 MinerU 的 content_list.json 组装成统一 chunk（结构化切分 pipeline）。

    content_list 条目关键字段：
        type:       "text" / "image" / "table" / "chart" / "header" / "footer"
        text:       文本内容（text 类型）
        text_level: 仅标题条目有，1=章 2=节 3=子标题...
        page_idx:   所在页（0 起，相对当前分片）
        table_caption / table_body: 表格（序列化为文本）
        chart_caption: 图表标题
        img_caption: 图片标题
    
    切分 Pipeline：
        1. parse_items       → 按标题分节 + 序列化内容类型
        2. merge_small       → 小节合并（< min_section_tokens）
        3. split_large       → 长文本细分（> max_chunk_tokens）
        4. format_chunks     → 组装最终 chunk 格式
    
    :param content_list: MinerU 解析的内容列表
    :param page_offset: 页码偏移（用于多分片 PDF）
    :param min_section_tokens: 小节最小 token 数，默认 50
    :param max_chunk_tokens: chunk 最大 token 数，默认 300
    :param overlap_tokens: 细分时重叠 token 数，默认 50
    """
    # Pipeline 阶段 1：解析条目 → 按标题分节
    sections = _parse_items_to_sections(content_list, page_offset)
    logger.debug("阶段 1 - 解析完成：%d 个小节", len(sections))
    
    # Pipeline 阶段 2：合并过小的节
    sections = _merge_small_sections(sections, min_section_tokens)
    logger.debug("阶段 2 - 小节合并后：%d 个小节", len(sections))
    
    # Pipeline 阶段 3：拆分超长的节
    sections = _split_large_sections(sections, max_chunk_tokens, overlap_tokens)
    logger.debug("阶段 3 - 长文本拆分后：%d 个 chunk", len(sections))
    
    # Pipeline 阶段 4：组装最终格式
    chunks = _format_chunks(sections)
    logger.debug("阶段 4 - 格式化完成：%d 个 chunk", len(chunks))
    
    return chunks


def _parse_items_to_sections(content_list: list[dict], page_offset: int) -> list[dict]:
    """Pipeline 阶段 1：解析条目，按标题分节 + 序列化各类型内容。
    
    Returns:
        [{"titles": [(level, text)], "lines": [str], "page": int}, ...]
    """
    sections = []
    titles = []    # 标题栈
    lines = []
    section_page = None

    def flush_section():
        """收集当前小节。"""
        nonlocal lines, section_page
        if not lines:
            return
        sections.append({
            "titles": titles.copy(),
            "lines": lines.copy(),
            "page": section_page,
        })
        lines = []
        section_page = None

    for item in content_list:
        item_type = item.get("type", "")
        page_idx = page_offset + item.get("page_idx", 0)
        
        # 跳过页眉页脚
        if item_type in ("header", "footer"):
            continue
        
        # 标题：先收上一节，再维护标题栈
        if item_type == "text" and item.get("text_level"):
            flush_section()
            level = item["text_level"]
            title_text = item["text"].strip()
            titles = [t for t in titles if t[0] < level] + [(level, title_text)]
            continue
        
        # 序列化各类型内容
        content = ""
        if item_type == "text":
            content = item.get("text", "").strip()
        elif item_type == "table":
            content = serialize_table(item)
        elif item_type == "chart":
            content = serialize_chart(item)
        elif item_type == "image":
            content = serialize_image(item)
        
        if content:
            if section_page is None:
                section_page = page_idx
            lines.append(content)
    
    flush_section()
    return sections


def _merge_small_sections(sections: list[dict], min_section_tokens: int) -> list[dict]:
    """Pipeline 阶段 2：合并过小的节（< min_section_tokens）。
    
    合并规则：同级或下级节可合并，跨章节不合并。
    """
    if not sections:
        return sections
    
    merged = []
    i = 0
    while i < len(sections):
        sec = sections[i]
        body = "\n".join(sec["lines"])
        
        # 过小的节尝试并入下一节
        while estimate_tokens(body) < min_section_tokens and i + 1 < len(sections):
            next_sec = sections[i + 1]
            # 只有同级或下级才能合并（避免跨章合并）
            if not next_sec["titles"] or (sec["titles"] and next_sec["titles"][0][0] > sec["titles"][-1][0]):
                sec["lines"].extend(next_sec["lines"])
                body = "\n".join(sec["lines"])
                logger.debug("合并小节: %s + %s", 
                            _get_section_title(sec), 
                            _get_section_title(next_sec))
                i += 1
            else:
                break
        
        merged.append(sec)
        i += 1
    
    return merged


def _split_large_sections(
    sections: list[dict],
    max_chunk_tokens: int,
    overlap_tokens: int
) -> list[dict]:
    """Pipeline 阶段 3：拆分超长的节（> max_chunk_tokens）。
    
    超长节按句子边界递归拆分，保持重叠。
    """
    result = []
    for sec in sections:
        body = "\n".join(sec["lines"])
        
        if estimate_tokens(body) > max_chunk_tokens:
            # 超长文本递归拆分
            sub_texts = recursive_split(body, max_chunk_tokens, overlap_tokens)
            logger.debug("拆分超长节: %s → %d 个子块", 
                        _get_section_title(sec), 
                        len(sub_texts))
            for sub in sub_texts:
                result.append({
                    "titles": sec["titles"],
                    "lines": [sub],  # 拆分后的文本作为单行
                    "page": sec["page"],
                })
        else:
            result.append(sec)
    
    return result


def _format_chunks(sections: list[dict]) -> list[dict]:
    """Pipeline 阶段 4：组装最终 chunk 格式。
    
    添加面包屑路径、提取 chapter/section 字段。
    """
    chunks = []
    for sec in sections:
        path = [t for _, t in sec["titles"]]
        breadcrumb = f"[{' > '.join(path)}]\n" if path else ""
        body = "\n".join(sec["lines"])
        
        chunks.append({
            "text": breadcrumb + body,
            "page": sec["page"],
            "chapter": path[0] if len(path) > 0 else "",
            "section": path[1] if len(path) > 1 else "",
            "path": path,
        })
    
    return chunks


def _get_section_title(sec: dict) -> str:
    """辅助函数：获取小节标题（用于日志）。"""
    if not sec["titles"]:
        return "(无标题)"
    return " > ".join(t for _, t in sec["titles"])


class MinerULoader(PDFLoader):
    """MinerU 在线解析加载器。"""

    def __init__(
        self,
        work_dir: Path | None = None,
        min_section_tokens: int = 50,
        max_chunk_tokens: int = 300,
        overlap_tokens: int = 50,
    ):
        """
        :param work_dir: zip 下载与解压目录，默认 <项目根>/processed_docs/mineru/raw/
        :param min_section_tokens: 小节最小 token 数，默认 50
        :param max_chunk_tokens: chunk 最大 token 数，默认 300
        :param overlap_tokens: 细分时重叠 token 数，默认 50
        """
        self.api_key = os.environ.get("MINERU_API_KEY")
        if not self.api_key:
            raise RuntimeError("未配置 MINERU_API_KEY：请在 .env 中填写（https://mineru.net 申请）")
        self.work_dir = work_dir or Path(__file__).resolve().parents[3] / "processed_docs" / "mineru" / "raw"
        self.dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")  # 空表体二次提取用
        self.min_section_tokens = min_section_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def load(self, pdf_path: Path) -> list[dict]:
        """解析 PDF（pipeline 模式，带本地缓存检查）。
        
        Pipeline：
            1. prepare_files    → PDF 分片（超 200 页）
            2. upload_to_mineru → 申请 batch + 上传
            3. download_results → 轮询 + 下载 zip + 解压
            4. parse_and_chunk  → 读 content_list + 切分 chunk
        """
        pdf_path = Path(pdf_path)
        logger.info("=" * 60)
        logger.info("开始处理: %s", pdf_path.name)
        total_start = time.time()
        
        # Pipeline 阶段 1：准备文件（分片，缓存和首次处理都需要页码区间）
        parts = self._prepare_files(pdf_path)
        
        # 检查本地缓存：每个分片的解析结果都齐（目录存在且有 content_list）才算命中
        cache_dir = self.work_dir / pdf_path.stem
        extract_dirs = {}
        for part_path, _, _ in parts:
            # 未分片时解析结果直接在 cache_dir/{原文件stem}/ 下
            extract_dir = cache_dir / part_path.stem if len(parts) > 1 else cache_dir / pdf_path.stem
            if extract_dir.exists() and list(extract_dir.rglob("*_content_list.json")):
                extract_dirs[part_path.name] = extract_dir
        
        if len(extract_dirs) == len(parts):
            logger.info("检测到完整本地缓存（%d/%d 个分片），跳过上传/下载，直接切分",
                        len(extract_dirs), len(parts))
            chunks = self._parse_and_chunk(parts, extract_dirs)
        else:
            if extract_dirs:
                logger.info("本地缓存不完整（%d/%d 个分片），重新走完整流程",
                            len(extract_dirs), len(parts))
            
            # Pipeline 阶段 2：上传到 MinerU
            batch_id, part_paths = self._upload_to_mineru(parts)
            
            # Pipeline 阶段 3：下载解析结果
            extract_dirs = self._download_results(batch_id, pdf_path.stem, part_paths)
            
            # Pipeline 阶段 4：解析 + 切分 chunk
            chunks = self._parse_and_chunk(parts, extract_dirs)
        
        total_time = time.time() - total_start
        logger.info("完成处理: %s，生成 %d 个 chunk，总耗时 %.1f 秒", 
                   pdf_path.name, len(chunks), total_time)
        logger.info("=" * 60)
        return chunks
    
    def _prepare_files(self, pdf_path: Path) -> list[tuple[Path, int, int]]:
        """Pipeline 阶段 1：准备文件（超 200 页自动分片）。
        
        Returns:
            [(分片路径, 起始页, 结束页), ...]
        """
        start_time = time.time()
        logger.info("阶段 1 - 准备文件")
        
        parts = split_pdf(pdf_path, output_dir=self.work_dir / pdf_path.stem)
        if len(parts) > 1:
            logger.info("  PDF 超过 %d 页，已切分为 %d 个分片", MAX_PAGES, len(parts))
        else:
            logger.info("  PDF 共 %d 页，无需分片", parts[0][2])
        
        elapsed = time.time() - start_time
        logger.info("  耗时: %.1f 秒", elapsed)
        return parts
    
    def _upload_to_mineru(self, parts: list[tuple[Path, int, int]]) -> tuple[str, list[Path]]:
        """Pipeline 阶段 2：上传到 MinerU。
        
        Returns:
            (batch_id, 分片路径列表)
        """
        start_time = time.time()
        logger.info("阶段 2 - 上传到 MinerU")
        
        part_paths = [p for p, _, _ in parts]
        batch_id, upload_urls = create_batch(self.api_key, part_paths)
        logger.info("  batch_id: %s", batch_id)
        upload_files(part_paths, upload_urls)
        logger.info("  已上传 %d 个文件", len(part_paths))
        
        elapsed = time.time() - start_time
        logger.info("  耗时: %.1f 秒", elapsed)
        return batch_id, part_paths
    
    def _download_results(
        self,
        batch_id: str,
        pdf_stem: str,
        part_paths: list[Path]
    ) -> dict[str, Path]:
        """Pipeline 阶段 3：下载解析结果。
        
        Returns:
            {文件名: 解压目录}
        """
        start_time = time.time()
        logger.info("阶段 3 - 下载解析结果（含 MinerU 在线解析）")
        
        extract_dirs = download_results(self.api_key, batch_id, self.work_dir / pdf_stem)
        logger.info("  已下载并解压 %d 个结果", len(extract_dirs))
        
        elapsed = time.time() - start_time
        logger.info("  耗时: %.1f 秒（包含 MinerU 在线解析 + 轮询 + 下载）", elapsed)
        return extract_dirs
    
    def _repair_empty_tables(self, content_list: list[dict], content_list_path: Path) -> list[dict]:
        """空表体二次提取：单元格全空的表格用 qwen-vl 读图转录，结果落盘缓存。

        修复结果写到 {content_list 文件名}_fixed.json，重跑直接读缓存，不重复调用。
        """
        to_repair = [
            item for item in content_list
            if item.get("type") == "table"
            and table_body_is_empty(item.get("table_body"))
            and item.get("img_path")  # 无裁剪图片的空表无法转录，保留 caption/footnote
        ]
        if not to_repair:
            return content_list

        if not self.dashscope_api_key:
            raise RuntimeError(
                f"检测到 {len(to_repair)} 个空表体表格，"
                "需要 DASHSCOPE_API_KEY 调用 qwen-vl 二次提取（请在 .env 中填写）"
            )
        vl_url = os.environ.get("DASHSCOPE_VL_URL")
        if not vl_url:
            raise RuntimeError("未配置 DASHSCOPE_VL_URL（qwen-vl 兼容模式端点，请在 .env 中填写）")

        base_dir = content_list_path.parent
        for item in to_repair:
            img_path = base_dir / item["img_path"]
            logger.info("  空表体，qwen-vl 读图转录: %s", item["img_path"])
            item["table_body"] = transcribe_table_image(img_path, self.dashscope_api_key, vl_url)

        fixed_path = content_list_path.with_name(f"{content_list_path.stem}_fixed.json")
        fixed_path.write_text(json.dumps(content_list, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("  已修复 %d 个空表体，缓存: %s", len(to_repair), fixed_path.name)
        return content_list

    def _parse_and_chunk(
        self,
        parts: list[tuple[Path, int, int]],
        extract_dirs: dict[str, Path]
    ) -> list[dict]:
        """Pipeline 阶段 4：解析 content_list + 切分 chunk。
        
        Returns:
            chunk 列表
        """
        start_time = time.time()
        logger.info("阶段 4 - 解析 + 切分 chunk")
        chunks = []
        
        for part_path, start, _end in parts:
            extract_dir = extract_dirs.get(part_path.name)
            if extract_dir is None:
                raise RuntimeError(f"分片解析失败: {part_path.name}")
            
            # 读取 content_list.json（文件名可能带 UUID 前缀）
            content_list_files = list(extract_dir.rglob("*_content_list.json"))
            if not content_list_files:
                # 兼容旧版无前缀格式
                content_list_files = list(extract_dir.rglob("content_list.json"))
            
            if not content_list_files:
                raise RuntimeError(f"未找到 content_list.json: {extract_dir}")
            
            content_list_path = content_list_files[0]
            # 优先使用空表体修复后的缓存
            fixed_path = content_list_path.with_name(f"{content_list_path.stem}_fixed.json")
            if fixed_path.exists():
                content_list = json.loads(fixed_path.read_text(encoding="utf-8"))
                logger.info("  使用修复缓存: %s", fixed_path.name)
            else:
                content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
                content_list = self._repair_empty_tables(content_list, content_list_path)
            logger.info("  处理分片: %s（第 %d 页起，%d 个条目）",
                       part_path.name, start, len(content_list))
            
            # 切分 chunk（page_idx 是 0 起相对页，start 是 1 起绝对页 → 偏移 = start - 1）
            part_chunks = chunks_from_content_list(
                content_list,
                page_offset=start - 1,
                min_section_tokens=self.min_section_tokens,
                max_chunk_tokens=self.max_chunk_tokens,
                overlap_tokens=self.overlap_tokens,
            )
            chunks.extend(part_chunks)
            logger.info("  生成 %d 个 chunk", len(part_chunks))
        
        elapsed = time.time() - start_time
        logger.info("  耗时: %.1f 秒", elapsed)
        return chunks
