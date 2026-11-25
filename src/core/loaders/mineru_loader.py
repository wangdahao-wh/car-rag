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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from html.parser import HTMLParser
from pypdf import PdfReader, PdfWriter

from .base import PDFLoader

MAX_PAGES = 200          # MinerU 单文件页数上限
POLL_INTERVAL = 10       # 轮询间隔（秒）
MINERU_MODEL_VERSION = "pipeline"  # MinerU 解析模型：vlm 对图片型/符号型表格识别明显优于 pipeline
VL_MODEL = "qwen3-vl-plus"  # 空表体二次提取用的视觉模型（qwen-vl-ocr 会丢弃符号单元格，不能用）
VL_MAX_TOKENS = 16384      # 转录输出上限：整页大表 HTML 约 1.5 万 token（.env 可覆盖）
VL_RETRY_TIMES = 3         # qwen-vl 转录请求的传输层重试次数（服务端偶发断连）

# 表格行级序列化开关（.env: TABLE_ROW_SERIALIZATION=1 开启，默认关）
# 开启后 serialize_table 把 HTML 表体转成「配置项 | 列名: 值」的自解释文本，
# 用于和默认的原始 HTML 方案做检索效果对比
TABLE_ROW_SERIALIZATION = os.environ.get("TABLE_ROW_SERIALIZATION", "0") == "1"

# 伪标题黑名单：MinerU 会把页脚水印和手册警示标签误判为标题（text_level）。
# 两条判定规则（都不能误杀"低压蓄电池处理的安全警告"这类真标题）：
#   - 含 "carob"：水印 www.carobook.com 及其 OCR 截断/正文粘连变体，整条丢弃
#   - 纯警示标签（警告/注意/说明/危险，可带叹号或英文对照）：降级为正文行
FAKE_TITLE_TAG_PATTERN = re.compile(r"^(警告|注意|说明|危险)！?(WARNING|CAUTION|NOTE|DANGER)?$")

# 正文水印：vlm 把 www.carobook.com 当作普通正文输出，还带 OCR 截断/粘连变体
# （如 "www.car"、"www.ca"、"ww.carobook.co"、".carobo"、"arobook.com"）。
# 只用 ASCII 字符集匹配，避免把相邻中文一起吃掉；官网 www.lynkco.com 不受影响
WATERMARK_PATTERN = re.compile(
    r"w{1,3}\.ca[a-z0-9]{0,10}(?:\.co(?:m)?)?"
    r"|\.carob[a-z0-9]{0,10}(?:\.co(?:m)?)?"
    r"|[a-z]{0,4}obook\.co(?:m)?",
    re.IGNORECASE,
)

# 目录页条目特征：正文行里出现 >= 3 处「条目名. . .页码」导航模式即判定为目录，整条丢弃
# （目录条目对检索无价值，还会抢占 top-k 名额）
TOC_LINE_PATTERN = re.compile(r"\S+\s*\.\s*\.\s*\.\s*\d+")
TOC_MIN_LINES = 3

# 配置表符号 → 语义文字（单元格内容精确匹配整个符号才替换）
# 领克用 ●/○/-，BMW 用 ■/□（■=标配、□=选装，图例见配置表末页）；
# "一" 是 BMW 配置表里 "-"（不含）的 OCR 变体
_TABLE_SYMBOL_MAP = {
    "●": "标配",
    "○": "选装",
    "O": "选装",
    "o": "选装",
    "-": "不含",
    "—": "不含",
    "一": "不含",
    "■": "标配",
    "□": "选装",
}

# 复合值前缀符号替换：□¥1,500 → 选装 ¥1,500（选装价格）
_VALUE_PREFIX_MAP = {"□": "选装 ", "■": "标配 "}

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

# urllib3 自带的重试日志（Retrying (_RetryWithLog(...)) ...）与 _RetryWithLog 的中文日志重复
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


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


def create_batch(
    api_key: str,
    file_paths: list[Path],
    model_version: str = MINERU_MODEL_VERSION,
) -> tuple[str, list[str]]:
    """申请批量上传，返回 (batch_id, 预签名上传地址列表)。

    :param model_version: MinerU 模型版本，pipeline / vlm / MinerU-HTML，
        默认 vlm（配置表符号矩阵在 pipeline 下会输出空表体）
    """
    res = requests.post(
        "https://mineru.net/api/v4/file-urls/batch",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json={
            "enable_formula": False,
            "language": "ch",
            "model_version": model_version,
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
    """把 HTML 表格解析为行列表（colspan / rowspan 均展开）。

    self.rows: [[单元格文本, ...], ...]，展开 colspan / rowspan 保证各行列对齐。
    rowspan 覆盖的后续行在对应列填空串（HTML 视觉合并语义），
    使列号与实际视觉列一致，行级序列化时列名映射不会整体左移。
    """

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._colspan = 1
        self._rowspan = 1
        self._cell_col = 0                  # 当前单元格的视觉起始列
        self._col = 0                       # 当前行下一个可写入的视觉列
        self._pending: dict[int, int] = {}  # 列号 -> 仍被上方单元格占用的剩余行数
        self._sup_depth = 0                 # <sup> 嵌套深度，内的内容不进入单元格文本

    def _next_free_col(self, start: int) -> int:
        """跳过被上方 rowspan 占用的列。"""
        col = start
        while col in self._pending:
            col += 1
        return col

    def _ensure_col(self, col: int) -> None:
        """把当前行补齐到 col 列，被占用的位置留空串。"""
        while len(self._row) < col:
            self._row.append("")

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
            self._col = 0
        elif tag in ("td", "th") and self._row is not None:
            attr = dict(attrs)
            self._cell = []
            self._cell_col = self._next_free_col(self._col)
            self._colspan = max(1, int(attr.get("colspan") or 1))
            self._rowspan = max(1, int(attr.get("rowspan") or 1))
            self._col = self._cell_col + self._colspan
            if self._rowspan > 1:
                # 含当前行共占用 rowspan 行，行结束时会递减一次
                for c in range(self._cell_col, self._cell_col + self._colspan):
                    self._pending[c] = self._rowspan
        elif tag == "br" and self._cell is not None:
            # <br> 换行转为空格，避免相邻文本粘连（如 xDrive 25i<br>M运动套装）
            self._cell.append(" ")
        elif tag == "sup":
            # 上标是脚注标号（250<sup>1</sup> 表示"250，见脚注 1"）。
            # 直接丢弃，否则会拼成 2501，把数值改成完全不同的数字
            self._sup_depth += 1

    def handle_endtag(self, tag):
        if tag == "tr" and self._row is not None:
            # 行尾仍被上方 rowspan 占用的列也要补空串
            if self._pending:
                self._ensure_col(max(self._pending) + 1)
            self.rows.append(self._row)
            # 本行结束，消费一行占位
            for col in list(self._pending):
                self._pending[col] -= 1
                if self._pending[col] <= 0:
                    del self._pending[col]
            self._row = None
        elif tag in ("td", "th") and self._cell is not None:
            text = "".join(self._cell).strip()
            self._ensure_col(self._cell_col)
            self._row.extend([text] * self._colspan)
            self._cell = None
        elif tag == "sup" and self._sup_depth > 0:
            self._sup_depth -= 1

    def handle_data(self, data):
        if self._cell is not None and self._sup_depth == 0:
            self._cell.append(data)


def doc_display_name(pdf_path: Path) -> str:
    """从 PDF 文件名生成文档显示名（行级 chunk 面包屑用）。

    文件名格式 {品牌}_{年份}_{车型}_{文档类型}：
        领克_2027_03_配置表   → 领克03 配置表
        BMW_2024_X3_配置表    → BMW X3 配置表
    """
    parts = pdf_path.stem.split("_")
    if len(parts) >= 4:
        company, model, doc_type = parts[0], parts[2], parts[3]
        # ASCII 品牌（BMW）和车型之间加空格，中文品牌（领克）直接拼接
        sep = " " if company.isascii() else ""
        return f"{company}{sep}{model} {doc_type}"
    return pdf_path.stem


def _pad_row(row: list[str], col_count: int) -> list[str]:
    """行内单元格不足列数时补空。"""
    return [row[i] if i < len(row) else "" for i in range(col_count)]


def _is_group_row(cells: list[str]) -> bool:
    """分组行判定。

    两种形态：
        1. colspan 跨全列的分组标题：展开后整行同值（安全性 安全性 安全性 安全性）
        2. 独立分组行：首列有值、其余全空
    首列为空的行不算分组（避免把首列丢失的数据行误判成分组）。
    """
    non_empty = [c for c in cells if c]
    if not non_empty:
        return False
    if cells[0] and len(set(non_empty)) == 1:
        return True
    return bool(cells[0]) and not any(cells[1:])


def _map_value(cell: str) -> str:
    """单元格值 → 语义文字：整格符号替换 + 前缀符号替换（□¥1,500 → 选装 ¥1,500）。"""
    val = _TABLE_SYMBOL_MAP.get(cell, cell)
    if val and val[0] in _VALUE_PREFIX_MAP:
        val = _VALUE_PREFIX_MAP[val[0]] + val[1:].lstrip()
    return val


def serialize_table_rows(
    html: str,
    fallback_col_names: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """表格行级序列化：每行解析为结构化条目。

    纯本地解析（html.parser 标准库），不调用任何模型，零 token 消耗。
    规则：
        1. 表头区：从顶部扫描，遇到分组行、或第 2 行起首列有值的行即停
           （覆盖单级/两级表头、无表头续表；跨页续表会把分组标签和列名
           合并在首行，首列的标签转为初始 category）
        2. 分组行（整行同值 / 首列有值其余空）更新当前 category，不生成条目
        3. 符号单元格替换为语义文字（●/■→标配、○/□→选装、-→不含；
           前缀符号如 □¥1,500 → 选装 ¥1,500）
        4. colspan 展开后，连续同值列合并（列A/列B=值）
        5. 无表头的续页表（整表以分组行开头）：列名继承同页上一张表，
           避免输出退化成「5.3；4.8」这种分不清车型的纯值

    :param fallback_col_names: 同页上一张表的列名，仅在本身无表头且
        列数一致时使用

    Returns:
        (entries, col_names)：
            entries = [{"category": 分组名, "param": 配置项,
                        "values": [(列名, 值), ...]}, ...]，空表体返回 []
            col_names = 本表解析出的列名（继承时即 fallback），供下一张表使用
    """
    parser = _TableRowsParser()
    parser.feed(html)
    # 丢弃整行为空的行
    rows = [r for r in parser.rows if any(r)]
    if not rows:
        return [], []

    col_count = max(len(r) for r in rows)
    padded = [_pad_row(r, col_count) for r in rows]

    # 表头边界：遇分组行、或第 2 行起首列有值的行（数据行）即停。
    # 首行允许首列有值——跨页续表的首行是「分组标签 + 列名」合并行
    header_end = 0
    for i, r in enumerate(padded):
        if _is_group_row(r):
            break
        if i > 0 and r[0]:
            break
        header_end = i + 1
    header = padded[:header_end]

    # 跨页续表：表头首行首列是上页带入的分组标签（内装/BMW技术）
    category = header[0][0] if header and header[0][0] else ""

    if header_end == 0 and fallback_col_names and len(fallback_col_names) == col_count:
        # 本表以分组行开头，没有表头：沿用同页上一张表的列名
        col_names = list(fallback_col_names)
    else:
        # 逐列拼接多级表头：某一级为空时只取非空的层级
        col_names = []
        for i in range(col_count):
            parts = [h[i] for h in header if h[i]]
            if len(parts) > 1:
                col_names.append("(".join(parts) + ")")
            else:
                col_names.append(parts[0] if parts else "")

    entries = []
    for cells in padded[header_end:]:
        if _is_group_row(cells):
            category = cells[0]
            continue
        # 数据行：首列是配置项名，其余列合并连续同值
        param = cells[0]
        values = []
        i = 1
        while i < col_count:
            val = _map_value(cells[i])
            j = i + 1
            while j < col_count and _map_value(cells[j]) == val:
                j += 1
            # 连续同名列去重：colspan 展开会让同一车型相邻多列同名
            names = []
            for n in col_names[i:j]:
                if n and (not names or names[-1] != n):
                    names.append(n)
            cols = "/".join(names)
            if val:
                values.append((cols, val))
            i = j
        if param and values:
            entries.append({"category": category, "param": param, "values": values})
    return entries, col_names


def serialize_table(item: dict) -> str:
    """序列化表格条目（整表模式：caption + 原始 HTML + footnote）。

    MinerU schema：table_caption / table_footnote 是字符串列表，table_body 是 HTML 字符串。
    行级序列化（TABLE_ROW_SERIALIZATION）不经过这里，见 _table_row_chunks。
    """
    caption = "\n".join(item.get("table_caption") or [])
    body = item.get("table_body") or ""
    footnote = "\n".join(item.get("table_footnote") or [])
    return "\n".join(p for p in (caption, body, footnote) if p)


def serialize_chart(item: dict) -> str:
    """序列化图表条目（只取标题，内容通常为空）。"""
    return "\n".join(item.get("chart_caption") or [])


def serialize_image(item: dict) -> str:
    """序列化图片条目（只取标题）。"""
    return "\n".join(item.get("img_caption") or [])


# 网格挤压特征：单元格文本内出现 >= 3 个连续空格分隔的多个非空片段。
# 正常表格一列一个 td，单元格内容（中文/数字/单空格英文短语）不会有连续空格；
# MinerU 结构识别失败时会把多个单元格的文字挤进同一个 td，
# 列名之间保留 PDF 原始列间距的空格串，列网格整个丢失
GRID_SQUASH_PATTERN = re.compile(r"\S\s{3,}\S")


def table_body_is_usable(body: str | None) -> bool:
    """判断表格 HTML 是否可用（结构完好，可供行级序列化）。

    不可用的两种形态，统一路由到 qwen-vl 读图重转录：
        1. 空表体：单元格全部无内容（MinerU 完全没读出，如符号矩阵）
        2. 网格挤压：多个单元格的文字挤进同一个 td（连续空格分隔），
           列网格丢失，行级序列化无法对齐列（图片型 PDF 常见）
    """
    if not body:
        return False
    cells = re.findall(r"<td[^>]*>([^<]*)</td>", body)
    if not any(c.strip() for c in cells):
        return False  # 空表体
    if any(GRID_SQUASH_PATTERN.search(c) for c in cells):
        return False  # 网格挤压
    return True


class _RetryWithLog(Retry):
    """重试时输出日志：同一张图偶发 60s 断连，需要能看到重试过程。"""

    def increment(self, method=None, url=None, response=None, error=None, _pool=None, _stacktrace=None):
        if error is not None:
            if self.total <= 0:
                logger.error("qwen-vl 转录请求重试耗尽（%s: %s）",
                             type(error).__name__, str(error)[:120])
            else:
                logger.warning("qwen-vl 转录请求异常（%s: %s），第 %d/%d 次重试",
                               type(error).__name__, str(error)[:120],
                               VL_RETRY_TIMES - self.total + 1, VL_RETRY_TIMES)
        return super().increment(method, url, response, error, _pool, _stacktrace)


def transcribe_table_image(image_path: Path, api_key: str, vl_url: str) -> str:
    """调用 qwen-vl 把表格图片转录为 HTML 表格（表体不可用二次提取）。

    MinerU 对图片型/符号型表格的两种失败形态（空表体/网格挤压），
    表格内容只在裁剪图片里，需要视觉模型重新读图转录。
    """
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    # dashscope 为国内服务，直连即可：requests 默认会读终端的 HTTP_PROXY/
    # HTTPS_PROXY，代理进程没起时会抛 ProxyError，禁用环境代理避免干扰
    session = requests.Session()
    session.trust_env = False
    # 实测同一张图偶发 RemoteDisconnected（连接建立后无响应被断开），
    # 重试 3 次并指数退避；这是传输层重试，与表格内容无关。
    # allowed_methods 必须显式放开 POST（urllib3 默认只重试幂等方法，
    # 否则 read error 会直接抛出，一次都不重试）
    retry = _RetryWithLog(total=VL_RETRY_TIMES, connect=VL_RETRY_TIMES, read=VL_RETRY_TIMES,
                          backoff_factor=2, status_forcelist=[500, 502, 503, 504],
                          allowed_methods=frozenset(["POST"]))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    res = session.post(
        vl_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": VL_MODEL,
            # 整页大表（14 列 × 80+ 行）的 HTML 很长，默认输出上限会在表格
            # 中途截断（实测停在「舒适与安防」分组），显式放大。
            # 仍不够时按图片上下切半转录（见 transcribe_table_image）
            "max_tokens": int(os.environ.get("VL_MAX_TOKENS", VL_MAX_TOKENS)),
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


def recursive_split(text: str, max_tokens: int = 500, overlap: int = 50) -> list[str]:
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
    min_section_tokens: int = 80,
    max_chunk_tokens: int = 500,
    overlap_tokens: int = 50,
    doc_name: str = "",
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

    行级序列化（TABLE_ROW_SERIALIZATION=1）：
        表格行不进上面的 pipeline，每行独立成 chunk（见 _table_row_chunks），
        行文本自带 doc_name/category/param 面包屑，metadata 同步存这三个字段，
        支持点查（向量命中）和枚举查询（category 过滤全量取回）。

    :param content_list: MinerU 解析的内容列表
    :param page_offset: 绝对页偏移：分片第一页的 1 起页码，
        chunk 页码 = page_offset + page_idx，与 pypdf 的 1-based 保持一致
    :param min_section_tokens: 小节最小 token 数，默认 80
    :param max_chunk_tokens: chunk 最大 token 数，默认 500
    :param overlap_tokens: 细分时重叠 token 数，默认 50
    :param doc_name: 文档显示名（行级 chunk 面包屑用，如 领克03 配置表）
    """
    # 行级序列化：表格行独立成 chunk（点查友好）
    row_chunks = _table_row_chunks(content_list, page_offset, doc_name) if TABLE_ROW_SERIALIZATION else []

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

    chunks.extend(row_chunks)
    return chunks


def _table_row_chunks(content_list: list[dict], page_offset: int, doc_name: str) -> list[dict]:
    """行级序列化：表格每行生成一个独立 chunk。

    行文本自带面包屑「doc_name > category > param」——metadata 不参与
    向量相似度计算，品牌/车型/类别/配置项必须写进 text 才能被
    embedding 命中（如查询「领克03 2.0T 有没有定速巡航」）。
    metadata 保存 doc_name/category/param 供检索侧过滤
    （如「领克03 有哪些智能驾驶功能」→ category=智能驾控 全量取回）。

    跨表继承：MinerU 会把跨页的配置表切成多张，续页那张以分组行
    开头、没有车型列名。只要两张表之间没有正文/标题条目，就把
    上一张表解析出的表头传给下一张（配置项示意图、页脚注释不打断），
    否则续页会输出「0-100公里/小时(秒)：5.3；4.8」这种分不清车型的纯值。
    """
    chunks = []
    last_col_names: list[str] = []
    for item in content_list:
        if item.get("type") != "table":
            # 只有正文/标题才断开继承；image / page_footnote 不影响
            # （X3 配置表 p3 有表头、p5 续页无表头，中间只隔着示意图）
            if (item.get("text") or "").strip():
                last_col_names = []
            continue
        body = item.get("table_body") or ""
        if not body:
            continue
        page = page_offset + item.get("page_idx", 0)

        rows, col_names = serialize_table_rows(body, last_col_names or None)
        if col_names:
            last_col_names = col_names

        for row in rows:
            # 面包屑只拼非空部分（手册表格可能没有分组行，category 为空）
            crumbs = " > ".join(p for p in (doc_name, row["category"], row["param"]) if p)
            values = "；".join(f"{cols}={val}" if cols else val for cols, val in row["values"])
            chunks.append({
                "text": f"{crumbs}：{values}",
                "page": page,
                "page_end": page,
                "chapter": "",
                "section": "",
                "path": [],
                "loader": "mineru",
                "doc_name": doc_name,
                "category": row["category"],
                "param": row["param"],
            })
    return chunks


def strip_watermark(text: str) -> str:
    """剥离正文中的 carobook 水印及其残缺变体。

    只做字符串替换，不含水印的文本原样返回；
    整条只有水印时返回空串，由调用方丢弃。
    """
    if not WATERMARK_PATTERN.search(text):
        return text
    return WATERMARK_PATTERN.sub("", text).strip()


def _parse_items_to_sections(content_list: list[dict], page_offset: int) -> list[dict]:
    """Pipeline 阶段 1：解析条目，按标题分节 + 序列化各类型内容。
    
    Returns:
        [{"titles": [(level, text)], "lines": [str], "page": int}, ...]
    """
    sections = []
    titles = []    # 标题栈
    lines = []
    section_page = None
    section_page_end = None  # 本节最后一条内容所在页（含），用于页码范围匹配

    def flush_section():
        """收集当前小节。"""
        nonlocal lines, section_page, section_page_end
        if not lines:
            return
        sections.append({
            "titles": titles.copy(),
            "lines": lines.copy(),
            "page": section_page,
            "page_end": section_page_end,
        })
        lines = []
        section_page = None
        section_page_end = None

    for item in content_list:
        item_type = item.get("type", "")
        page_idx = page_offset + item.get("page_idx", 0)
        
        # 跳过页眉页脚
        if item_type in ("header", "footer"):
            continue
        
        # 标题：先收上一节，再维护标题栈
        if item_type == "text" and item.get("text_level"):
            # 剥离 MinerU 的上下标 HTML 标签，标题树保持干净
            title_text = re.sub(r"</?(?:sub|sup)>", "", item["text"]).strip()
            # 水印及粘连变体：剥离后为空则整条丢弃（真标题不可能只剩水印）
            title_text = strip_watermark(title_text)
            if not title_text:
                continue
            # 纯警示标签：不开新节，降级为正文行留在当前节
            if FAKE_TITLE_TAG_PATTERN.match(title_text):
                if section_page is None:
                    section_page = page_idx
                section_page_end = page_idx
                lines.append(title_text)
                continue
            flush_section()
            level = item["text_level"]
            titles = [t for t in titles if t[0] < level] + [(level, title_text)]
            continue
        
        # 序列化各类型内容
        content = ""
        if item_type == "text":
            content = item.get("text", "").strip()
            # 水印剥离：整条只有水印的条目清洗后为空，直接丢弃
            content = strip_watermark(content)
            if not content:
                continue
            # 目录页判定：一条文本里大量「条目. . .页码」导航行 → 整条丢弃
            if len(TOC_LINE_PATTERN.findall(content)) >= TOC_MIN_LINES:
                continue
        elif item_type == "table":
            # 行级模式：表格行已由 _table_row_chunks 独立处理，整表跳过；
            # 行解析失败的表格（返回空）仍走整表序列化，避免内容丢失
            if TABLE_ROW_SERIALIZATION and serialize_table_rows(item.get("table_body") or "")[0]:
                continue
            content = serialize_table(item)
        elif item_type == "chart":
            content = serialize_chart(item)
        elif item_type == "image":
            content = serialize_image(item)
        
        if content:
            if section_page is None:
                section_page = page_idx
            section_page_end = page_idx
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
                sec["page_end"] = next_sec.get("page_end") or sec.get("page_end")
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
                    "page_end": sec.get("page_end"),
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
            "page_end": sec.get("page_end") or sec["page"],
            "chapter": path[0] if len(path) > 0 else "",
            "section": path[1] if len(path) > 1 else "",
            "path": path,
            "loader": "mineru",  # 切分来源标记（auto 模式下区分路由结果）
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
        doc_name = doc_display_name(pdf_path)  # 行级 chunk 面包屑用（如 领克03 配置表）
        # 配置表全部表格重转录（文件名字段：{品牌}_{年份}_{车型}_{文档类型}）
        name_parts = pdf_path.stem.split("_")
        is_config = len(name_parts) >= 4 and name_parts[3] == "配置表"

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
            chunks = self._parse_and_chunk(parts, extract_dirs, doc_name, is_config)
        else:
            if extract_dirs:
                logger.info("本地缓存不完整（%d/%d 个分片），重新走完整流程",
                            len(extract_dirs), len(parts))

            # Pipeline 阶段 2：上传到 MinerU
            batch_id, part_paths = self._upload_to_mineru(parts)

            # Pipeline 阶段 3：下载解析结果
            extract_dirs = self._download_results(batch_id, pdf_path.stem, part_paths)

            # Pipeline 阶段 4：解析 + 切分 chunk
            chunks = self._parse_and_chunk(parts, extract_dirs, doc_name, is_config)
        
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
    
    def _repair_broken_tables(
        self,
        content_list: list[dict],
        content_list_path: Path,
        transcribe_all: bool = False,
    ) -> list[dict]:
        """表格二次提取：用 qwen-vl 读图转录，结果落盘缓存。

        :param transcribe_all: 配置表模式为 True，全部表格重转录。MinerU vlm 对
            小字号密集表格的字词识别错误率高（如「智能驾控」→「智能领投」、
            「AQS」→「AQ25」），而枚举检索完全依赖 param/category 的精确文本，
            错一个字就取不回。为 False 时只处理坏表（空表体 / 网格挤压）。

        修复结果写到 {content_list 文件名}_fixed_all.json（全量）或
        _fixed.json（仅坏表），两种模式互不覆盖，重跑直接读缓存。
        """
        if transcribe_all:
            to_repair = [
                item for item in content_list
                if item.get("type") == "table" and item.get("img_path")
            ]
        else:
            to_repair = [
                item for item in content_list
                if item.get("type") == "table"
                and not table_body_is_usable(item.get("table_body"))
                and item.get("img_path")  # 无裁剪图片的坏表无法转录，保留 caption/footnote
            ]
        if not to_repair:
            return content_list

        if not self.dashscope_api_key:
            raise RuntimeError(
                f"检测到 {len(to_repair)} 个表体不可用的表格（空表体/网格挤压），"
                "需要 DASHSCOPE_API_KEY 调用 qwen-vl 二次提取（请在 .env 中填写）"
            )
        vl_url = os.environ.get("DASHSCOPE_VL_URL")
        if not vl_url:
            raise RuntimeError("未配置 DASHSCOPE_VL_URL（qwen-vl 兼容模式端点，请在 .env 中填写）")

        base_dir = content_list_path.parent
        for item in to_repair:
            img_path = base_dir / item["img_path"]
            logger.info("  qwen-vl 读图转录: %s", item["img_path"])
            item["table_body"] = transcribe_table_image(img_path, self.dashscope_api_key, vl_url)

        suffix = "_fixed_all" if transcribe_all else "_fixed"
        fixed_path = content_list_path.with_name(f"{content_list_path.stem}{suffix}.json")
        fixed_path.write_text(json.dumps(content_list, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("  已转录 %d 个表格，缓存: %s", len(to_repair), fixed_path.name)
        return content_list

    def _parse_and_chunk(
        self,
        parts: list[tuple[Path, int, int]],
        extract_dirs: dict[str, Path],
        doc_name: str = "",
        is_config: bool = False,
    ) -> list[dict]:
        """Pipeline 阶段 4：解析 content_list + 切分 chunk。

        Args:
            parts: [(分片路径, 起始页, 结束页), ...]，页码均为 1 起绝对值
            extract_dirs: {分片文件名: 解压目录}
            doc_name: 文档显示名（行级 chunk 面包屑用）
            is_config: 是否配置表。True 时全部表格走 qwen-vl 重转录
                （枚举检索依赖精确的 category/param 文本）

        Returns:
            chunk 列表，页码 = 分片起始页 + content_list 的 0 起 page_idx，
            与 pypdf 加载器的 1-based 页码一致
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
            # 优先使用已转录的缓存（配置表读 _fixed_all，手册读 _fixed）
            suffix = "_fixed_all" if is_config else "_fixed"
            fixed_path = content_list_path.with_name(f"{content_list_path.stem}{suffix}.json")
            if fixed_path.exists():
                content_list = json.loads(fixed_path.read_text(encoding="utf-8"))
                logger.info("  使用转录缓存: %s", fixed_path.name)
            else:
                content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
                content_list = self._repair_broken_tables(
                    content_list, content_list_path, transcribe_all=is_config,
                )
            logger.info("  处理分片: %s（第 %d 页起，%d 个条目）",
                       part_path.name, start, len(content_list))
            
            # 切分 chunk（page_idx 是 0 起相对页，start 是 1 起绝对页 → 偏移 = start）
            part_chunks = chunks_from_content_list(
                content_list,
                page_offset=start,
                min_section_tokens=self.min_section_tokens,
                max_chunk_tokens=self.max_chunk_tokens,
                overlap_tokens=self.overlap_tokens,
                doc_name=doc_name,
            )
            chunks.extend(part_chunks)
            logger.info("  生成 %d 个 chunk", len(part_chunks))
        
        elapsed = time.time() - start_time
        logger.info("  耗时: %.1f 秒", elapsed)
        return chunks
