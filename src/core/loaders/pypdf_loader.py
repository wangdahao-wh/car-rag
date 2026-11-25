"""pypdf 加载器：优先使用 PDF 书签目录（outline）做结构化切分。

书签目录是出版社写入 PDF 元数据的权威标题树，包含标题文字和起始页码，
比基于字号的启发式推断更准确（避免把页眉/书眉/提示语当成标题）。

路由判据（由 loaders/__init__.py 的 AutoLoader 调用 detect_outline 决定）：
    有书签且层级数 >= 2  → load_from_outline（快速本地解析）
    其他                 → MinerU（在线 OCR，如配置表、扁平书签手册、扫描件）

切分流程（锚点模型）：
    1. 读取书签树，扁平化成 [(depth, title, page), ...]
    2. 标记噪声条目（车辆装备 / 便捷查阅分支）：不生成 chunk，但仍作为锚点
    3. 每个条目在起始页文本中定位标题位置，得到锚点 (页码, 页内偏移)
    4. section 内容 = 从自己的锚点到下一个条目的锚点（同页也能精确切开）
    5. 第一个锚点之前的内容归入隐式「前言」section（书签是切分点而非覆盖范围，
       封面/前言/目录/书签未收录的章节不应丢弃）
    6. 页眉页脚裁剪（Seite XXX / Online Edition / 版权行）
    7. 长 section 按 500 token 递归切分
"""
import json
import logging
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from pypdf import PdfReader

from .base import PDFLoader

MAX_CHUNK_TOKENS = 500  # 单个 chunk 最大 token 数（与 MinerU 的 MAX_TOKENS 对齐）
MIN_SECTION_TOKENS = 80  # section 少于此 token 时合并到前一个 section

# 噪声书签标题：整条跳过
SKIP_TITLES = {
    "车辆装备",          # 每个 L2 下第一个 L3，内容是重复的免责声明 + 装备索引表
}

# 噪声 L1 章节：整棵子树跳过
SKIP_L1_TITLES = {
    "便捷查阅",          # 技术数据 / 附录 / 从 A 至 Z 都是手册末尾的参考信息
}

# 页眉页脚裁剪规则
FOOTER_PATTERNS = [
    re.compile(r"^Seite\s+\d+\s*$"),                       # BMW 手册页脚 Seite 100
    re.compile(r"^Online Edition for Part no\."),          # BMW 手册页脚 Online Edition for Part no. 01405A...
    re.compile(r"^\d{8}.*zh-cn\s*$"),                     # 版本号 01405A89DE4 zh-cn
    re.compile(r"^©\s+\d{4}\s+Bayerische"),               # 版权页行
]

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """估算 token 数（中文按字符数，英文按空格分词数）。"""
    chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
    english = len(re.findall(r'\b\w+\b', text))
    return chinese + english


def detect_outline(pdf_path: Path) -> bool:
    """检测 PDF 是否有可用的书签目录。

    返回 True 当且仅当：
        - 书签条目数 >= 10（不是空的或只有几个条目）
        - 层级数 >= 2（扁平书签如领克 03 用户手册全 L1，不值得用 outline）
    """
    reader = PdfReader(str(pdf_path))
    if not reader.outline:
        return False

    depth_counter = Counter()
    entry_count = 0

    def walk(items, depth):
        nonlocal entry_count
        for it in items:
            if isinstance(it, list):
                walk(it, depth + 1)
            else:
                entry_count += 1
                depth_counter[depth] += 1

    walk(reader.outline, 1)
    # 约束：目录条目个数 和 目录深度
    if entry_count < 10:
        return False
    if len(depth_counter) < 2:
        return False
    return True


def flatten_outline(reader: PdfReader) -> list[tuple[int, str, int]]:
    """把 PDF 书签树扁平化成 [(depth, title, page), ...]。

    页码从 1 开始（pypdf 的 get_destination_page_number 返回 0-based）。
    """
    result = []

    def walk(items, depth):
        for it in items:
            if isinstance(it, list):
                walk(it, depth + 1)
            else:
                try:
                    page = reader.get_destination_page_number(it) + 1
                except Exception:
                    page = -1
                result.append((depth, str(it.title).strip(), page))

    walk(reader.outline, 1)
    return result


def should_skip_l1(title: str) -> bool:
    """判断 L1 标题是否整棵子树跳过。"""
    return title in SKIP_L1_TITLES


def build_section_entries(outline: list[tuple[int, str, int]]) -> list[tuple[list[str], str, int, bool]]:
    """把扁平化的 outline 转成 [(完整路径, 标题, 起始页, 是否保留), ...]。

    keep=False 的噪声条目（车辆装备 / 便捷查阅分支）不生成 chunk，
    但仍保留在列表里——它们参与锚点定位，把相邻 section 的边界
    切在正确位置（否则同页 section 会丢内容或重复，末尾 section
    会吃掉被跳过章节的页面如 A-Z 索引）。
    """
    result = []
    stack: dict[int, str] = {}  # depth -> title
    skip_l1_root = None  # 当前跳过的 L1 根标题

    for depth, title, page in outline:
        # 清理更深层的栈
        for d in list(stack.keys()):
            if d >= depth:
                del stack[d]

        # 判断是否保留：便捷查阅整棵子树跳过；车辆装备单条跳过
        if depth == 1 and should_skip_l1(title):
            skip_l1_root = title
            keep = False
        elif skip_l1_root is not None and depth > 1:
            keep = False
        else:
            if depth == 1:
                skip_l1_root = None  # 新的 L1，退出跳过分支
            keep = title not in SKIP_TITLES

        if page < 1:
            continue  # 无效页码的条目直接丢弃（也不作为锚点）

        if not keep:
            # 噪声条目路径留空，下游整节丢弃；但保留在列表里作为锚点，
            # 让下一个 section 从「警示闪光灯」等标题处开始切
            result.append(([], title, page, False))
            continue

        stack[depth] = title

        # 构建完整路径（按 depth 顺序）
        full = []
        for d in sorted(stack):
            if stack[d]:
                full.append(stack[d])

        # 相邻同名去重（如 X3 的 L1 和 L2 都叫 "提示"）
        deduped = []
        for t in full:
            if not deduped or deduped[-1] != t:
                deduped.append(t)

        result.append((deduped, title, page, True))

    return result


def find_title_pos(page_text: str, title: str, search_from: int = 0) -> int:
    """在页面文本中定位标题的起始位置（页内锚点）。

    先原文匹配，失败再忽略空白差异匹配（标题可能被 PDF 换行拆开）。
    从 search_from 开始找，保证同页多个锚点位置单调递增。
    找不到时返回 search_from（退化为从前一个锚点处切分）。
    """
    pos = page_text.find(title, search_from)
    if pos >= 0:
        return pos

    # 空白归一化匹配：去掉空白后查找，再映射回原文下标
    stripped_chars = []
    index_map = []
    for i, ch in enumerate(page_text):
        if not ch.isspace():
            stripped_chars.append(ch)
            index_map.append(i)
    stripped = "".join(stripped_chars)
    key = "".join(ch for ch in title if not ch.isspace())
    if not key:
        return search_from
    start = 0
    while True:
        idx = stripped.find(key, start)
        if idx < 0:
            break
        orig = index_map[idx]
        if orig >= search_from:
            return orig
        start = idx + 1
    return search_from


def strip_headers_footers(text: str) -> str:
    """裁剪页眉页脚（BMW 手册的 Seite XXX / Online Edition / 版权行）。"""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        if any(p.match(stripped) for p in FOOTER_PATTERNS):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def split_long_section(path: list[str], text: str, page: int, page_end: int) -> list[dict]:
    """长 section 按 500 token 递归切分，每个 chunk 都带完整标题路径。"""
    tokens = estimate_tokens(text)
    if tokens <= MAX_CHUNK_TOKENS:
        return [_make_chunk(path, text, page, page_end)]

    # 按句子边界切分，每片 <= MAX_CHUNK_TOKENS
    sentences = re.split(r"(?<=[。！？；\n])", text)
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_tokens = estimate_tokens(sent)
        if sent_tokens > MAX_CHUNK_TOKENS:
            # 单段本身超上限且内部无句子边界（如整行无标点的目录导航）：
            # 按字符数硬切。estimate_tokens(text) <= len(text) 恒成立，
            # 按 MAX_CHUNK_TOKENS 字符切保证每片不超限
            if current:
                chunks.append(_make_chunk(path, "".join(current), page, page_end))
                current, current_tokens = [], 0
            for k in range(0, len(sent), MAX_CHUNK_TOKENS):
                piece = sent[k:k + MAX_CHUNK_TOKENS].strip()
                if piece:
                    chunks.append(_make_chunk(path, piece, page, page_end))
            continue
        if current_tokens + sent_tokens > MAX_CHUNK_TOKENS and current:
            chunks.append(_make_chunk(path, "".join(current), page, page_end))
            current = []
            current_tokens = 0
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(_make_chunk(path, "".join(current), page, page_end))

    return chunks


def _make_chunk(path: list[str], body: str, page: int, page_end: int) -> dict:
    """组装 chunk 字典。

    page_end 是 section 覆盖的最后一页（含）：跨页 section 的 chunk 从 page 跨到
    page_end，评测按页码范围匹配，避免只认起始页导致中间页的答案被判未命中。
    """
    path_str = " > ".join(path)
    text = f"[{path_str}]\n{body}" if path_str else body
    return {
        "text": text,
        "page": page,
        "page_end": page_end,
        "chapter": path[0] if len(path) > 0 else "",
        "section": path[1] if len(path) > 1 else "",
        "path": list(path),
        "loader": "pypdf",
    }


def merge_short_sections(sections: list[tuple[list[str], str, int, str]]) -> list[tuple[list[str], str, int, str]]:
    """把过短的 section 合并到前一个同父级 section。

    输入: [(path, start_page, end_page, text), ...]
    输出: 过滤过短 section 后，文本并入前一个。

    父级不同不合并：否则故障救援章下的「警示闪光灯」会被并进上一章的
    「保险丝」，标题路径与正文串味。
    """
    if not sections:
        return sections

    merged = []
    for path, start_p, end_p, text in sections:
        tokens = estimate_tokens(text)
        prev = merged[-1] if merged else None
        same_parent = (
            prev is not None
            and len(path) == len(prev[0])
            and path[:-1] == prev[0][:-1]
        )
        if tokens < MIN_SECTION_TOKENS and same_parent:
            prev_path, prev_sp, _, prev_text = prev
            merged[-1] = (
                prev_path,
                prev_sp,
                end_p,
                prev_text + "\n" + text,
            )
        else:
            merged.append((path, start_p, end_p, text))
    return merged


class PyPDFLoader(PDFLoader):
    """pypdf 加载器：书签目录优先，按 PDF 内置标题树切分。"""

    def __init__(self, raw_dir: Path | None = None):
        """
        :param raw_dir: 原始抽取数据落盘目录，
            默认 <项目根>/processed_docs/pypdf/raw/（与 mineru/raw 对齐）
        """
        self.raw_dir = raw_dir or Path(__file__).resolve().parents[3] / "processed_docs" / "pypdf" / "raw"

    def load(self, pdf_path: Path) -> list[dict]:
        """加载 PDF，优先使用 outline 切分。

        - 有可用 outline → load_from_outline
        - 无可用 outline → 抛异常（由 AutoLoader 降级到 MinerU）
        """
        pdf_path = Path(pdf_path)

        if detect_outline(pdf_path):
            logger.info(f"{pdf_path.name}: 检测到书签目录 → 使用 outline 切分")
            return self.load_from_outline(pdf_path)
        else:
            raise RuntimeError(
                f"{pdf_path.name} 没有可用的书签目录。"
                f"请使用 MinerU: PDF_LOADER=mineru python src/core/build_chunks_new.py"
            )

    def load_from_outline(self, pdf_path: Path) -> list[dict]:
        """按 PDF 书签目录切分（锚点模型）。"""
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)

        # 1. 扁平化书签树 flatten：压平 动词
        outline = flatten_outline(reader)

        # 2. 标记噪声条目（不生成 chunk，但仍作为锚点参与边界切分）
        entries = build_section_entries(outline)
        kept = [e for e in entries if e[3]]
        if not kept:
            logger.warning(f"{pdf_path.name}: 过滤后无有效书签条目")
            return []
        logger.info(f"{pdf_path.name}: 原始书签 {len(outline)} 条，保留 {len(kept)} 个 section（跳过 {len(entries) - len(kept)} 条噪声）")

        # 3. 页文本缓存（每页只提取一次，裁剪页眉页脚）
        page_cache: dict[int, str] = {}

        def page_text(p: int) -> str:
            if p not in page_cache:
                try:
                    page_cache[p] = strip_headers_footers(reader.pages[p - 1].extract_text() or "")
                except Exception:
                    page_cache[p] = ""
            return page_cache[p]

        # 4. 计算全部条目的锚点 (页码, 页内偏移)。
        #    同页多条目从上一个锚点之后开始找，保证位置单调递增
        anchors = []
        last_pos_on_page: dict[int, int] = {}
        for _, title, page, _ in entries:
            text = page_text(page)
            search_from = last_pos_on_page.get(page, 0)
            pos = find_title_pos(text, title, search_from)
            anchors.append((page, pos))
            last_pos_on_page[page] = pos

        # 5. section 内容 = 从自己的锚点到下一个条目的锚点
        #    （下一个条目无论保留与否，都作为当前 section 的终点）
        raw_sections = []  # [(path, start_page, end_page, text)]

        # 前置 section：第一个锚点之前的内容归入隐式「前言」section。
        # 任何手册的开头都有封面/前言/目录等未被书签收录的内容，直接丢弃
        # 会造成内容缺失（如 2 系第 6-7 页「提示」章节里的车辆识别号）
        fp, fpos = anchors[0]
        parts = [page_text(p) for p in range(1, fp)]
        parts.append(page_text(fp)[:fpos])
        preface = "\n".join(t for t in parts if t.strip()).strip()
        if preface and estimate_tokens(preface) >= 10:
            raw_sections.append((["前言"], 1, fp, preface))

        for i, (path, title, start_page, keep) in enumerate(entries):
            if not keep and not path:
                continue  # 便捷查阅子树：整节丢弃
            sp, spos = anchors[i]
            if i + 1 < len(anchors):
                ep, epos = anchors[i + 1]
            else:
                ep, epos = total_pages, len(page_text(total_pages))
            if (sp, spos) >= (ep, epos):
                continue  # 锚点异常（标题定位失败等），跳过该 section

            # 跨页拼装：起始页取锚点之后，中间页全取，结束页取锚点之前
            if sp == ep:
                text = page_text(sp)[spos:epos].strip()
            else:
                parts = [page_text(sp)[spos:]]
                for p in range(sp + 1, ep):
                    parts.append(page_text(p))
                parts.append(page_text(ep)[:epos])
                text = "\n".join(t for t in parts if t.strip()).strip()

            if text and estimate_tokens(text) >= 10:
                raw_sections.append((path, sp, ep, text))

        # 6. 合并过短 section
        merged = merge_short_sections(raw_sections)

        # 7. 长 section 切分（page_end 记录 section 实际覆盖到的最后一页）
        chunks = []
        for path, sp, ep, text in merged:
            chunks.extend(split_long_section(path, text, sp, ep))

        logger.info(f"{pdf_path.name}: 最终 {len(chunks)} 个 chunk（MAX_CHUNK_TOKENS={MAX_CHUNK_TOKENS}）")

        # 8. 落盘原始抽取数据（书签树 + 每页文本），供调试和切分对比
        self._save_raw(pdf_path, total_pages, outline, entries, page_text)

        return chunks

    def _save_raw(
        self,
        pdf_path: Path,
        total_pages: int,
        outline: list[tuple[int, str, int]],
        entries: list[tuple[list[str], str, int, bool]],
        page_text: Callable[[int], str],
    ) -> None:
        """落盘原始抽取数据，与 processed_docs/mineru/raw 对齐。

        mineru 存 content_list.json，pypdf 存同一个 JSON：
        书签树（含被过滤的噪声条目）+ 每页文本。页文本已裁剪页眉页脚，
        与切分输入一致，可以直接对照 chunk 排查切分问题。
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        raw = {
            "pdf_name": pdf_path.name,
            "total_pages": total_pages,
            "header_footer_stripped": True,
            "outline": [[depth, title, page] for depth, title, page in outline],
            "sections": [
                {"path": path, "title": title, "page": page, "keep": keep}
                for path, title, page, keep in entries
            ],
            "pages": {str(p): page_text(p) for p in range(1, total_pages + 1)},
        }
        out = self.raw_dir / f"{pdf_path.stem}.json"
        out.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"{pdf_path.name}: 原始抽取数据已保存 {out}")
