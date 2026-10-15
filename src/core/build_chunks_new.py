import json
import sys
from pathlib import Path
from collections import Counter
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pypdf import PdfReader
ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "docs" / "x3-2024.pdf"
JSON_PATH = ROOT / "data" / "x3-2024-chunks.json"

MIN_CHARS = 30          # 少于这个字数的 chunk 丢弃
SPLIT_SIZE = 11.0       # >= 这个字号 → 切分成新小节
HEADING_SIZE = 10.0     # >= 这个字号 → 算标题（进路径）

import re
def is_index_page(text):
    """判断是否是索引页： 大量「中文+数字」配对 (关键字 + 页码)"""
    return len(re.findall(r"[一-鿿]{2,}\s*\d{2,4}", text)) > 8

# 抽取
def extract_blocks(page):
    """用字体信息抽取片段，并剔除页脚
    Returns:
        list[tuple]: [(字号, 文本), ...]
    """
    parts = []

    def visitor(text, cm, tm, font_dict, font_size):
        t = text.strip()
        if t:
            parts.append((round(font_size,1), t))
    page.extract_text(visitor_text=visitor)

    for i in range(len(parts)-1,-1,-1):
        if parts[i][1].startswith("Seite "):
            parts = parts[:i]
            break
    return parts

# 切分
def join_lines(lines):
    """把 PDF 拆行的片段拼回完成的句子

    规则：上一片段末尾不是句末标点-->说明话没说完--->直接接上
    """
    out = ""
    for l in lines:
        if out and out[-1] not in "。！？；：":
            out += l
        else:
            out += ("\n" if out else "") + l
    return out

def build_sections(parts : list):
    """按标题切分，每节带【完整标题路径】
    Returns:
    list[tuple]: [([(字号,标题), ...], [行, 行, ...]), ...]
    """
    sections = []
    titles = []
    lines = []

    for size, text in parts:
        if size >= HEADING_SIZE:
            if size >= SPLIT_SIZE and lines:
                sections.append((list(titles), lines))
                lines=[]
            titles = [t for t in titles if t[0] > size] + [(size, text)]
        else:
            lines.append(text)

    if lines:
        sections.append((list(titles), lines))

    return sections

def section_text(titles, lines):
    """把标题路径+正文拼成 chunk 文本"""
    path = " > ".join(t for _, t in titles)
    body = join_lines(lines)
    if not path:
        return body
    return f"[{path}]\n{body}"


# 3 组装
def build_chunks(pdf_path=PDF_PATH):
    reader = PdfReader(str(pdf_path))
    chunks = []

    for pno, page in enumerate(reader.pages, 1):
        parts = extract_blocks(page)
        if not parts:
            continue

        for titles, lines in build_sections(parts):
            text = section_text(titles, lines)

            if len(text) < MIN_CHARS: # 过短的丢弃
                continue
            if text.count("...") > 5: # 目录页丢弃
                continue
            if is_index_page(text):
                continue

            path = [t for _, t in titles]
            chunks.append({
                "text": text,
                "page": pno,
                "chapter": path[0] if len(path) > 0 else "",
                "section": path[1] if len(path) > 1 else "",
                "path": path, # 完整路径（如 ["车灯","自动车灯控制","系统的局限性"]）
            })

    return chunks

def print_stats(chunks, total_pages):
    """打印统计"""
    lens = [len(c["text"]) for c in chunks]
    print(f"总页数:       {total_pages}")
    print(f"生成 chunk:   {len(chunks)}")
    print(f"总字数:       {sum(lens)}")
    print(f"平均字数:     {sum(lens) // len(lens)}")
    print(f"中位字数:     {int(median(lens))}")
    print(f"最长/最短:    {max(lens)} / {min(lens)}")

    print(f"\n字数分布:")
    for lo, hi, name in [(0,100,'<100'), (100,300,'100-300'), (300,600,'300-600'), (600,99999,'>600')]:
        n = sum(1 for x in lens if lo <= x < hi)
        print(f"  {name:<10} {n:>4} 个")

    print(f"\n章节分布（前 10）:")
    counter = Counter(c["path"][0] if c["path"] else "(无)" for c in chunks)
    for name, n in counter.most_common(10):
        print(f"  {name:<20} {n:>4} 个")

def save_chunks(chunks, path):
    """保存成 JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 {len(chunks)} 个 chunk 到 {path}")

if __name__ == "__main__":
    chunks = build_chunks()
    reader = PdfReader(str(PDF_PATH))
    print_stats(chunks, len(reader.pages))
    save_chunks(chunks, JSON_PATH)

    # 抽 3 个看内容
    print("\n" + "=" * 60)
    for i in [0, len(chunks) // 2, len(chunks) - 1]:
        c = chunks[i]
        print(f"\n----- chunk[{i}] 第{c['page']}页 -----")
        print(c["text"][:200])