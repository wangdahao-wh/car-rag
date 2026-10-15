import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pypdf import PdfReader
from src.core.clean_page import clean_page

import re

MIN_CHARS = 20
PDF_PATH = Path(__file__).resolve().parent.parent.parent / "docs/x3-2024.pdf"
JSON_PATH = Path(__file__).resolve().parent.parent.parent / "data/x3-2024-chunks.json"
KEY = "本章描述了该车型系列中所提供的所有标准装"
TEMPLATE = (
    "本章描述了该车型系列中所提供的所有标准装备、国别装备和特殊装备。"
    "由于所选特殊装备或国别定制车辆的不同，也会提及车内不可用的装备和功能。"
    "对于安全功能和系统而言同样如此。使用相应功能和系统时注意相应适用的法律法规。"
)
TEMPLATE_RE = re.compile(r"\s*".join(re.escape(ch) for ch in TEMPLATE)) # /s 匹配任何空白（含换行）

def build_chunks(pdf_path=PDF_PATH):
    """把 PDF 切分成 chunk 列表

    Return:
        list[dict] : 每个元素形如
            {
                "text": "正文...",  # 文本内容
                "page": 100, # 页码
                "breadcrumb": ["操作","打开和关闭"] # 章 节
            }
    """
    reader = PdfReader(pdf_path)
    chunks = []
    for pno, page in enumerate(reader.pages, 1):
        raw = page.extract_text() or ""
        text, meta = clean_page(raw, pno)
        if len(text) < MIN_CHARS:
            print(f"第{pno}页内容过短，忽略")
            continue

        if text.count('...') > 5:
            print(f"第{pno}页为目录页，忽略")
            continue

        # 替换模版话
        text = remove_boilerplate(text)

        chunks.append({
            "text": text,
            "page": meta["page"],
            "breadcrumb": meta["breadcrumb"]
        })

    return chunks

def remove_boilerplate(text: str) -> str:
    """删除章节模版话"""
    return TEMPLATE_RE.sub("", text)

def print_stats(chunks, total_pages):
    """打印统计信息"""
    lengths = [len(c["text"]) for c in chunks]
    print(f"总页数：{total_pages}")
    print(f"生成的 chunk： {len(chunks)}")
    print(f"跳过的页数： {total_pages - len(chunks)}")
    print(f"总字数： {sum(lengths)}")
    print(f"平均字数: {sum(lengths) / len(lengths):.2f}")
    print(f"最长chunk： {max(lengths)}")
    print(f"最短chunk： {min(lengths)}")

    print("\n章节分布 （前12）：")
    counter = Counter(
        ">".join(c["breadcrumb"]) if c["breadcrumb"] else "(无章节)" for c in chunks
    )
    print(f"共 {len(counter)} 个章节")
    for name, n in counter.most_common(12):
        print(f"  {name:<10} : {n:>3} 个")

import json
def save_chunks(chunks, path):
    with open(path, "w", encoding= "utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(chunks)} 个 chunk 到 {path}")

if __name__ == "__main__":
    chunks = build_chunks()

    reader = PdfReader(PDF_PATH)
    print_stats(chunks, len(reader.pages))

    # 抽样看 chunk 内容
    print("\n抽样查看 chunk 内容：")
    for idx in [0, len(chunks) // 3, len(chunks) // 2, len(chunks) - 1]:
        c = chunks[idx]
        head = " > ".join(c["breadcrumb"]) if c["breadcrumb"] else "(无章节)"
        print(f"\n---- chunk[{idx}] 第{c['page']}页 | {head} | 字数: {len(c['text'])}")
        print(f"  {c['text'][:100]}")

    save_chunks(chunks, JSON_PATH)

