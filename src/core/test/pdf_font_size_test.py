import os
from pathlib import Path
import sys
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = ROOT / "docs"

def pdf_font_size_print(pdf_path: Path):
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    print(f"一共有{len(pdf_files)}个 pdf 文件")
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"开始处理第{idx}个 pdf 文件，文件名称为\n：{pdf_path.name}")

        reader = PdfReader(str(pdf_path))
        sample_pages = min(3, len(reader.pages))

        font_sizes = []
        # 记录每个字号对应的示例文字，用于判断该字号是标题还是正文
        size_texts: dict[float, list[str]] = {}
        for i in range(sample_pages):
            
            page = reader.pages[i]
            def visitor(text, cm, tm, font_dict, font_size):
                if text.strip():
                    size = round(font_size, 1)
                    font_sizes.append(size)
                    size_texts.setdefault(size, []).append(text.strip())
            try:
                page.extract_text(visitor_text=visitor)
            except Exception:
                pass
        
        if not font_sizes:
            print(f"没有字体大小")
            continue
            
        unique_sizes = set(font_sizes)
        print("unique_sizes = \n", unique_sizes)

        unique_sizes_number = len(unique_sizes)
        print(f"字号种类数: {unique_sizes_number}")

        # 每个字号打印前 5 条文字，按字号从大到小排列（大字通常是标题）
        for size in sorted(unique_sizes, reverse=True):
            samples = size_texts.get(size, [])[:5]
            preview = " | ".join(s.replace("\n", " ")[:30] for s in samples)
            print(f"  字号 {size}: {preview}")
        


if __name__ == "__main__":
    print("docs_dir = ", DOCS_DIR)
    pdf_font_size_print(DOCS_DIR)
