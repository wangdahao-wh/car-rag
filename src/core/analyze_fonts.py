"""分析 PDF 字体分布，用于调整切分参数。

用法：
    python src/core/analyze_fonts.py docs/领克_2025_03_用户手册.pdf
"""
import sys
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


def analyze_pdf_fonts(pdf_path: Path, sample_pages: int = 10):
    """分析 PDF 的字体分布。"""
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    
    print(f"=" * 60)
    print(f"文件: {pdf_path.name}")
    print(f"总页数: {total_pages}")
    print(f"=" * 60)
    
    # 采样页面（前几页 + 中间几页）
    sample_indices = list(range(min(5, total_pages))) + \
                     list(range(total_pages // 2, min(total_pages // 2 + 5, total_pages)))
    
    font_sizes = []
    font_text_samples = {}  # {字号: [文本示例]}
    
    for page_idx in sample_indices:
        page = reader.pages[page_idx]
        
        def visitor(text, cm, tm, font_dict, font_size):
            t = text.strip()
            if t and len(t) > 2:  # 过滤太短的
                size = round(font_size, 1)
                font_sizes.append(size)
                if size not in font_text_samples:
                    font_text_samples[size] = []
                if len(font_text_samples[size]) < 3:  # 每个字号保留3个示例
                    font_text_samples[size].append(t[:50])  # 截断
        
        page.extract_text(visitor_text=visitor)
    
    # 统计字号分布
    counter = Counter(font_sizes)
    
    print(f"\n字号分布（采样 {len(sample_indices)} 页）：")
    print(f"{'字号':<10} {'出现次数':<15} {'示例文本'}")
    print("-" * 60)
    
    for size, count in sorted(counter.items(), key=lambda x: -x[0]):
        samples = font_text_samples.get(size, [])
        sample_text = samples[0] if samples else ""
        print(f"{size:<10} {count:<15} {sample_text}")
    
    # 给出建议
    sizes_sorted = sorted(counter.keys(), reverse=True)
    if len(sizes_sorted) >= 3:
        print(f"\n建议参数：")
        print(f"SPLIT_SIZE  = {sizes_sorted[1]}  # 第2大字号（章节标题）")
        print(f"HEADING_SIZE = {sizes_sorted[2]}  # 第3大字号（子标题）")
    
    print(f"\n完整字号列表（从大到小）：")
    print(sizes_sorted)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python src/core/analyze_fonts.py <pdf文件路径>")
        print("示例: python src/core/analyze_fonts.py docs/领克_2025_03_用户手册.pdf")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"错误：文件不存在 {pdf_path}")
        sys.exit(1)
    
    analyze_pdf_fonts(pdf_path)
