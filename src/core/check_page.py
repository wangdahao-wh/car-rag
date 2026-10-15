from pypdf import PdfReader
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
PDF_PATH = Path(__file__).resolve().parent.parent.parent / "docs/x3-2024.pdf"

from src.core.clean_page import clean_page

reader = PdfReader(str(PDF_PATH))
for idx in [19,38,269,281,335]:
    page = reader.pages[idx-1].extract_text() or ""
    text, meta = clean_page(page, idx)
    print(f"===== 第 {idx} 页 =====")
    print(f"  清理前: {len(page)} 字, 内容: |{page[:200]}|")
    print(f"  清理后: {len(text)} 字")
    print(f"  内容: |{text[:200]}|")
    print()

longest_page, longest_len = 0,0
for pno, page in enumerate(reader.pages, 1):
    text, _ = clean_page(page.extract_text() or "", pno)
    if len(text) > longest_len:
        longest_len = len(text)
        longest_page = pno
print(f"最长页：第 {longest_page} 页，共 {longest_len} 字")