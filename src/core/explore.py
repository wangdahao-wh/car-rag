from pypdf import PdfReader
from statistics import median

reader = PdfReader("docs/x3-2024.pdf")

texts = []
print(len(reader.pages))
for page in reader.pages:
    t = page.extract_text() or ""
    texts.append(t)

# counts = [len(t) for t in texts]
# print("页总数：", len(texts))
# print("字数： 最小/最大/中位数：", min(counts), max(counts), int(median(counts)))

# print("\n 字数 < 50 页（可能是封面/插图页）:")
# for i, c in enumerate(counts, 1):
#     if c < 50:
#         print(f"  第 {i} 页，共 {c} 个字")

# for i in [5,10,15,50,100,200]:
#     print(f"\n====第{i}页（共{counts[i-1]}个字）")
#     print(texts[i-1][:300])

# for start,page in enumerate(reader.pages,1):
#     text = page.extract_text(extraction_mode="plain")
#     print(f"{start} : {text.count}")

for i in [20,21,50,51,100]:
    # print(f"\n====第{i}页（共{counts[i-1]}个字）")
    # print(f"开头5行输出：")
    # for text in texts[i-1].split("\n")[:5]:
    #     print(f"| {text[:30]}")
    print(f"末尾5行输出：")
    for text in texts[i-1].split("\n")[-5:]:
        print(f"| {text}")

# for i, text in enumerate(texts, 1):
#     if not text.count("Seite"):
#         print(f"第{i}页不包含 'Seite' 字样，可能是页码或页脚")

# for pno in [1,2,3,4,5,352,353,354,355,356]:
#     t = reader.pages[pno-1].extract_text() or ""
#     lines = [l for l in t.split("\n") if l.strip()]
#     print(f"\n====第{pno}页（共{len(lines)}行）")
#     if lines:
#         print(f"开头5行输出：")
#         for text in lines[:5]:
#             print(f"| {text[:30]}")
#         print(f"末尾5行输出：")
#         for text in lines[-5:]:
#             print(f"| {text}")
#     else:
#         print("该页没有内容")
    


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.clean_page import clean_page
texts_after = []
meta_after = []
for i,text in enumerate(texts,1):
    after_text, m = clean_page(text,i)
    texts_after.append(after_text)
    meta_after.append(m)

for i in [20,21,50,51,100,101,200,201]:
    print(f" 第{i}页，末尾5行输出 ")
    for text in texts_after[i-1].split("\n")[-5:]:
        print(f"| {text}")
    print(f" 第{i}页，元数据输出 ")
    print(f" | {meta_after[i-1]}")

# for i in [4,5]:
#     page = reader.pages[i-1]
#     t = page.extract_text() or ""
#     lines = [l for l in t.split("\n") if l.strip()]
#     print(f"\n====第{i}页（共{len(lines)}行）")
#     if lines:
#         for line in lines:
#             print(f"| {line[:30]}")

# chapters = set()
# for m in meta_after:
#     if len(m["breadcrumb"]) == 2:
#         chapters.add(m["breadcrumb"][0])
# print(f"共 {len(chapters)} 个章节：")
# for c in sorted(chapters):
#     print(f"  {c}")
