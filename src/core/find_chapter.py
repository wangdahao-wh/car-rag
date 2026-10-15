from pypdf import PdfReader

# reader = PdfReader("docs/x3-2024.pdf")

# texts = []
# print(f"页总数：{len(reader.pages)}")
# for page,i in enumerate(reader.pages,1):
#     if not i % 2:


import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

ROOT = Path(__file__).resolve().parent.parent.parent
JSON_PAth = ROOT / "data" / "x3-2024-chunks.json"
chunks = json.load(open(JSON_PAth, encoding="utf-8"))

for c in chunks:
    if c['page'] >= 330:
        print(f"  第{c['page']}页 [{'>'.join(c['path'][:2])}]: {c['text'][:70]}")