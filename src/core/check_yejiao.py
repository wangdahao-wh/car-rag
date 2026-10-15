from pypdf import PdfReader

reader = PdfReader("docs/x3-2024.pdf")
for i in [5, 25, 60, 110, 160, 210, 260, 320]:     # 分散抽样
    t = reader.pages[i-1].extract_text() or ""
    lines = [l for l in t.split("\n") if l.strip()]
    print(f"第{i:>3}页 | 首行: {lines[0][:45] if lines else '(空)'}")
    print(f"        | 末行: {lines[-1][:45] if lines else '(空)'}")

