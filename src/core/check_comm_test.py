import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
JSON_PATH = Path(__file__).resolve().parent.parent.parent / "data/x3-2024-chunks.json"

chunks = json.load(open(JSON_PATH, encoding="utf-8"))
if not chunks:
    print("chunks 为 空")
KEY_REPLACE = "本章描述了该车型系列中所提供的所有标准装备、国别装备和特殊装备。由于所选特殊装备或国别定制车辆的不同，也会提及车内不可用的装备和功能。对于安全功能和系统而言同样如此。使用相应功能和系统时注意相应适用的法律法规。"
KEY = "本章描述了该车型系列中所提供的所有标准装"
for c in chunks:
    if KEY in c["text"]:
        i = c['text'].find(KEY)
        print(f"===第{c['page']}页===")
        if c['page'] == 330 :
            print(f"{c['text'].replace(chr(10), '')}")
        else:
            print(f"{c['text'][i:i+150].replace(chr(10), '')}")
        print()
        text = c["text"].replace(chr(10), "").replace(KEY_REPLACE, "")
        print(f"替换后\n{text}")
        print()
    # else:
        # print(f"第 {c['page']} 页 没找到")
