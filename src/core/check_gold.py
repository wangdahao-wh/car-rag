"""核对评估集的 gold 页码

方法：用标准答案的多个片段去 chunks 里搜索，
      命中片段最多的页 = 答案最可能所在的页。
      再和标注的页码对比，不一致的人工核对。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
chunks = json.load(open(ROOT / "data" / "x3-2024-chunks.json", encoding="utf-8"))

QA = [
    {"id": 1,  "question": "胎压数据在哪看？",              "gt": "驾驶员侧车门的门柱上",                         "pages": [284]},
    {"id": 2,  "question": "多久检查一次轮胎压力？",         "gt": "每月两次，或长途行车前",                       "pages": [284]},
    {"id": 3,  "question": "检查胎压有什么前提条件？",       "gt": "轮胎需冷却：行驶不超过 2 km，或车辆静止至少 2 小时", "pages": [285]},
    {"id": 4,  "question": "胎压过低有什么危险？",           "gt": "轮胎发烫并损坏，影响转向和制动性能，有事故危险",   "pages": [284]},
    {"id": 5,  "question": "怎么知道油箱加满了？",           "gt": "加油枪首次自动关闭时，燃油箱已满",              "pages": [282]},
    {"id": 6,  "question": "油表见底了还能开多远？",         "gt": "可达里程低于 50 公里时可能无法供应足够燃油，应及时加油", "pages": [149,282]},
    {"id": 7,  "question": "油箱盖怎么关才对？",             "gt": "顺时针旋转直到听见清晰的咔止声，再按紧盖板到卡入", "pages": [283]},
    {"id": 8,  "question": "油箱盖打不开怎么办？",           "gt": "需由授权售后服务经销商手动解锁",                "pages": [283]},
    {"id": 9,  "question": "车灯开关在哪？",                "gt": "方向盘附近",                                 "pages": [155]},
    {"id": 10, "question": "自动车灯控制怎么工作？",         "gt": "近光灯根据周边环境亮度自动开启或关闭",           "pages": [155]},
    {"id": 11, "question": "雾天自动大灯能识别吗？",         "gt": "不能，传感器无法识别雾天和潮湿天气，需手动开启车灯", "pages": [155]},
    {"id": 12, "question": "车内空气是怎么净化的？",         "gt": "通过车厢内排放监测器、车内过滤器、空调系统、驻车通风、电离技术、香氛等", "pages": [241]},
    {"id": 13, "question": "空调滤芯什么时候换？",           "gt": "车辆制造商建议在车辆保养时更换车内过滤器",        "pages": [241]},
    {"id": 14, "question": "警告三角标志放哪？",             "gt": "行李箱左侧储物格中",                          "pages": [318]},
    {"id": 15, "question": "反光安全背心放在哪？",           "gt": "驾驶侧或前乘客侧车门的储物格，或手套箱内",        "pages": [318]},
    {"id": 16, "question": "警示闪光灯按钮在哪？",           "gt": "中央控制台内",                              "pages": [318]},
    {"id": 17, "question": "座椅位置能记忆吗？",             "gt": "可以电动调整的座椅支持用记忆功能存储当前座椅位置", "pages": [102]},
    {"id": 18, "question": "调整座椅后要注意什么？",         "gt": "调整后稍微前后移动座椅，确认已正确嵌入锁定",      "pages": [102]},
    {"id": 19, "question": "冬天该换什么轮胎？怎么识别？",    "gt": "建议冬季使用冬季用轮胎；可通过雪花山峰图标及轮胎侧壁的 M+S 标记识别", "pages": [288]},
    {"id": 20, "question": "带 M+S 标记的四季轮胎能代替冬季轮胎吗？", "gt": "冬季特性优于夏季轮胎，但通常达不到冬季轮胎的性能", "pages": [288]},
    {"id": 21, "question": "前后轮胎可以对调吗？",           "gt": "仅当前后轴轮胎尺寸和轮辋尺寸相同时才可对调",      "pages": [288]},
    {"id": 22, "question": "行驶中提示轮胎损坏该怎么办？",    "gt": "立刻减速并停车，检查车轮和轮胎，小心驾驶到售后服务经销商；不要修理，应更换", "pages": [286]},
    {"id": 23, "question": "什么情况会损坏轮胎？",           "gt": "驶过路缘、道路损坏、轮胎压力过低、车辆超载、不正确的轮胎储存", "pages": [286]},
    {"id": 24, "question": "限速信息是怎么采集的？",         "gt": "车内后视镜区域的摄像头采集道路边缘的道路标志",    "pages": [197]},
    {"id": 25, "question": "长途出行前应该做哪些检查？",      "gt": "检查轮胎压力、检查轮胎状况、确认燃油充足",        "pages": [32, 284, 286, 282]},
    {"id": 26, "question": "车辆保养相关的事有哪些？",        "gt": "保养系统提示保养措施、空调滤芯在保养时更换、车况保养", "pages": [312, 241]},
    {"id": 27, "question": "车内有哪些应急装备？分别放在哪？", "gt": "反光安全背心、警告三角标志、警示闪光灯、包扎绷带", "pages": [318]},
    {"id": 28, "question": "这辆车在火星上能开吗？",         "gt": None, "pages": []},
    {"id": 29, "question": "换个轮胎要多少钱？",             "gt": None, "pages": []},
    {"id": 30, "question": "这车和奔驰 GLC 哪个好？",        "gt": None, "pages": []},
]

def find_candidata_pages(gt, win=5, step=2):
    """滑动窗口搜索，返回 [(页码, 命中片段数), ...] 按命中数降序"""
    pages = {}
    print(f"type of pages is {type(pages)}")
    for i in range(0, max(1, len(gt) - win + 1), step):
        frag = gt[i:i+win]
        if len(frag) < win:
            continue
        for c in chunks:
            if frag in c["text"]:
                pages[c["page"]] = pages.get(c['page'], 0) + 1
    return sorted(pages.items(), key=lambda x: -x[1])

for item in QA:
    if item["gt"] is None:
        continue
    print(f"===== 题 {item['id']}: {item['question']} =====")
    print(f"  标准答案: {item['gt']}")
    print(f"  标注页码: 第 {item['pages']} 页")
    cands = find_candidata_pages(item['gt'])
    if not cands:
        print("  ⚠️ 没搜到任何片段（答案措辞和原文差异大，需人工核对）")
    else:
        top = cands[:5]
        top_pages = [p for p,_ in top]
        # mark = "✅" if item["page"] in [p for p, _ in top] else "❌"
        mark = "✅" if any(p in top_pages for p in item["pages"])else "❌"
        print(f"  {mark} 候选页（页码, 命中片段数）: {top}")

    print()

out = []
for item in QA:
    out.append({
        "id": item["id"],
        "question": item["question"],
        "groud_truth": item["gt"],
        "source_pages": item["pages"],
        "type": "refuse" if item["gt"] is None else "basic",
    })
for o in out:
    if o["id"] == 25:
        o["type"] = "multi_hop"


QA_PATH = ROOT / "eval" / "qa_set.json"
QA_PATH.parent.mkdir(exist_ok=True)
with open(QA_PATH, "w", encoding="utf-8") as f:
    for row in out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"已保存 {len(out)} 题到 {QA_PATH}")

