import requests
import time
from datetime import datetime, timedelta

#获取玩家数据的脚本，支持翻页和时间范围查询，并将结果保存到文件中，文件名包含日期范围。
# ========== 配置 ==========
PLAYER_ID = "8888621"
MODE = 12
TOKEN = "****************"
LIMIT = 100
TAG = 24
DELAY = 1.2

# 日期范围（格式：YYYY-MM-DD）
START_DATE = "2026-01-01"
END_DATE = "2026-08-31"


# ==========================

def date_to_ts(date_str):
    """YYYY-MM-DD -> 毫秒时间戳 (UTC+8)"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dt_utc = dt - timedelta(hours=8)
    return int(dt_utc.timestamp() * 1000)


def now_ts():
    return int(datetime.now().timestamp() * 1000)


def fetch_page(end_ts, start_ts):
    url = f"https://5-data.amae-koromo.com/api/v2/pl4/player_records/{PLAYER_ID}/{end_ts}/{start_ts}"
    params = {
        "limit": LIMIT,
        "mode": MODE,
        "descending": "true",
        "tag": TAG,
        "cap_token_refreshed": now_ts()
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://amae-koromo.sapk.ch/",
        "Authorization": f"Bearer {TOKEN}"
    }
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    if resp.status_code == 429:
        time.sleep(30)
        return fetch_page(end_ts, start_ts)
    return resp.json()

def file_begin_end_name(start_date, end_date):
    return f"uuids_{start_date.replace('-', '')}_{end_date.replace('-', '')}.txt"


def save(uuids=list[str]):
    with open(file_begin_end_name(START_DATE, END_DATE), "a", encoding="utf-8") as f:
        for uuid in uuids:
            f.write(uuid + "\n")



# 获取起始时间戳
start_ts = date_to_ts(START_DATE)
end_ts = date_to_ts(END_DATE) + 24 * 3600 * 1000  # 包含结束日全天

all_uuids_number = 0
current_end = end_ts

while True:
    data = fetch_page(current_end, start_ts)
    # 注意：返回的数据可能是数组（直接就是列表）
    if isinstance(data, list):
        records = data
    else:
        records = data.get("records", [])

    if not records:
        break

    uuids = [r["uuid"] for r in records if "uuid" in r]
    all_uuids_number += len(uuids)
    print(f"获取 {len(uuids)} 个，累计 {all_uuids_number}")
    save(uuids)                                    #及时储存


    if len(records) < LIMIT:
        break

    # 翻页：取最后一条的 sort 字段
    new_end = records[-1].get("sort") or records[-1].get("startTime")
    if not new_end or new_end >= current_end:
        break
    current_end = new_end
    time.sleep(DELAY)


