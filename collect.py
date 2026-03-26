"""
collect.py ─ レース結果収集スクリプト
使い方: python collect.py [--jcd 01] [--days 7]
"""
import os
import sys
import argparse
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "race_data.csv")
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BoatAI/2.0)"}
VENUES  = {
    "01":"桐生","02":"戸田","03":"江戸川","04":"平和島","05":"多摩川",
    "06":"浜名湖","07":"蒲郡","08":"常滑","09":"津","10":"三国",
    "11":"びわこ","12":"住之江","13":"尼崎","14":"鳴門","15":"丸亀",
    "16":"児島","17":"宮島","18":"徳山","19":"下関","20":"若松",
    "21":"芦屋","22":"福岡","23":"唐津","24":"大村",
}


def safe_get(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  [WARN] {url} → {e}")
        return None


def parse_result(jcd: str, rno: str, date_str: str) -> list[dict]:
    """レース結果ページをパース"""
    url = (
        f"https://www.boatrace.jp/owpc/pc/race/raceresult"
        f"?rno={rno}&jcd={jcd}&hd={date_str}"
    )
    html = safe_get(url)
    if html is None:
        return []

    soup = BeautifulSoup(html, "lxml")
    data = []

    for row in soup.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 6:
            continue
        try:
            entry = {
                "レースID":   f"{date_str}_{jcd}_{rno}",
                "場コード":   jcd,
                "場名":       VENUES.get(jcd, "?"),
                "レース番号": int(rno),
                "日付":       date_str,
                "艇番":       int(cols[1].text.strip()),
                "着順":       int(cols[0].text.strip()),
                "コース":     int(cols[1].text.strip()),  # 簡易
                "ST":         float(cols[3].text.strip()),
                "展示タイム": None,  # 直前情報から別途取得が必要
                "モーター勝率": None,
                "全国勝率":    None,
            }
            data.append(entry)
        except Exception:
            continue

    return data


def collect_date(date_str: str, jcd_list: list[str]) -> list[dict]:
    """1日分を収集"""
    all_data = []

    for jcd in jcd_list:
        print(f"  {VENUES.get(jcd, jcd)}場 ({jcd}) ...")
        for rno in range(1, 13):
            rows = parse_result(jcd, str(rno).zfill(2), date_str)
            all_data.extend(rows)
            time.sleep(0.8)  # サーバー負荷軽減

    return all_data


def main():
    parser = argparse.ArgumentParser(description="ボートレース結果収集")
    parser.add_argument("--days",  type=int, default=3,  help="直近何日分を収集するか")
    parser.add_argument("--jcd",   type=str, default="all", help="場コード (例: 01) or 'all'")
    parser.add_argument("--append", action="store_true",  help="既存CSVに追記")
    args = parser.parse_args()

    # 対象場
    if args.jcd == "all":
        jcd_list = list(VENUES.keys())
    else:
        jcd_list = [args.jcd.zfill(2)]

    # 対象日
    dates = [
        (datetime.today() - timedelta(days=i)).strftime("%Y%m%d")
        for i in range(1, args.days + 1)
    ]

    all_data = []
    for d in dates:
        print(f"\n▶ {d} 収集中...")
        rows = collect_date(d, jcd_list)
        all_data.extend(rows)
        print(f"  取得行数: {len(rows)}")

    if not all_data:
        print("データが取得できませんでした。")
        sys.exit(1)

    df_new = pd.DataFrame(all_data)

    if args.append and os.path.exists(DATA_PATH):
        df_old = pd.read_csv(DATA_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["レースID", "艇番"])
    else:
        df = df_new

    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"\n保存完了: {DATA_PATH}  (合計 {len(df)} 行)")


if __name__ == "__main__":
    main()
