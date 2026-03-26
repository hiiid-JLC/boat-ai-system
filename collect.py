"""
collect.py ─ ボートレース結果収集スクリプト

使い方:
    # 全場・期間指定
    python collect.py --start 2025-01-01 --end 2025-01-31

    # 場所を絞る（桐生・浜名湖・住之江）
    python collect.py --start 2025-01-01 --end 2025-01-07 --jcd 01 06 12

    # 直近N日・全場
    python collect.py --days 7

    # 直近3日・特定場・既存CSVに追記
    python collect.py --days 3 --jcd 01 12 --append

    # 1〜6Rのみ
    python collect.py --start 2025-01-01 --end 2025-01-03 --rno 1 2 3 4 5 6
"""
import argparse
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────
# 定数
# ─────────────────────────────────────────
BASE = "https://www.boatrace.jp/owpc/pc/race"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BoatAI/2.0)"}

VENUES = {
    "01":"桐生",  "02":"戸田",   "03":"江戸川", "04":"平和島",
    "05":"多摩川","06":"浜名湖", "07":"蒲郡",   "08":"常滑",
    "09":"津",    "10":"三国",   "11":"びわこ", "12":"住之江",
    "13":"尼崎",  "14":"鳴門",   "15":"丸亀",   "16":"児島",
    "17":"宮島",  "18":"徳山",   "19":"下関",   "20":"若松",
    "21":"芦屋",  "22":"福岡",   "23":"唐津",   "24":"大村",
}

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "race_data.csv")


# ─────────────────────────────────────────
# ネットワーク（リトライ付き）
# ─────────────────────────────────────────
def get_soup(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            return BeautifulSoup(r.text, "lxml")
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def _safe_float(text: str):
    try:
        return float(str(text).strip().replace(",", ""))
    except Exception:
        return None


# ─────────────────────────────────────────
# レース結果（着順・ST）
# ─────────────────────────────────────────
def get_result(jcd: str, rno: str, date: str) -> list:
    url  = f"{BASE}/raceresult?jcd={jcd}&rno={rno}&hd={date}"
    soup = get_soup(url)
    if not soup:
        return []

    data = []
    for row in soup.select("table tbody tr"):
        cols = row.find_all("td")
        if len(cols) < 4:
            continue
        try:
            tno      = int(cols[1].text.strip())
            chakujun = int(cols[0].text.strip())
            st_txt   = cols[3].text.strip()
            # F(フライング)・L(出遅れ)等はNaNに
            st_val = _safe_float(st_txt) if st_txt not in ("F","L","K","S","") else None
            if not (1 <= tno <= 6 and 1 <= chakujun <= 6):
                continue
            data.append({"艇番": tno, "着順": chakujun, "ST": st_val})
        except Exception:
            continue
    return data


# ─────────────────────────────────────────
# 出走表（全国勝率・モーター勝率）
# ─────────────────────────────────────────
def get_racelist(jcd: str, rno: str, date: str) -> dict:
    url  = f"{BASE}/racelist?jcd={jcd}&rno={rno}&hd={date}"
    soup = get_soup(url)
    if not soup:
        return {}

    data = {}
    for row in soup.select("table tbody tr"):
        cols = row.find_all("td")
        if len(cols) < 7:
            continue
        try:
            boat = int(cols[0].text.strip())
            if not (1 <= boat <= 6):
                continue
            data[boat] = {
                "全国勝率":    _safe_float(cols[2].text),
                "モーター勝率": _safe_float(cols[6].text),
            }
        except Exception:
            continue
    return data


# ─────────────────────────────────────────
# 直前情報（展示タイム）
# ─────────────────────────────────────────
def get_beforeinfo(jcd: str, rno: str, date: str) -> dict:
    url  = f"{BASE}/beforeinfo?jcd={jcd}&rno={rno}&hd={date}"
    soup = get_soup(url)
    if not soup:
        return {}

    data = {}
    for row in soup.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 4:
            continue
        try:
            tno = int(cols[0].text.strip())
            if not (1 <= tno <= 6):
                continue
            # 展示タイムは 6.5〜7.5 秒の範囲で探す
            for ci in range(1, min(len(cols), 10)):
                val = _safe_float(cols[ci].text)
                if val and 6.0 < val < 8.0:
                    data[tno] = val
                    break
        except Exception:
            continue
    return data


# ─────────────────────────────────────────
# 1レース分まとめて収集
# ─────────────────────────────────────────
def collect_race(jcd: str, rno: str, date: str) -> list:
    result = get_result(jcd, rno, date)
    if not result:
        return []

    racelist   = get_racelist(jcd, rno, date)
    beforeinfo = get_beforeinfo(jcd, rno, date)
    race_id    = f"{date}_{jcd}_{rno}"

    rows = []
    for r in result:
        boat = r["艇番"]
        rows.append({
            "レースID":    race_id,
            "日付":        date,
            "場コード":    jcd,
            "場名":        VENUES.get(jcd, "?"),
            "レース番号":  int(rno),
            "艇番":        boat,
            "コース":      boat,
            "着順":        r["着順"],
            "ST":          r["ST"],
            "展示タイム":  beforeinfo.get(boat),
            "全国勝率":    racelist.get(boat, {}).get("全国勝率"),
            "モーター勝率": racelist.get(boat, {}).get("モーター勝率"),
        })
    return rows


# ─────────────────────────────────────────
# メイン収集ループ
# ─────────────────────────────────────────
def collect(
    start_date: str,
    end_date: str,
    jcd_list: list = None,
    rno_list: list = None,
    append: bool = False,
    sleep: float = 1.0,
) -> pd.DataFrame:
    jcd_list = jcd_list or [str(j).zfill(2) for j in range(1, 25)]
    rno_list = rno_list or [str(r).zfill(2) for r in range(1, 13)]
    dates    = pd.date_range(start_date, end_date)

    total = len(dates) * len(jcd_list) * len(rno_list)

    print("=" * 55)
    print("  ボートレース データ収集")
    print("=" * 55)
    print(f"  期間:   {start_date} 〜 {end_date}  ({len(dates)}日)")
    print(f"  競艇場: {', '.join(VENUES.get(j, j) for j in jcd_list)}")
    print(f"  レース: {rno_list[0]}R〜{rno_list[-1]}R  ({len(rno_list)}R)")
    print(f"  合計:   最大 {total} レース")
    print("=" * 55)

    all_data = []
    done     = 0

    for d in dates:
        date = d.strftime("%Y%m%d")
        print(f"\n▶ {date}")

        for jcd in jcd_list:
            name = VENUES.get(jcd, jcd)

            for rno in rno_list:
                done += 1
                pct   = done / total * 100

                rows = collect_race(jcd, rno, date)

                status = f"{len(rows)}艇" if rows else "スキップ"
                print(f"  [{pct:5.1f}%] {name}({jcd}) {rno}R → {status}")

                if rows:
                    all_data.extend(rows)

                time.sleep(sleep)

    print(f"\n収集完了: {len(all_data)} 行")

    if not all_data:
        print("取得データが0件でした。日付・場コードを確認してください。")
        return pd.DataFrame()

    df_new = pd.DataFrame(all_data)

    # 追記 or 新規保存
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    if append and os.path.exists(DATA_PATH):
        df_old = pd.read_csv(DATA_PATH)
        df     = pd.concat([df_old, df_new], ignore_index=True)
        df     = df.drop_duplicates(subset=["レースID", "艇番"])
        print(f"追記: {len(df_old)}行 + {len(df_new)}行 → {len(df)}行（重複除去後）")
    else:
        df = df_new

    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"保存: {DATA_PATH}")
    return df


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def build_parser():
    parser = argparse.ArgumentParser(
        description="ボートレース結果収集スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python collect.py --start 2025-01-01 --end 2025-01-31
  python collect.py --start 2025-01-01 --end 2025-01-07 --jcd 01 06 12
  python collect.py --days 7
  python collect.py --days 3 --jcd 01 12 --append
  python collect.py --start 2025-01-01 --end 2025-01-03 --rno 1 2 3 4 5 6

場コード:
  01:桐生  02:戸田  03:江戸川 04:平和島 05:多摩川 06:浜名湖
  07:蒲郡  08:常滑  09:津     10:三国   11:びわこ 12:住之江
  13:尼崎  14:鳴門  15:丸亀   16:児島   17:宮島   18:徳山
  19:下関  20:若松  21:芦屋   22:福岡   23:唐津   24:大村
        """,
    )

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--start", type=str,
                     help="開始日 YYYY-MM-DD  (--end と一緒に使う)")
    grp.add_argument("--days",  type=int,
                     help="直近N日分を収集")

    parser.add_argument("--end",    type=str,   default=None,
                        help="終了日 YYYY-MM-DD (デフォルト: 今日)")
    parser.add_argument("--jcd",    type=str,   nargs="+", default=None,
                        help="場コード 例: 01 06 12  (省略で全24場)")
    parser.add_argument("--rno",    type=int,   nargs="+", default=None,
                        help="レース番号 例: 1 2 3  (省略で1〜12R)")
    parser.add_argument("--append", action="store_true",
                        help="既存CSVに追記する（重複は自動除去）")
    parser.add_argument("--sleep",  type=float, default=1.0,
                        help="リクエスト間隔（秒）デフォルト: 1.0")
    return parser


def main():
    args = build_parser().parse_args()

    # 期間を確定
    today = datetime.today().strftime("%Y-%m-%d")
    if args.days:
        end_date   = today
        start_date = (datetime.today() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    else:
        start_date = args.start
        end_date   = args.end or today

    # 場コード正規化・バリデーション
    jcd_list = None
    if args.jcd:
        jcd_list = [j.zfill(2) for j in args.jcd]
        invalid  = [j for j in jcd_list if j not in VENUES]
        if invalid:
            print(f"[ERROR] 無効な場コード: {invalid}")
            print(f"        有効: {list(VENUES.keys())}")
            return

    # レース番号正規化・バリデーション
    rno_list = None
    if args.rno:
        invalid = [r for r in args.rno if not (1 <= r <= 12)]
        if invalid:
            print(f"[ERROR] レース番号は1〜12で指定してください: {invalid}")
            return
        rno_list = [str(r).zfill(2) for r in args.rno]

    collect(
        start_date = start_date,
        end_date   = end_date,
        jcd_list   = jcd_list,
        rno_list   = rno_list,
        append     = args.append,
        sleep      = args.sleep,
    )


if __name__ == "__main__":
    main()
