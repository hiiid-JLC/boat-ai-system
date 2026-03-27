"""
utils.py ─ ボートAIシステム 共通ロジック
"""
import itertools
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


# ─────────────────────────────────────────
# 定数
# ─────────────────────────────────────────
VENUES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島",
    "05": "多摩川", "06": "浜名湖", "07": "蒲郡",  "08": "常滑",
    "09": "津",    "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀",   "16": "児島",
    "17": "宮島", "18": "徳山", "19": "下関",   "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津",   "24": "大村",
}

FEATURES = ["コース", "ST順位", "展示順位", "内枠"]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BoatAI/2.0)"}


# ─────────────────────────────────────────
# ネットワーク
# ─────────────────────────────────────────
def safe_request(url: str, timeout: int = 8) -> str | None:
    try:
        res = requests.get(url, headers=HEADERS, timeout=timeout)
        res.raise_for_status()
        return res.text
    except Exception:
        return None


# ─────────────────────────────────────────
# 数学
# ─────────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    e = np.exp(x - np.max(x))
    return e / e.sum()


def kelly(p: float, odds: float) -> float:
    """ケリー基準（負の場合は0）"""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max((p * odds - 1.0) / b, 0.0)


# ─────────────────────────────────────────
# スクレイピング：直前情報
# ─────────────────────────────────────────
def get_data(jcd: str, rno: str) -> pd.DataFrame:
    """
    直前情報ページから6艇分のデータを取得。
    失敗時は空のDataFrameを返す。
    """
    # rno・jcd を必ずゼロパディング（公式サイト要件）
    rno = str(rno).zfill(2)
    jcd = str(jcd).zfill(2)
    url = (
        f"https://www.boatrace.jp/owpc/pc/race/beforeinfo"
        f"?rno={rno}&jcd={jcd}"
    )
    html = safe_request(url)
    if html is None:
        return pd.DataFrame()

    soup = BeautifulSoup(html, "lxml")
    boats = []

    for row in soup.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 4:
            continue
        try:
            tno = int(cols[0].text.strip())
            if not (1 <= tno <= 6):
                continue
            # 全カラムから展示タイム(6.0〜8.0)・ST(-0.1〜0.5)を探す
            ex_time = st_val = None
            for ci in range(1, len(cols)):
                txt = cols[ci].text.strip().replace("\n", "").replace(" ", "")
                try:
                    val = float(txt)
                    if 6.0 < val < 8.0 and ex_time is None:
                        ex_time = val
                    if -0.1 < val < 0.5 and st_val is None:
                        st_val = val
                except ValueError:
                    pass
            # 展示タイムだけあれば登録（STは後で補完）
            if ex_time is not None:
                boats.append({
                    "艇番": tno,
                    "展示タイム": ex_time,
                    "ST": st_val if st_val is not None else 0.15,  # ST未公開時はデフォルト
                })
        except Exception:
            continue

    if not boats:
        return pd.DataFrame()

    df = pd.DataFrame(boats).sort_values("艇番").reset_index(drop=True)
    df = add_features(df)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量を付与"""
    df = df.copy()
    df["コース"]   = df["艇番"]
    df["ST順位"]   = df["ST"].rank(method="min")
    df["展示順位"] = df["展示タイム"].rank(method="min")
    df["内枠"]     = (df["コース"] <= 2).astype(int)
    return df


# ─────────────────────────────────────────
# スクレイピング：3連単オッズ
# ─────────────────────────────────────────
def get_odds(jcd: str, rno: str) -> dict:
    """
    3連単オッズを取得。失敗時は空dictを返す。
    """
    rno = str(rno).zfill(2)
    jcd = str(jcd).zfill(2)
    url = (
        f"https://www.boatrace.jp/owpc/pc/race/odds3t"
        f"?rno={rno}&jcd={jcd}"
    )
    html = safe_request(url)
    if html is None:
        return {}

    soup = BeautifulSoup(html, "lxml")
    odds_dict = {}

    # パターン1: "1-2-3" 形式のセル
    for row in soup.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 2:
            continue
        try:
            combo_txt = cols[0].text.strip()
            if "-" in combo_txt:
                combo = tuple(map(int, combo_txt.split("-")))
                odds_dict[combo] = float(
                    cols[1].text.strip().replace(",", "")
                )
        except Exception:
            continue

    # パターン2: oddsPointクラスのセル
    if not odds_dict:
        cells = soup.find_all("td", class_=lambda c: c and "oddsPoint" in c)
        combos = list(itertools.permutations([1, 2, 3, 4, 5, 6], 3))
        for i, cell in enumerate(cells):
            if i >= len(combos):
                break
            try:
                odds_dict[combos[i]] = float(
                    cell.text.strip().replace(",", "")
                )
            except Exception:
                pass

    return odds_dict


# ─────────────────────────────────────────
# スクレイピング：レース結果（学習データ収集用）
# ─────────────────────────────────────────
def get_race_result(jcd: str, rno: str) -> list[dict]:
    url = (
        f"https://www.boatrace.jp/owpc/pc/race/raceresult"
        f"?rno={rno}&jcd={jcd}"
    )
    res = safe_request(url)
    if res is None:
        return []

    soup = BeautifulSoup(res, "lxml")
    data = []
    for row in soup.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 6:
            continue
        try:
            data.append({
                "艇番": int(cols[1].text.strip()),
                "着順": int(cols[0].text.strip()),
                "ST":   float(cols[3].text.strip()),
            })
        except Exception:
            continue
    return data


# ─────────────────────────────────────────
# 予測ロジック
# ─────────────────────────────────────────
def predict_probs_rule(df: pd.DataFrame) -> pd.DataFrame:
    """モデルなし：ルールベースで1着確率を計算"""
    score = (
        (7 - df["ST順位"]) * 3.0
        + (7 - df["展示順位"]) * 2.0
        + df["内枠"] * 1.5
    )
    df = df.copy()
    df["スコア"] = score.values
    df["確率"]   = softmax(score.values)
    return df


def predict_probs_model(df: pd.DataFrame, model) -> pd.DataFrame:
    """LightGBMモデルで1着確率を計算"""
    df = df.copy()
    raw = model.predict_proba(df[FEATURES])[:, 1]
    df["確率"] = softmax(raw)
    return df


# ─────────────────────────────────────────
# 3連単確率計算
# ─────────────────────────────────────────
def trifecta(df: pd.DataFrame) -> dict:
    """
    連鎖確率法で3連単の全120通りの確率を計算。
    df には "艇番" と "確率" 列が必要。
    """
    probs = {}
    boats = df["艇番"].values

    for combo in itertools.permutations(boats, 3):
        tmp = df.copy()

        p1 = tmp.loc[tmp["艇番"] == combo[0], "確率"].values[0]
        tmp = tmp[tmp["艇番"] != combo[0]]

        p2s = softmax(tmp["確率"].values)
        tmp = tmp.copy()
        tmp["p2"] = p2s
        p2 = tmp.loc[tmp["艇番"] == combo[1], "p2"].values[0]
        tmp = tmp[tmp["艇番"] != combo[1]]

        p3s = softmax(tmp["確率"].values)
        tmp = tmp.copy()
        tmp["p3"] = p3s
        p3 = tmp.loc[tmp["艇番"] == combo[2], "p3"].values[0]

        probs[combo] = p1 * p2 * p3

    return probs


# ─────────────────────────────────────────
# 買い目抽出
# ─────────────────────────────────────────
def get_bets(
    prob_dict: dict,
    odds_dict: dict,
    ev_thresh: float = 1.2,
    top_n: int = 5,
) -> pd.DataFrame:
    rows = []
    for combo, p in prob_dict.items():
        o = odds_dict.get(combo, 0)
        if o <= 0:
            continue
        ev = p * o
        if ev >= ev_thresh:
            rows.append({
                "買い目": f"{combo[0]}-{combo[1]}-{combo[2]}",
                "_combo": combo,
                "確率": p,
                "オッズ": o,
                "期待値": round(ev, 3),
            })

    if not rows:
        return pd.DataFrame(columns=["買い目", "確率", "オッズ", "期待値"])

    result = (
        pd.DataFrame(rows)
        .sort_values("期待値", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return result


# ─────────────────────────────────────────
# ケリー基準による資金配分
# ─────────────────────────────────────────
def allocate(bets: pd.DataFrame, budget: int = 10_000) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame(columns=["買い目", "投資額", "期待値"])

    stakes = [kelly(row["確率"], row["オッズ"]) for _, row in bets.iterrows()]
    total = sum(stakes)

    result = []
    for i, (_, row) in enumerate(bets.iterrows()):
        ratio  = stakes[i] / total if total > 0 else 0
        amount = int(budget * ratio / 100) * 100  # 100円単位
        result.append({
            "買い目": row["買い目"],
            "投資額": amount,
            "確率":   f"{row['確率']*100:.2f}%",
            "オッズ": row["オッズ"],
            "期待値": row["期待値"],
        })

    return pd.DataFrame(result)
