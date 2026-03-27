"""
app/db.py ─ データ蓄積・GitHub自動コミット・モデル再学習
"""
import os
import io
import sys
import base64
import json
import time
import itertools
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────
# 定数
# ─────────────────────────────────────────
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BoatAI/2.0)"}

VENUES = {
    "01":"桐生","02":"戸田","03":"江戸川","04":"平和島","05":"多摩川",
    "06":"浜名湖","07":"蒲郡","08":"常滑","09":"津","10":"三国",
    "11":"びわこ","12":"住之江","13":"尼崎","14":"鳴門","15":"丸亀",
    "16":"児島","17":"宮島","18":"徳山","19":"下関","20":"若松",
    "21":"芦屋","22":"福岡","23":"唐津","24":"大村",
}

# GitHubリポジトリ設定（Streamlit Secretsから取得）
GH_API = "https://api.github.com"
DATA_FILE  = "data/race_data.csv"
STATS_FILE = "data/sim_stats.json"
MODEL_FILE = "model/model.pkl"


# ─────────────────────────────────────────
# GitHub API ヘルパー
# ─────────────────────────────────────────
def _gh_headers(token: str) -> dict:
    # 改行・スペース・非ASCII文字を除去（Secrets貼り付け時の混入対策）
    clean_token = str(token).strip().encode("ascii", errors="ignore").decode("ascii")
    return {
        "Authorization": f"token {clean_token}",
        "Accept": "application/vnd.github.v3+json",
    }


def gh_get_file(token: str, repo: str, path: str) -> tuple[str | None, str | None]:
    """ファイル内容とSHAを取得。存在しない場合は(None, None)"""
    url = f"{GH_API}/repos/{repo}/contents/{path}"
    res = requests.get(url, headers=_gh_headers(token), timeout=15)
    if res.status_code == 404:
        return None, None
    res.raise_for_status()
    data = res.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]


def gh_put_file(
    token: str, repo: str, path: str,
    content: str, message: str, sha: str | None = None
) -> bool:
    """ファイルをコミット（新規 or 更新）"""
    url = f"{GH_API}/repos/{repo}/contents/{path}"
    body = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
    }
    if sha:
        body["sha"] = sha
    res = requests.put(url, headers=_gh_headers(token), json=body, timeout=20)
    return res.status_code in (200, 201)


def gh_put_binary(
    token: str, repo: str, path: str,
    data: bytes, message: str, sha: str | None = None
) -> bool:
    """バイナリファイル（model.pkl等）をコミット"""
    url = f"{GH_API}/repos/{repo}/contents/{path}"
    body = {
        "message": message,
        "content": base64.b64encode(data).decode("utf-8"),
    }
    if sha:
        body["sha"] = sha
    res = requests.put(url, headers=_gh_headers(token), json=body, timeout=30)
    return res.status_code in (200, 201)


# ─────────────────────────────────────────
# スクレイピング：レース結果
# ─────────────────────────────────────────
def _safe_get(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


def scrape_day_results(date_str: str, jcd_list: list[str]) -> list[dict]:
    """
    指定日・指定場のレース結果をスクレイピング。
    date_str: "YYYYMMDD"
    """
    all_rows = []
    for jcd in jcd_list:
        for rno in range(1, 13):
            url = (
                f"https://www.boatrace.jp/owpc/pc/race/raceresult"
                f"?rno={rno:02d}&jcd={jcd}&hd={date_str}"
            )
            html = _safe_get(url)
            if html is None:
                continue

            soup = BeautifulSoup(html, "lxml")

            # 展示タイム取得（直前情報ページから）
            ex_url = (
                f"https://www.boatrace.jp/owpc/pc/race/beforeinfo"
                f"?rno={rno:02d}&jcd={jcd}&hd={date_str}"
            )
            ex_html = _safe_get(ex_url)
            ex_times = {}
            st_vals  = {}
            if ex_html:
                ex_soup = BeautifulSoup(ex_html, "lxml")
                for row in ex_soup.find_all("tr"):
                    cols = row.find_all("td")
                    if len(cols) < 6:
                        continue
                    try:
                        tno = int(cols[0].text.strip())
                        if not (1 <= tno <= 6):
                            continue
                        ex_t = st_v = None
                        for ci in range(2, min(len(cols), 9)):
                            txt = cols[ci].text.strip()
                            try:
                                val = float(txt)
                                if 6.0 < val < 8.0 and ex_t is None:
                                    ex_t = val
                                if -0.1 < val < 0.5 and st_v is None:
                                    st_v = val
                            except ValueError:
                                pass
                        if ex_t:
                            ex_times[tno] = ex_t
                        if st_v is not None:
                            st_vals[tno] = st_v
                    except Exception:
                        pass

            # 結果テーブルパース
            for row in soup.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) < 4:
                    continue
                try:
                    chakujun = int(cols[0].text.strip())
                    tno      = int(cols[1].text.strip())
                    if not (1 <= chakujun <= 6 and 1 <= tno <= 6):
                        continue
                    all_rows.append({
                        "レースID":   f"{date_str}_{jcd}_{rno:02d}",
                        "日付":       date_str,
                        "場コード":   jcd,
                        "場名":       VENUES.get(jcd, "?"),
                        "レース番号": rno,
                        "艇番":       tno,
                        "コース":     tno,
                        "着順":       chakujun,
                        "ST":         st_vals.get(tno, np.nan),
                        "展示タイム": ex_times.get(tno, np.nan),
                        "モーター勝率": 40.0,
                        "全国勝率":    6.0,
                    })
                except Exception:
                    continue

            time.sleep(0.8)  # サーバー負荷軽減

    return all_rows


def scrape_recent(days: int = 7, progress_cb=None) -> pd.DataFrame:
    """
    直近 days 日分の全場レース結果を収集。
    progress_cb: (current, total, message) を受け取るコールバック
    """
    jcd_list = list(VENUES.keys())
    dates = [
        (datetime.today() - timedelta(days=i)).strftime("%Y%m%d")
        for i in range(1, days + 1)
    ]

    all_rows = []
    total = len(dates)
    for i, d in enumerate(dates):
        if progress_cb:
            progress_cb(i, total, f"{d} スクレイピング中… ({i+1}/{total}日)")
        rows = scrape_day_results(d, jcd_list)
        all_rows.extend(rows)

    if progress_cb:
        progress_cb(total, total, "完了")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["レースID", "艇番"])
    return df


# ─────────────────────────────────────────
# データ統合（既存CSV + 新規）
# ─────────────────────────────────────────
def merge_data(existing_csv: str | None, new_df: pd.DataFrame) -> pd.DataFrame:
    """既存CSVと新規データをマージして重複除去"""
    if existing_csv:
        try:
            old_df = pd.read_csv(io.StringIO(existing_csv))
            merged = pd.concat([old_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["レースID", "艇番"])
            return merged
        except Exception:
            pass
    return new_df


# ─────────────────────────────────────────
# シミュレーション統計の蓄積
# ─────────────────────────────────────────
def load_stats(token: str, repo: str) -> dict:
    """GitHub から統計JSONを取得"""
    _empty = {"history": [], "total_bet": 0, "total_return": 0, "hits": 0, "bets": 0}
    if not token or not repo:
        return _empty
    try:
        content, _ = gh_get_file(token, repo, STATS_FILE)
        if content is None:
            return _empty
        return json.loads(content)
    except Exception:
        return _empty


def save_stats(token: str, repo: str, stats: dict) -> bool:
    """統計JSONをGitHubにコミット"""
    _, sha = gh_get_file(token, repo, STATS_FILE)
    content = json.dumps(stats, ensure_ascii=False, indent=2)
    return gh_put_file(
        token, repo, STATS_FILE, content,
        f"📊 シミュレーション統計更新 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        sha,
    )


def append_sim_result(stats: dict, result: dict) -> dict:
    """シミュレーション結果を統計に追記"""
    stats["history"].append({
        "日時":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "回収率":  result["roi"],
        "的中率":  result["win_rate"],
        "ベット数": result["bets"],
        "レース数": result["races"],
        "EV閾値":  result["ev_thresh"],
    })
    stats["total_bet"]    += result.get("total_bet", 0)
    stats["total_return"] += result.get("total_return", 0)
    stats["hits"]         += result.get("hits", 0)
    stats["bets"]         += result.get("bets", 0)
    # 直近50件のみ保持
    stats["history"] = stats["history"][-50:]
    return stats


# ─────────────────────────────────────────
# LightGBM 再学習
# ─────────────────────────────────────────
def retrain_model(df: pd.DataFrame):
    """
    DataFrameからLightGBMを再学習して返す。
    データ不足時はNoneを返す。
    """
    try:
        import lightgbm as lgb
        from sklearn.model_selection import GroupKFold

        FEATURES = ["コース", "ST順位", "展示順位", "内枠"]

        df = df.copy()
        df["ST順位"]   = df.groupby("レースID")["ST"].rank(method="min")
        df["展示順位"] = df.groupby("レースID")["展示タイム"].rank(method="min")
        df["内枠"]     = (df["コース"] <= 2).astype(int)
        df = df.dropna(subset=FEATURES + ["着順"])

        if len(df) < 60 or df["レースID"].nunique() < 10:
            return None, "データが少なすぎます（最低10レース必要）"

        X = df[FEATURES]
        y = (df["着順"] == 1).astype(int)

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X, y)
        return model, f"学習完了（{df['レースID'].nunique()}レース / {len(df)}行）"

    except ImportError:
        return None, "lightgbm がインストールされていません"
    except Exception as e:
        return None, f"学習エラー: {e}"


def save_model_to_github(token: str, repo: str, model) -> bool:
    """学習済みモデルをGitHubにコミット"""
    import io as _io
    import joblib

    buf = _io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    model_bytes = buf.read()

    _, sha = gh_get_file(token, repo, MODEL_FILE)
    return gh_put_binary(
        token, repo, MODEL_FILE, model_bytes,
        f"🤖 モデル更新 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        sha,
    )


def load_model_from_github(token: str, repo: str):
    """GitHubからモデルをダウンロードして返す"""
    import io as _io
    import joblib

    url = f"{GH_API}/repos/{repo}/contents/{MODEL_FILE}"
    res = requests.get(url, headers=_gh_headers(token), timeout=20)
    if res.status_code == 404:
        return None
    res.raise_for_status()
    data = res.json()
    model_bytes = base64.b64decode(data["content"])
    buf = _io.BytesIO(model_bytes)
    return joblib.load(buf)
