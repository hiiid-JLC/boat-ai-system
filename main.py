"""
bot/main.py ─ 自動監視・LINE通知ボット
使い方:
    LINE_NOTIFY_TOKEN=xxxx python bot/main.py --races 0101,0102,0103
"""
import os
import sys
import argparse
import time

# パス設定（appフォルダの utils を参照）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import joblib
import pandas as pd
from app.utils import get_data, predict_probs_model, predict_probs_rule
from app.utils import trifecta, get_bets, get_odds, allocate, FEATURES
from bot.notifier import send_line, format_bets

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")


def load_model():
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] モデル読み込み: {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    print("[WARN] model.pkl が見つかりません。ルールベース予測を使用します。")
    return None


def predict_race(jcd: str, rno: str, model, ev_thresh: float, budget: int) -> pd.DataFrame | None:
    df = get_data(jcd, rno)
    if df.empty:
        return None

    # 全艇の展示タイムが揃っているか確認
    if df["展示タイム"].isnull().any():
        return None

    # 確率計算
    if model:
        df = predict_probs_model(df, model)
    else:
        df = predict_probs_rule(df)

    prob_dict = trifecta(df)
    odds_dict = get_odds(jcd, rno)

    if not odds_dict:
        print(f"  [WARN] {jcd}場{rno}R オッズ取得失敗")

    bets = get_bets(prob_dict, odds_dict, ev_thresh=ev_thresh)
    if bets.empty:
        return None

    return allocate(bets, budget=budget)


def monitor(race_list: list[tuple], ev_thresh: float, budget: int, interval: int):
    """
    race_list: [(jcd, rno), ...] のリスト
    interval:  チェック間隔（秒）
    """
    model  = load_model()
    sent   = set()

    print(f"\n[INFO] 監視開始 ({len(race_list)} レース, {interval}秒間隔)")
    print("  停止するには Ctrl+C\n")

    while True:
        for jcd, rno in race_list:
            race_key = f"{jcd}_{rno}"
            if race_key in sent:
                continue

            print(f"  チェック: {jcd}場 {rno}R")
            alloc = predict_race(jcd, rno, model, ev_thresh, budget)

            if alloc is not None:
                label  = f"{jcd}場 第{rno}R"
                msg    = format_bets(label, alloc)
                result = send_line(msg)
                status = "✅ 送信完了" if result else "⚠️ 送信失敗（ログ確認）"
                print(f"  {label}: {status}")
                print(msg)
                sent.add(race_key)

        if len(sent) == len(race_list):
            print("\n[INFO] 全レースの通知が完了しました。")
            break

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="ボートAI 自動通知ボット")
    parser.add_argument("--races",    type=str, default="",    help="レースID カンマ区切り (例: 0101,0202)")
    parser.add_argument("--ev",       type=float, default=1.2, help="期待値しきい値")
    parser.add_argument("--budget",   type=int, default=10000, help="1レース投資予算（円）")
    parser.add_argument("--interval", type=int, default=60,    help="チェック間隔（秒）")
    args = parser.parse_args()

    if not args.races:
        print("[ERROR] --races を指定してください (例: --races 0101,0102)")
        sys.exit(1)

    # "JJRR" 形式のレースIDをパース
    race_list = []
    for r in args.races.split(","):
        r = r.strip()
        if len(r) == 4:
            race_list.append((r[:2], r[2:]))
        else:
            print(f"[WARN] 不正なレースID: {r}（4桁で指定してください）")

    if not race_list:
        print("[ERROR] 有効なレースIDがありません。")
        sys.exit(1)

    monitor(race_list, args.ev, args.budget, args.interval)


if __name__ == "__main__":
    main()
