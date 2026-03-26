"""
notifier.py ─ LINE Notify 送信モジュール
"""
import os
import requests
import pandas as pd

# 環境変数から取得（セキュリティのためコードに直書きしない）
LINE_TOKEN = os.environ.get("LINE_NOTIFY_TOKEN", "")


def send_line(message: str) -> bool:
    """
    LINE Notify にメッセージを送信。
    成功時 True、失敗時 False を返す。
    """
    if not LINE_TOKEN:
        print("[WARN] LINE_NOTIFY_TOKEN が設定されていません。")
        return False

    try:
        res = requests.post(
            "https://notify-api.line.me/api/notify",
            headers={"Authorization": f"Bearer {LINE_TOKEN}"},
            data={"message": message},
            timeout=10,
        )
        res.raise_for_status()
        return True
    except Exception as e:
        print(f"[ERROR] LINE通知失敗: {e}")
        return False


def format_bets(race_label: str, alloc_df: pd.DataFrame) -> str:
    """資金配分DataFrameを通知文字列に整形"""
    lines = [f"\n【AI予想】{race_label}"]
    lines.append("─" * 20)

    for _, row in alloc_df.iterrows():
        lines.append(
            f"  {row['買い目']}  ¥{row['投資額']:,}円"
            f"  (EV:{row['期待値']:.2f})"
        )

    total = alloc_df["投資額"].sum()
    lines.append("─" * 20)
    lines.append(f"  合計投資額: ¥{total:,}円")
    return "\n".join(lines)
