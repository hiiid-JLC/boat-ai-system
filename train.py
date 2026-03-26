"""
train.py ─ LightGBMモデルの学習スクリプト
使い方: python train.py
"""
import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

# パス設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "race_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

FEATURES = ["コース", "ST順位", "展示順位", "内枠", "モーター勝率", "全国勝率"]

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 必須カラムチェック
    required = ["レースID", "艇番", "コース", "ST", "展示タイム", "着順"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] 以下のカラムが不足しています: {missing}")
        sys.exit(1)

    # 特徴量生成
    df["ST順位"]   = df.groupby("レースID")["ST"].rank(method="min")
    df["展示順位"] = df.groupby("レースID")["展示タイム"].rank(method="min")
    df["内枠"]     = (df["コース"] <= 2).astype(int)

    # オプション列がなければデフォルト値
    if "モーター勝率" not in df.columns:
        df["モーター勝率"] = 40.0
    if "全国勝率" not in df.columns:
        df["全国勝率"] = 6.0

    df = df.dropna(subset=FEATURES + ["着順"])
    return df


def train(df: pd.DataFrame):
    X = df[FEATURES]
    y = (df["着順"] == 1).astype(int)
    groups = df["レースID"]

    # GroupKFold でレース単位の CV
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(df))

    models = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        m = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va_idx] = m.predict_proba(X_va)[:, 1]
        models.append(m)
        auc = roc_auc_score(y_va, oof[va_idx])
        print(f"  Fold {fold+1} AUC: {auc:.4f}")

    oof_auc = roc_auc_score(y, oof)
    print(f"\n全体 OOF AUC: {oof_auc:.4f}")

    # 全データで最終モデル学習
    final = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    final.fit(X, y)

    # 特徴量重要度
    imp = pd.Series(final.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n特徴量重要度:")
    print(imp.to_string())

    return final


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] データファイルが見つかりません: {DATA_PATH}")
        print("collect.py を先に実行してください。")
        sys.exit(0)  # exitコード0でエラー扱いしない

    print(f"データ読み込み: {DATA_PATH}")
    df = load_and_prepare(DATA_PATH)
    print(f"  レース数: {df['レースID'].nunique()}, 行数: {len(df)}")

    print("\n学習開始...")
    model = train(df)

    joblib.dump(model, MODEL_PATH)
    print(f"\nモデル保存: {MODEL_PATH}")
