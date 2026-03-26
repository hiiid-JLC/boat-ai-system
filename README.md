# 🚤 ボートAI予想システム

LightGBM × ケリー基準による競艇3連単 期待値ベース予想ツール。

---

## ディレクトリ構成

```
boat-ai-system/
├── app/
│   ├── app.py          ← Streamlit アプリ（メイン画面）
│   └── utils.py        ← 共通ロジック（スクレイピング・予測・算出）
├── bot/
│   ├── main.py         ← 自動監視・LINE通知ボット
│   └── notifier.py     ← LINE Notify 送信モジュール
├── data/
│   └── race_data.csv   ← 学習データ（collect.py で収集）
├── model/
│   └── model.pkl       ← 学習済みモデル（train.py で生成）
├── collect.py          ← レース結果収集スクリプト
├── train.py            ← AI学習スクリプト
├── requirements.txt
└── README.md
```

---

## セットアップ

```bash
# 1. 依存パッケージインストール
pip install -r requirements.txt

# 2. データ収集（直近3日分・全場）
python collect.py --days 3

# 3. AI学習
python train.py

# 4. アプリ起動
streamlit run app/app.py
```

---

## 使い方

### Streamlit アプリ（手動予想）

| タブ | 機能 |
|------|------|
| 🎯 単レース予想 | 競艇場・レース番号を選択して予想 |
| 📋 一括予想 | 複数レースIDをまとめて予想 |
| 📊 シミュレーション | CSVアップロードで過去データ検証 |
| ℹ️ 場コード | 全24場のコード一覧 |

### 自動通知ボット

```bash
# LINE トークンを環境変数に設定
export LINE_NOTIFY_TOKEN=your_token_here

# 監視開始（0101=桐生1R, 0202=戸田2R）
python bot/main.py --races 0101,0202 --ev 1.2 --budget 10000
```

---

## レースID 形式

`JJRR`：場コード2桁 + レース番号2桁  
例: `0101` = 桐生1R, `1212` = 住之江12R

---

## 学習データの CSV 形式

| カラム | 説明 |
|--------|------|
| レースID | 一意のID |
| 艇番 | 1〜6 |
| コース | 1〜6 |
| ST | スタートタイミング |
| 展示タイム | 展示タイム（秒） |
| 着順 | 1〜6 |
| モーター勝率 | （任意）省略時は 40.0 |
| 全国勝率 | （任意）省略時は 6.0 |

---

## ⚠️ 注意事項

- 直前情報は発走約30分前に公開されます
- 本ツールは予想補助目的です。投票判断はご自身の責任で行ってください
- 公式サイトへの過度なアクセスはお控えください（`collect.py` は1秒スリープ済み）
