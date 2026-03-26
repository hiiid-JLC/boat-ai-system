"""
app/app.py ─ ボートAI予想 Streamlit アプリ
起動: streamlit run app/app.py
"""
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# utils参照
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    VENUES, FEATURES,
    get_data, get_odds,
    predict_probs_model, predict_probs_rule,
    trifecta, get_bets, allocate, softmax,
)

# ─────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ボートAI予想",
    page_icon="🚤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;700;900&family=Orbitron:wght@700;900&display=swap');

.stApp { background: linear-gradient(135deg,#0a0e1a,#0d1b2e,#0a1628); }

.main-title {
    font-family:'Orbitron',sans-serif; font-size:2.6rem; font-weight:900;
    background:linear-gradient(90deg,#00d4ff,#00ffcc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.subtitle { color:#4a9abb; font-size:.85rem; letter-spacing:.15em; text-transform:uppercase; }

.card {
    background:rgba(255,255,255,.04); border:1px solid rgba(0,212,255,.15);
    border-radius:12px; padding:20px; margin:10px 0;
}
.card-header {
    font-family:'Orbitron',sans-serif; font-size:.8rem; color:#00d4ff;
    letter-spacing:.2em; text-transform:uppercase;
    border-bottom:1px solid rgba(0,212,255,.2); padding-bottom:8px; margin-bottom:14px;
}
.metric-box {
    background:rgba(0,212,255,.06); border:1px solid rgba(0,212,255,.2);
    border-radius:10px; padding:16px; text-align:center;
}
.metric-value { font-family:'Orbitron',sans-serif; font-size:1.8rem; font-weight:700; color:#00d4ff; }
.metric-label { font-size:.75rem; color:#6699bb; margin-top:4px; }

.bet-row {
    background:rgba(255,255,255,.03); border:1px solid rgba(0,212,255,.12);
    border-radius:10px; padding:14px 18px; margin:8px 0;
}
.ev-bar-wrap { background:rgba(255,255,255,.05); border-radius:4px; height:5px; margin-top:6px; }
.ev-bar      { background:linear-gradient(90deg,#00d4ff,#00ffcc); border-radius:4px; height:5px; }

.badge { display:inline-flex; align-items:center; justify-content:center;
         width:28px; height:28px; border-radius:50%; font-weight:900; font-size:.9rem; margin-right:4px; }
.b1{background:#fff;color:#000} .b2{background:#1a1aff;color:#fff} .b3{background:#ff2222;color:#fff}
.b4{background:#999;color:#fff} .b5{background:#ffcc00;color:#000} .b6{background:#00aa44;color:#fff}

.ok  { background:rgba(0,255,100,.1); border:1px solid rgba(0,255,100,.3);
       color:#00ff64; border-radius:6px; padding:8px 16px; font-size:.85rem; }
.err { background:rgba(255,50,50,.1);  border:1px solid rgba(255,50,50,.3);
       color:#ff5050; border-radius:6px; padding:8px 16px; font-size:.85rem; }
.warn{ background:rgba(255,200,0,.1);  border:1px solid rgba(255,200,0,.3);
       color:#ffcc00; border-radius:6px; padding:8px 16px; font-size:.85rem; }

.stButton>button {
    background:linear-gradient(135deg,#0077aa,#00aacc)!important;
    color:#fff!important; border:none!important; border-radius:8px!important;
    font-weight:700!important; padding:.6rem 2rem!important;
}
.stButton>button:hover {
    background:linear-gradient(135deg,#0099cc,#00ccee)!important;
    box-shadow:0 4px 20px rgba(0,212,255,.3)!important;
}
hr.div { border:none; border-top:1px solid rgba(0,212,255,.1); margin:18px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# モデル読み込み（キャッシュ）
# ─────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# ─────────────────────────────────────────
# ヘッダー
# ─────────────────────────────────────────
st.markdown('<h1 class="main-title">🚤 BOAT AI PREDICTOR</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">競艇 AI 予想システム ─ LightGBM × ケリー基準</p>', unsafe_allow_html=True)

if model:
    st.markdown('<span style="color:#00ff64;font-size:.85rem">✅ AI モデル読み込み済み</span>', unsafe_allow_html=True)
else:
    st.markdown('<span style="color:#ffcc00;font-size:.85rem">⚠️ model.pkl なし → ルールベース予測を使用</span>', unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)

# ─────────────────────────────────────────
# タブ
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎯 単レース予想", "📋 一括予想", "📊 シミュレーション", "ℹ️ 場コード"])


# ══════════════════════════════════════════════
# TAB 1: 単レース予想
# ══════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown('<div class="card"><div class="card-header">レース入力</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            jcd = st.selectbox(
                "競艇場",
                list(VENUES.keys()),
                format_func=lambda k: f"{k}：{VENUES[k]}",
            )
        with c2:
            rno = st.selectbox("レース番号", [str(i) for i in range(1, 13)])

        c3, c4 = st.columns(2)
        with c3:
            ev_thresh = st.slider("期待値しきい値", 0.5, 2.5, 1.2, 0.05)
        with c4:
            budget = st.number_input("投資予算（円）", 1000, 100000, 10000, 1000)

        top_n = st.slider("表示買い目数", 3, 10, 5)
        run   = st.button("🚀 予想を実行", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
<div class="card">
<div class="card-header">システム概要</div>
<p style="color:#88aacc;font-size:.83rem;line-height:1.9">
<b style="color:#00d4ff">① データ取得</b><br>
　公式サイトから展示タイム・ST を取得<br>
<b style="color:#00d4ff">② AI予測</b><br>
　LightGBM で1着確率を計算<br>
<b style="color:#00d4ff">③ 期待値算出</b><br>
　確率 × オッズ ＞ しきい値を抽出<br>
<b style="color:#00d4ff">④ ケリー基準</b><br>
　確率・オッズから最適投資額を算出<br>
</p>
</div>""", unsafe_allow_html=True)

    if run:
        with st.spinner("データ取得中…"):
            df = get_data(jcd, rno)
            time.sleep(0.3)

        if df.empty:
            st.markdown(
                f'<div class="err">❌ データ取得失敗。直前情報未公開か構造変更の可能性があります。</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ok">✅ {VENUES[jcd]}場 第{rno}R のデータを取得しました</div>',
                unsafe_allow_html=True,
            )

            # 確率計算
            if model:
                df = predict_probs_model(df, model)
            else:
                df = predict_probs_rule(df)

            # ── 直前情報テーブル ──────────────────
            st.markdown('<div class="card"><div class="card-header">直前情報 & 予測スコア</div>', unsafe_allow_html=True)
            disp = df[["艇番","展示タイム","ST","内枠","確率"]].copy()
            disp["確率"] = (disp["確率"]*100).round(2).astype(str) + "%"
            disp["内枠"] = disp["内枠"].map({1:"◎", 0:""})
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── オッズ取得 ────────────────────────
            with st.spinner("オッズ取得中…"):
                odds = get_odds(jcd, rno)

            odds_status = "オッズ取得済み" if odds else "オッズ取得失敗（期待値計算不可）"
            cls = "ok" if odds else "warn"
            st.markdown(f'<div class="{cls}">{odds_status}</div>', unsafe_allow_html=True)

            if odds:
                prob_dict = trifecta(df)
                bets      = get_bets(prob_dict, odds, ev_thresh=ev_thresh, top_n=top_n)
                alloc     = allocate(bets, budget=budget)

                # ── 買い目カード ──────────────────
                st.markdown('<div class="card"><div class="card-header">推奨買い目 & ケリー基準投資額</div>', unsafe_allow_html=True)

                if alloc.empty:
                    st.warning("期待値がしきい値を超える買い目がありません。しきい値を下げてください。")
                else:
                    rank_icons = ["🥇","🥈","🥉"] + ["　"]*10
                    total_invest = alloc["投資額"].sum()

                    for i, row in alloc.iterrows():
                        ev   = float(row["期待値"])
                        parts = str(row["買い目"]).split("-")
                        badges = "".join(f'<span class="badge b{p}">{p}</span>' for p in parts)
                        bar_w  = min(int(ev / 3.0 * 100), 100)
                        ev_col = "#00d4ff" if ev >= 1.5 else ("#ffcc44" if ev >= 1.2 else "#ff8844")

                        st.markdown(f"""
<div class="bet-row" style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
  <span style="font-size:1.4rem">{rank_icons[i]}</span>
  {badges}
  <span style="font-family:'Orbitron',sans-serif;color:#e0f4ff;font-weight:700;font-size:1rem;letter-spacing:.05em">{row['買い目']}</span>
  <span style="margin-left:auto;text-align:right;white-space:nowrap">
    <span style="color:#888;font-size:.78rem">確率 </span><span style="color:#00ffcc;font-weight:700">{row['確率']}</span>
    &nbsp;&nbsp;
    <span style="color:#888;font-size:.78rem">オッズ </span><span style="color:#ffcc44;font-weight:700">{row['オッズ']:.1f}倍</span>
    &nbsp;&nbsp;
    <span style="color:#888;font-size:.78rem">EV </span><span style="color:{ev_col};font-weight:900;font-size:1.05rem">{ev:.3f}</span>
    &nbsp;&nbsp;
    <span style="color:#888;font-size:.78rem">投資 </span><span style="color:#fff;font-weight:700">¥{row['投資額']:,}</span>
  </span>
</div>
<div class="ev-bar-wrap"><div class="ev-bar" style="width:{bar_w}%"></div></div>
""", unsafe_allow_html=True)

                    st.markdown(
                        f'<p style="text-align:right;color:#00d4ff;font-family:Orbitron,sans-serif;'
                        f'font-size:.9rem;margin-top:12px">合計投資額: ¥{total_invest:,}</p>',
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2: 一括予想
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="card"><div class="card-header">複数レース一括予想</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#88aacc;font-size:.83rem">レースIDを「場コード(2桁)+レース番号(2桁)」でカンマ区切り入力 (例: 0101,0202,1203)</p>', unsafe_allow_html=True)

    race_ids_input = st.text_input("レースID一覧", placeholder="0101,0202,1203")
    b2_ev  = st.slider("期待値しきい値 (一括)", 0.5, 2.5, 1.2, 0.05, key="b2ev")
    b2_bud = st.number_input("1レース予算（円）", 1000, 100000, 10000, 1000, key="b2bud")
    run_bulk = st.button("📋 一括予想を実行", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run_bulk and race_ids_input:
        ids = [r.strip() for r in race_ids_input.split(",") if len(r.strip()) == 4]

        if not ids:
            st.error("有効なレースIDがありません。")
        else:
            for rid in ids:
                jcd_b, rno_b = rid[:2], rid[2:]
                label = f"{VENUES.get(jcd_b,'?')}場 第{rno_b}R ({rid})"
                with st.expander(f"📌 {label}", expanded=True):
                    with st.spinner("取得中…"):
                        df_b = get_data(jcd_b, rno_b)
                        time.sleep(0.3)

                    if df_b.empty:
                        st.error("データ取得失敗")
                        continue

                    if model:
                        df_b = predict_probs_model(df_b, model)
                    else:
                        df_b = predict_probs_rule(df_b)

                    odds_b = get_odds(jcd_b, rno_b)
                    if not odds_b:
                        st.warning("オッズ取得失敗")
                        continue

                    prob_b  = trifecta(df_b)
                    bets_b  = get_bets(prob_b, odds_b, ev_thresh=b2_ev)
                    alloc_b = allocate(bets_b, budget=b2_bud)

                    if alloc_b.empty:
                        st.info("推奨買い目なし")
                    else:
                        st.dataframe(alloc_b, use_container_width=True, hide_index=True)
                        st.markdown(
                            f'<p style="color:#00d4ff;font-size:.85rem">合計: ¥{alloc_b["投資額"].sum():,}</p>',
                            unsafe_allow_html=True,
                        )


# ══════════════════════════════════════════════
# TAB 3: シミュレーション
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="card"><div class="card-header">回収率シミュレーション</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#88aacc;font-size:.83rem">アップロードした race_data.csv（学習データ）を使って過去データで回収率を検証します。</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("race_data.csv をアップロード", type="csv")
    s_ev     = st.slider("期待値しきい値", 0.5, 2.5, 1.2, 0.05, key="sev")
    s_top    = st.slider("1レースの最大買い目数", 1, 5, 3, key="stop")
    run_sim  = st.button("📊 シミュレーション実行", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run_sim:
        if uploaded is None:
            st.error("CSVファイルをアップロードしてください。")
        else:
            df_sim = pd.read_csv(uploaded)

            required_cols = ["レースID","艇番","コース","ST","展示タイム","着順"]
            missing = [c for c in required_cols if c not in df_sim.columns]

            if missing:
                st.error(f"以下のカラムが不足しています: {missing}")
            else:
                # 特徴量生成
                df_sim["ST順位"]   = df_sim.groupby("レースID")["ST"].rank(method="min")
                df_sim["展示順位"] = df_sim.groupby("レースID")["展示タイム"].rank(method="min")
                df_sim["内枠"]     = (df_sim["コース"] <= 2).astype(int)
                if "モーター勝率" not in df_sim.columns: df_sim["モーター勝率"] = 40.0
                if "全国勝率"    not in df_sim.columns: df_sim["全国勝率"]    = 6.0

                total_bet = total_ret = hits = bets_cnt = 0
                race_ids_sim = df_sim["レースID"].unique()

                prog = st.progress(0)
                for i, rid in enumerate(race_ids_sim):
                    race = df_sim[df_sim["レースID"] == rid].copy()
                    if len(race) < 6:
                        continue

                    if model:
                        raw = model.predict_proba(race[FEATURES])[:, 1]
                        race["確率"] = softmax(raw)
                    else:
                        score = (7 - race["ST順位"]) * 3.0 + (7 - race["展示順位"]) * 2.0 + race["内枠"] * 1.5
                        race["確率"] = softmax(score.values)

                    prob_d = trifecta(race)

                    # シミュレーション用：ランダムオッズ（実オッズがないため）
                    import itertools
                    odds_d = {c: round(np.random.uniform(5, 80), 1)
                              for c in itertools.permutations(race["艇番"].values, 3)}

                    bets_d = get_bets(prob_d, odds_d, ev_thresh=s_ev, top_n=s_top)
                    result = tuple(race.sort_values("着順")["艇番"].values[:3])

                    for _, row in bets_d.iterrows():
                        parts = str(row["買い目"]).split("-")
                        combo = tuple(int(p) for p in parts)
                        total_bet += 100
                        bets_cnt  += 1
                        if combo == result:
                            total_ret += odds_d.get(combo, 0) * 100
                            hits      += 1

                    prog.progress((i + 1) / len(race_ids_sim))

                roi      = total_ret / total_bet if total_bet > 0 else 0
                win_rate = hits / bets_cnt if bets_cnt > 0 else 0

                m1, m2, m3, m4 = st.columns(4)
                for col, lbl, val, c in [
                    (m1, "回収率",   f"{roi*100:.1f}%",   "#00ff64" if roi >= 1.0 else "#ff5050"),
                    (m2, "的中率",   f"{win_rate*100:.2f}%", "#00d4ff"),
                    (m3, "総ベット数", f"{bets_cnt:,}",    "#00d4ff"),
                    (m4, "純損益",   f"¥{int(total_ret-total_bet):,}", "#00ff64" if total_ret >= total_bet else "#ff5050"),
                ]:
                    with col:
                        st.markdown(f"""
<div class="metric-box">
  <div class="metric-value" style="color:{c}">{val}</div>
  <div class="metric-label">{lbl}</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4: 場コード一覧
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="card"><div class="card-header">競艇場コード一覧</div>', unsafe_allow_html=True)
    cols_v = st.columns(6)
    for i, (code, name) in enumerate(VENUES.items()):
        with cols_v[i % 6]:
            st.markdown(f"""
<div style="background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.18);
            border-radius:8px;padding:10px;margin:4px 0;text-align:center">
  <div style="font-family:'Orbitron',sans-serif;color:#00d4ff;font-size:.9rem;font-weight:700">{code}</div>
  <div style="color:#c0dde8;font-size:.85rem;margin-top:4px">{name}</div>
</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# フッター
# ─────────────────────────────────────────
st.markdown("""
<hr class="div">
<p style="text-align:center;color:#2a4a5a;font-size:.72rem">
  ⚠️ 本ツールは予想補助目的の参考情報です。投票判断はご自身の責任で行ってください。
</p>
""", unsafe_allow_html=True)
