"""
app/app.py ─ ボートAI予想 Streamlit アプリ v2
起動: streamlit run app/app.py
"""
import os
import sys
import io
import time
import itertools
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    VENUES, FEATURES,
    get_data, get_odds,
    predict_probs_model, predict_probs_rule,
    trifecta, get_bets, allocate, softmax,
)
from db import (
    scrape_recent, merge_data,
    gh_get_file, gh_put_file,
    load_stats, save_stats, append_sim_result,
    retrain_model, save_model_to_github, load_model_from_github,
    DATA_FILE,
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
       color:#00ff64; border-radius:6px; padding:8px 16px; font-size:.85rem; margin:6px 0; display:block; }
.err { background:rgba(255,50,50,.1);  border:1px solid rgba(255,50,50,.3);
       color:#ff5050; border-radius:6px; padding:8px 16px; font-size:.85rem; margin:6px 0; display:block; }
.warn{ background:rgba(255,200,0,.1);  border:1px solid rgba(255,200,0,.3);
       color:#ffcc00; border-radius:6px; padding:8px 16px; font-size:.85rem; margin:6px 0; display:block; }
.info{ background:rgba(0,100,255,.1);  border:1px solid rgba(0,100,255,.3);
       color:#88aaff; border-radius:6px; padding:8px 16px; font-size:.85rem; margin:6px 0; display:block; }

.stButton>button {
    background:linear-gradient(135deg,#0077aa,#00aacc)!important;
    color:#fff!important; border:none!important; border-radius:8px!important;
    font-weight:700!important; padding:.6rem 1.5rem!important;
}
.stButton>button:hover {
    background:linear-gradient(135deg,#0099cc,#00ccee)!important;
    box-shadow:0 4px 20px rgba(0,212,255,.3)!important;
}
hr.div { border:none; border-top:1px solid rgba(0,212,255,.1); margin:18px 0; }

.step-badge {
    display:inline-flex; align-items:center; justify-content:center;
    width:22px; height:22px; border-radius:50%;
    background:rgba(0,212,255,.2); border:1px solid #00d4ff;
    color:#00d4ff; font-size:.72rem; font-weight:700; margin-right:8px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Secrets
# ─────────────────────────────────────────
def get_secrets():
    try:
        return st.secrets["GITHUB_TOKEN"], st.secrets["GITHUB_REPO"]
    except Exception:
        return None, None


# ─────────────────────────────────────────
# モデル（キャッシュ）
# ─────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

@st.cache_resource
def load_local_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


# ─────────────────────────────────────────
# セッション初期化
# ─────────────────────────────────────────
if "scraped_df"  not in st.session_state: st.session_state.scraped_df  = None
if "sim_result"  not in st.session_state: st.session_state.sim_result  = None
if "model"       not in st.session_state: st.session_state.model       = load_local_model()
if "stats"       not in st.session_state: st.session_state.stats       = None

token, repo = get_secrets()
model = st.session_state.model


# ─────────────────────────────────────────
# ヘッダー
# ─────────────────────────────────────────
st.markdown('<h1 class="main-title">🚤 BOAT AI PREDICTOR</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">競艇 AI 予想システム v2 ─ 自動収集 × 蓄積学習</p>', unsafe_allow_html=True)

hc1, hc2, hc3 = st.columns(3)
with hc1:
    st.markdown(f'<span class="{"ok" if model else "warn"}">{"✅ AIモデル読み込み済み" if model else "⚠️ モデルなし → ルールベース使用"}</span>', unsafe_allow_html=True)
with hc2:
    st.markdown(f'<span class="{"ok" if token else "warn"}">{"✅ GitHub連携済み" if token else "⚠️ GitHub未設定 → 蓄積不可"}</span>', unsafe_allow_html=True)
with hc3:
    rows = len(st.session_state.scraped_df) if st.session_state.scraped_df is not None else 0
    st.markdown(f'<span class="info">📦 セッション内データ: {rows}行</span>', unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)


# ─────────────────────────────────────────
# タブ
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 単レース予想", "📋 一括予想",
    "📊 データ収集 & シミュレーション",
    "🤖 モデル管理", "ℹ️ 場コード",
])


# ══════════════════════════════════════════════
# TAB 1: 単レース予想
# ══════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown('<div class="card"><div class="card-header">レース入力</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            jcd = st.selectbox("競艇場", list(VENUES.keys()), format_func=lambda k: f"{k}：{VENUES[k]}")
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
<div class="card"><div class="card-header">システム概要</div>
<p style="color:#88aacc;font-size:.83rem;line-height:1.9">
<b style="color:#00d4ff">① データ取得</b><br>　公式から展示タイム・ST取得<br>
<b style="color:#00d4ff">② AI予測</b><br>　LightGBMで1着確率を計算<br>
<b style="color:#00d4ff">③ 期待値算出</b><br>　確率×オッズ＞しきい値を抽出<br>
<b style="color:#00d4ff">④ ケリー基準</b><br>　最適投資額を自動算出<br>
</p></div>""", unsafe_allow_html=True)

    if run:
        with st.spinner("データ取得中…"):
            df_r = get_data(jcd, rno)
        if df_r.empty:
            st.markdown('<span class="err">❌ データ取得失敗。以下を確認してください。<br>・直前情報がまだ公開されていない（発走30分前ごろ公開）<br>・場コードまたはレース番号が間違っている<br>・公式サイトのHTML構造が変わった可能性あり</span>', unsafe_allow_html=True)
            # デバッグ情報
            import requests
            from bs4 import BeautifulSoup
            debug_url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}"
            try:
                res = requests.get(debug_url, headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
                soup = BeautifulSoup(res.text, "lxml")
                tds = soup.find_all("td")
                sample = [td.text.strip()[:15] for td in tds if td.text.strip()][:30]
                with st.expander("🔍 デバッグ情報（取得できたデータ）"):
                    st.write(f"ステータス: {res.status_code}")
                    st.write(f"tdセル数: {len(tds)}")
                    st.write(f"サンプル: {sample}")
            except Exception as e:
                st.error(f"通信エラー: {e}")
        else:
            st.markdown(f'<span class="ok">✅ {VENUES[jcd]}場 第{rno}R データ取得完了</span>', unsafe_allow_html=True)
            df_r = predict_probs_model(df_r, model) if model else predict_probs_rule(df_r)

            st.markdown('<div class="card"><div class="card-header">直前情報 & 予測スコア</div>', unsafe_allow_html=True)
            disp = df_r[["艇番","展示タイム","ST","内枠","確率"]].copy()
            disp["確率"] = (disp["確率"]*100).round(2).astype(str) + "%"
            disp["内枠"] = disp["内枠"].map({1:"◎", 0:""})
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            with st.spinner("オッズ取得中…"):
                odds = get_odds(jcd, rno)

            if odds:
                prob_dict = trifecta(df_r)
                bets      = get_bets(prob_dict, odds, ev_thresh=ev_thresh, top_n=top_n)
                alloc     = allocate(bets, budget=budget)

                st.markdown('<div class="card"><div class="card-header">推奨買い目 & ケリー基準投資額</div>', unsafe_allow_html=True)
                rank_icons = ["🥇","🥈","🥉"] + ["　"]*10
                if alloc.empty:
                    st.warning("期待値がしきい値を超える買い目がありません。")
                else:
                    for i, row in alloc.iterrows():
                        ev    = float(row["期待値"])
                        parts = str(row["買い目"]).split("-")
                        badges = "".join(f'<span class="badge b{p}">{p}</span>' for p in parts)
                        bar_w  = min(int(ev/3.0*100), 100)
                        ev_col = "#00d4ff" if ev>=1.5 else ("#ffcc44" if ev>=1.2 else "#ff8844")
                        st.markdown(f"""
<div class="bet-row" style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
  <span style="font-size:1.4rem">{rank_icons[i]}</span>{badges}
  <span style="font-family:'Orbitron',sans-serif;color:#e0f4ff;font-weight:700">{row['買い目']}</span>
  <span style="margin-left:auto;text-align:right;white-space:nowrap">
    <span style="color:#888;font-size:.78rem">確率 </span><span style="color:#00ffcc;font-weight:700">{row['確率']}</span>&nbsp;&nbsp;
    <span style="color:#888;font-size:.78rem">オッズ </span><span style="color:#ffcc44;font-weight:700">{row['オッズ']:.1f}倍</span>&nbsp;&nbsp;
    <span style="color:#888;font-size:.78rem">EV </span><span style="color:{ev_col};font-weight:900;font-size:1.05rem">{ev:.3f}</span>&nbsp;&nbsp;
    <span style="color:#888;font-size:.78rem">投資 </span><span style="color:#fff;font-weight:700">¥{row['投資額']:,}</span>
  </span>
</div>
<div class="ev-bar-wrap"><div class="ev-bar" style="width:{bar_w}%"></div></div>
""", unsafe_allow_html=True)
                    st.markdown(f'<p style="text-align:right;color:#00d4ff;font-family:Orbitron,sans-serif;font-size:.9rem;margin-top:12px">合計: ¥{alloc["投資額"].sum():,}</p>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<span class="warn">⚠️ オッズ取得失敗</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2: 一括予想
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="card"><div class="card-header">複数レース一括予想</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#88aacc;font-size:.83rem">レースIDを「場コード(2桁)+レース番号(2桁)」でカンマ区切り (例: 0101,0202,1203)</p>', unsafe_allow_html=True)
    race_ids_input = st.text_input("レースID一覧", placeholder="0101,0202,1203")
    b2_ev  = st.slider("期待値しきい値", 0.5, 2.5, 1.2, 0.05, key="b2ev")
    b2_bud = st.number_input("1レース予算（円）", 1000, 100000, 10000, 1000, key="b2bud")
    run_bulk = st.button("📋 一括予想を実行", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run_bulk and race_ids_input:
        ids = [r.strip() for r in race_ids_input.split(",") if len(r.strip()) == 4]
        for rid in ids:
            jcd_b, rno_b = rid[:2], rid[2:]
            with st.expander(f"📌 {VENUES.get(jcd_b,'?')}場 第{rno_b}R", expanded=True):
                with st.spinner("取得中…"):
                    df_b = get_data(jcd_b, rno_b)
                if df_b.empty:
                    st.error("データ取得失敗")
                    continue
                df_b = predict_probs_model(df_b, model) if model else predict_probs_rule(df_b)
                odds_b = get_odds(jcd_b, rno_b)
                if not odds_b:
                    st.warning("オッズ取得失敗")
                    continue
                bets_b  = get_bets(trifecta(df_b), odds_b, ev_thresh=b2_ev)
                alloc_b = allocate(bets_b, budget=b2_bud)
                if alloc_b.empty:
                    st.info("推奨買い目なし")
                else:
                    st.dataframe(alloc_b, use_container_width=True, hide_index=True)
                    st.markdown(f'<p style="color:#00d4ff;font-size:.85rem">合計: ¥{alloc_b["投資額"].sum():,}</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3: データ収集 & シミュレーション
# ══════════════════════════════════════════════
with tab3:

    # ────── STEP 1: 収集 ──────────────────────
    st.markdown('<div class="card"><div class="card-header"><span class="step-badge">1</span>最新データを自動収集</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        scrape_days = st.slider("収集する日数（直近）", 1, 14, 7)
        st.markdown('<p style="color:#88aacc;font-size:.8rem">⚠️ 日数が多いほど時間がかかります（1日あたり約2〜3分）</p>', unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
<div style="background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.15);border-radius:8px;padding:12px;margin-top:8px;text-align:center">
  <div style="color:#6699bb;font-size:.75rem">収集レース数の目安</div>
  <div style="color:#00d4ff;font-family:'Orbitron',sans-serif;font-size:1.4rem;font-weight:700">{scrape_days*24*12:,}</div>
  <div style="color:#6699bb;font-size:.72rem">最大（24場×12R×{scrape_days}日）</div>
</div>""", unsafe_allow_html=True)

    btn_scrape = st.button("🌐 データ収集を開始", use_container_width=True)

    if btn_scrape:
        prog_bar = st.progress(0)
        status   = st.empty()

        def progress_cb(cur, total, msg):
            prog_bar.progress(cur / max(total, 1))
            status.markdown(f'<span class="info">⏳ {msg}</span>', unsafe_allow_html=True)

        with st.spinner("スクレイピング中…"):
            new_df = scrape_recent(days=scrape_days, progress_cb=progress_cb)

        prog_bar.progress(1.0)

        if new_df.empty:
            status.markdown('<span class="err">❌ データ取得失敗。時間をおいて再試行してください。</span>', unsafe_allow_html=True)
        else:
            existing_csv = None
            if token and repo:
                existing_csv, _ = gh_get_file(token, repo, DATA_FILE)
            merged_df = merge_data(existing_csv, new_df)
            st.session_state.scraped_df = merged_df

            if token and repo:
                csv_str = merged_df.to_csv(index=False)
                _, sha  = gh_get_file(token, repo, DATA_FILE)
                ok = gh_put_file(token, repo, DATA_FILE, csv_str,
                                 f"📦 データ更新 直近{scrape_days}日分 ({len(merged_df)}行)", sha)
                cls = "ok" if ok else "warn"
                msg = f"✅ {len(new_df)}行取得 → 合計{len(merged_df)}行 → GitHub保存完了" if ok else f"⚠️ {len(new_df)}行取得済み。GitHub保存失敗"
            else:
                cls = "warn"
                msg = f"⚠️ {len(new_df)}行取得済み（GitHub未設定のためセッション内のみ）"
            status.markdown(f'<span class="{cls}">{msg}</span>', unsafe_allow_html=True)

    if token and repo and st.session_state.scraped_df is None:
        if st.button("📥 GitHubから既存データを読み込む"):
            with st.spinner("読み込み中…"):
                csv_content, _ = gh_get_file(token, repo, DATA_FILE)
            if csv_content:
                st.session_state.scraped_df = pd.read_csv(io.StringIO(csv_content))
                st.markdown(f'<span class="ok">✅ {len(st.session_state.scraped_df)}行 読み込み完了</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="warn">⚠️ GitHubにデータがありません。先に収集してください。</span>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ────── STEP 2: シミュレーション ──────────
    st.markdown('<div class="card"><div class="card-header"><span class="step-badge">2</span>シミュレーション実行</div>', unsafe_allow_html=True)

    df_for_sim = st.session_state.scraped_df
    uploaded   = st.file_uploader("または race_data.csv を手動アップロード", type="csv")
    if uploaded:
        df_for_sim = pd.read_csv(uploaded)
        st.session_state.scraped_df = df_for_sim

    if df_for_sim is not None:
        n_races_sim = df_for_sim["レースID"].nunique() if "レースID" in df_for_sim.columns else "?"
        st.markdown(f'<span class="info">📦 使用データ: {len(df_for_sim)}行 / {n_races_sim}レース</span>', unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1: s_ev  = st.slider("期待値しきい値", 0.5, 2.5, 1.2, 0.05, key="sev")
    with s2: s_top = st.slider("1レース最大買い目数", 1, 5, 3, key="stop")

    run_sim = st.button("📊 シミュレーション実行", use_container_width=True,
                        disabled=(df_for_sim is None))

    if run_sim and df_for_sim is not None:
        required = ["レースID","艇番","コース","ST","展示タイム","着順"]
        missing  = [c for c in required if c not in df_for_sim.columns]
        if missing:
            st.error(f"カラム不足: {missing}")
        else:
            df_s = df_for_sim.copy()
            df_s["ST順位"]   = df_s.groupby("レースID")["ST"].rank(method="min")
            df_s["展示順位"] = df_s.groupby("レースID")["展示タイム"].rank(method="min")
            df_s["内枠"]     = (df_s["コース"] <= 2).astype(int)
            if "モーター勝率" not in df_s.columns: df_s["モーター勝率"] = 40.0
            if "全国勝率"    not in df_s.columns: df_s["全国勝率"]    = 6.0

            total_bet = total_ret = hits = bets_cnt = 0
            race_ids_s = df_s["レースID"].unique()
            prog2 = st.progress(0)

            for i, rid in enumerate(race_ids_s):
                race = df_s[df_s["レースID"] == rid].copy()
                if len(race) < 6:
                    continue
                if model:
                    race["確率"] = softmax(model.predict_proba(race[FEATURES])[:, 1])
                else:
                    score = (7-race["ST順位"])*3.0 + (7-race["展示順位"])*2.0 + race["内枠"]*1.5
                    race["確率"] = softmax(score.values)

                prob_d = trifecta(race)
                odds_d = {c: round(np.random.uniform(5, 80), 1)
                          for c in itertools.permutations(race["艇番"].values, 3)}
                bets_d = get_bets(prob_d, odds_d, ev_thresh=s_ev, top_n=s_top)
                result = tuple(race.sort_values("着順")["艇番"].values[:3])

                for _, brow in bets_d.iterrows():
                    combo = tuple(int(p) for p in str(brow["買い目"]).split("-"))
                    total_bet += 100
                    bets_cnt  += 1
                    if combo == result:
                        total_ret += odds_d.get(combo, 0) * 100
                        hits      += 1

                prog2.progress((i+1)/len(race_ids_s))

            roi      = total_ret / total_bet if total_bet > 0 else 0
            win_rate = hits / bets_cnt if bets_cnt > 0 else 0
            sim_result = {"roi":roi,"win_rate":win_rate,"total_bet":total_bet,
                          "total_return":int(total_ret),"hits":hits,"bets":bets_cnt,
                          "races":len(race_ids_s),"ev_thresh":s_ev}
            st.session_state.sim_result = sim_result

            m1,m2,m3,m4 = st.columns(4)
            for col,lbl,val,c in [
                (m1,"回収率",    f"{roi*100:.1f}%",      "#00ff64" if roi>=1.0 else "#ff5050"),
                (m2,"的中率",    f"{win_rate*100:.2f}%", "#00d4ff"),
                (m3,"総ベット数", f"{bets_cnt:,}",        "#00d4ff"),
                (m4,"純損益",    f"¥{int(total_ret-total_bet):,}", "#00ff64" if total_ret>=total_bet else "#ff5050"),
            ]:
                with col:
                    st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{c}">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

            if token and repo:
                stats = load_stats(token, repo)
                stats = append_sim_result(stats, sim_result)
                save_stats(token, repo, stats)
                st.session_state.stats = stats
                st.markdown('<span class="ok">✅ 結果をGitHubに蓄積しました</span>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ────── STEP 3: 蓄積統計 ──────────────────
    st.markdown('<div class="card"><div class="card-header"><span class="step-badge">3</span>蓄積された統計</div>', unsafe_allow_html=True)

    if st.button("📈 統計を読み込む", disabled=(not token)):
        st.session_state.stats = load_stats(token, repo)

    stats = st.session_state.stats
    if stats and stats.get("history"):
        total_b = stats.get("total_bet", 0)
        total_r = stats.get("total_return", 0)
        cumroi  = total_r / total_b if total_b > 0 else 0
        cum_hr  = stats.get("hits", 0) / max(stats.get("bets", 1), 1)

        sm1,sm2,sm3,sm4 = st.columns(4)
        for col,lbl,val,c in [
            (sm1,"累計回収率", f"{cumroi*100:.1f}%", "#00ff64" if cumroi>=1.0 else "#ff5050"),
            (sm2,"累計的中率", f"{cum_hr*100:.2f}%", "#00d4ff"),
            (sm3,"総ベット額", f"¥{total_b:,}",      "#00d4ff"),
            (sm4,"累計純損益", f"¥{total_r-total_b:,}", "#00ff64" if total_r>=total_b else "#ff5050"),
        ]:
            with col:
                st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{c}">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

        hist_df = pd.DataFrame(stats["history"])
        st.dataframe(hist_df[["日時","回収率","的中率","ベット数","レース数","EV閾値"]].tail(10),
                     use_container_width=True, hide_index=True)
    else:
        st.markdown('<p style="color:#4a6a7a;font-size:.85rem">まだ統計がありません。シミュレーションを実行すると蓄積されます。</p>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4: モデル管理
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="card"><div class="card-header">🤖 AIモデル管理</div>', unsafe_allow_html=True)

    df_for_train = st.session_state.scraped_df
    if df_for_train is not None:
        n_r = df_for_train["レースID"].nunique() if "レースID" in df_for_train.columns else "?"
        st.markdown(f'<span class="info">📦 学習データ: {len(df_for_train)}行 / {n_r}レース</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="warn">⚠️ データがありません。先にデータ収集タブで取得してください。</span>', unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown('<p style="color:#00d4ff;font-size:.85rem;font-weight:700">🔄 モデルを再学習</p>', unsafe_allow_html=True)
        if st.button("🤖 再学習を実行", use_container_width=True, disabled=(df_for_train is None)):
            with st.spinner("LightGBM 学習中…"):
                new_model, msg = retrain_model(df_for_train)
            if new_model:
                st.session_state.model = new_model
                st.markdown(f'<span class="ok">✅ {msg}</span>', unsafe_allow_html=True)
                if token and repo:
                    with st.spinner("GitHubにアップロード中…"):
                        ok = save_model_to_github(token, repo, new_model)
                    cls = "ok" if ok else "warn"
                    st.markdown(f'<span class="{cls}">{"✅ model.pkl をGitHubに保存" if ok else "⚠️ GitHub保存失敗"}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="err">❌ {msg}</span>', unsafe_allow_html=True)

    with col_t2:
        st.markdown('<p style="color:#00d4ff;font-size:.85rem;font-weight:700">📥 GitHubから読み込む</p>', unsafe_allow_html=True)
        if st.button("📥 最新モデルを読み込む", use_container_width=True, disabled=(not token)):
            with st.spinner("ダウンロード中…"):
                loaded = load_model_from_github(token, repo)
            if loaded:
                st.session_state.model = loaded
                st.markdown('<span class="ok">✅ GitHubからモデルを読み込みました</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="warn">⚠️ GitHubにモデルがありません</span>', unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(0,212,255,.04);border:1px solid rgba(0,212,255,.15);border-radius:8px;padding:14px">
<p style="color:#00d4ff;font-weight:700;font-size:.85rem;margin:0 0 10px">精度向上サイクル</p>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">
  <div style="text-align:center;padding:10px;background:rgba(0,212,255,.06);border-radius:8px">
    <div style="font-size:1.6rem">🌐</div>
    <div style="color:#00d4ff;font-size:.78rem;font-weight:700;margin-top:4px">① 収集</div>
    <div style="color:#6699bb;font-size:.7rem;margin-top:2px">直近データを<br>スクレイピング</div>
  </div>
  <div style="text-align:center;padding:10px;background:rgba(0,212,255,.06);border-radius:8px">
    <div style="font-size:1.6rem">📊</div>
    <div style="color:#00d4ff;font-size:.78rem;font-weight:700;margin-top:4px">② 検証</div>
    <div style="color:#6699bb;font-size:.7rem;margin-top:2px">回収率・的中率を<br>シミュレーション</div>
  </div>
  <div style="text-align:center;padding:10px;background:rgba(0,212,255,.06);border-radius:8px">
    <div style="font-size:1.6rem">🤖</div>
    <div style="color:#00d4ff;font-size:.78rem;font-weight:700;margin-top:4px">③ 再学習</div>
    <div style="color:#6699bb;font-size:.7rem;margin-top:2px">LightGBMを<br>新データで更新</div>
  </div>
  <div style="text-align:center;padding:10px;background:rgba(0,212,255,.06);border-radius:8px">
    <div style="font-size:1.6rem">🎯</div>
    <div style="color:#00d4ff;font-size:.78rem;font-weight:700;margin-top:4px">④ 予想</div>
    <div style="color:#6699bb;font-size:.7rem;margin-top:2px">精度向上した<br>モデルで予想</div>
  </div>
</div>
</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 5: 場コード
# ══════════════════════════════════════════════
with tab5:
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
