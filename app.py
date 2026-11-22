"""محلل الأسهم: فني + مالي + قرار + Backtesting
التشغيل:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="محلل الأسهم", layout="wide")

# ---------- 1) البيانات ----------
@st.cache_data(ttl=3600)
def load_prices(ticker, period):
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    return df.dropna()

@st.cache_data(ttl=3600)
def load_info(ticker):
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

# ---------- 2) التحليل الفني ----------
def add_indicators(df):
    d = df.copy()
    c = d["Close"]
    d["SMA50"], d["SMA200"] = c.rolling(50).mean(), c.rolling(200).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_UP"], d["BB_LO"] = mid + 2 * sd, mid - 2 * sd
    d["VOL_RATIO"] = d["Volume"] / d["Volume"].rolling(20).mean()
    return d

def tech_score_series(d):
    """درجة من -100 إلى +100 لكل يوم."""
    s = pd.Series(0.0, index=d.index)
    s += np.where(d["Close"] > d["SMA50"], 1, -1)
    s += np.where(d["SMA50"] > d["SMA200"], 1, -1)
    s += np.where(d["RSI"] < 30, 1, np.where(d["RSI"] > 70, -1, 0))
    s += np.where(d["MACD"] > d["MACD_SIG"], 1, -1)
    s += np.where(d["Close"] < d["BB_LO"], 1, np.where(d["Close"] > d["BB_UP"], -1, 0))
    return (s * 20).where(d["SMA200"].notna())

# ---------- 3) التحليل المالي ----------
def fundamental_score(info):
    """كل نسبة تاخد -1 / 0 / +1، والنتيجة متوسطها × 100."""
    rules = {
        "P/E": ("trailingPE", lambda v: 1 if 0 < v < 15 else (-1 if v <= 0 or v > 30 else 0)),
        "P/B": ("priceToBook", lambda v: 1 if v < 1.5 else (-1 if v > 5 else 0)),
        "ROE": ("returnOnEquity", lambda v: 1 if v > 0.15 else (-1 if v < 0.05 else 0)),
        "الدين/حقوق الملكية": ("debtToEquity", lambda v: 1 if v < 50 else (-1 if v > 150 else 0)),
        "نمو الإيرادات": ("revenueGrowth", lambda v: 1 if v > 0.10 else (-1 if v < 0 else 0)),
        "نمو الأرباح": ("earningsGrowth", lambda v: 1 if v > 0.10 else (-1 if v < 0 else 0)),
        "هامش الربح": ("profitMargins", lambda v: 1 if v > 0.15 else (-1 if v < 0.03 else 0)),
    }
    rows, scores = [], []
    for name, (key, fn) in rules.items():
        v = info.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            rows.append((name, None, "غير متاح"))
            continue
        sc = fn(v)
        scores.append(sc)
        rows.append((name, round(float(v), 3), {1: "جيد", 0: "محايد", -1: "ضعيف"}[sc]))
    total = float(np.mean(scores) * 100) if scores else None
    return total, pd.DataFrame(rows, columns=["النسبة", "القيمة", "التقييم"])

# ---------- 4) القرار ----------
def decide(tech, fund, w_tech):
    if fund is None:
        combined, note = tech, "مفيش بيانات مالية، القرار فني بس"
    else:
        combined, note = w_tech * tech + (1 - w_tech) * fund, ""
    action = "شراء 🟢" if combined >= 30 else ("بيع 🔴" if combined <= -30 else "احتفاظ 🟡")
    return combined, action, note

# ---------- 5) Backtesting ----------
def backtest(d, score, threshold=20, fee=0.001):
    """نشتري لما الدرجة >= threshold ونبيع غير كده. التنفيذ تاني يوم (من غير تحيز المستقبل)."""
    ret = d["Close"].pct_change().fillna(0)
    pos = (score >= threshold).astype(float).shift(1).fillna(0)
    cost = pos.diff().abs().fillna(0) * fee
    strat = (pos * ret - cost)
    eq_s, eq_b = (1 + strat).cumprod(), (1 + ret).cumprod()

    def stats(eq, r):
        years = max(len(eq) / 252, 1e-9)
        cagr = eq.iloc[-1] ** (1 / years) - 1
        dd = (eq / eq.cummax() - 1).min()
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        return {"العائد السنوي": f"{cagr:.1%}", "أقصى هبوط": f"{dd:.1%}", "Sharpe": f"{sharpe:.2f}"}

    table = pd.DataFrame({"الاستراتيجية": stats(eq_s, strat), "شراء واحتفاظ": stats(eq_b, ret)})
    trades = int((pos.diff() == 1).sum())
    return eq_s, eq_b, table, trades

# ---------- 6) الواجهة ----------
st.title("📈 محلل الأسهم: فني + مالي")
with st.sidebar:
    ticker = st.text_input("رمز السهم", "AAPL").strip().upper()
    st.caption("أمثلة: AAPL, MSFT, 2222.SR, COMI.CA")
    period = st.selectbox("الفترة", ["1y", "2y", "5y", "10y"], index=2)
    w_tech = st.slider("وزن التحليل الفني", 0.0, 1.0, 0.6, 0.05)

try:
    prices = load_prices(ticker, period)
except Exception as e:
    st.error(f"مقدرتش أجيب البيانات: {e}")
    st.stop()
if len(prices) < 220:
    st.warning("البيانات قليلة. جرّب فترة أطول أو رمز تاني (محتاج 200 يوم على الأقل).")
    st.stop()

d = add_indicators(prices)
score = tech_score_series(d)
tech_now = float(score.dropna().iloc[-1])
fund_now, fund_table = fundamental_score(load_info(ticker))
combined, action, note = decide(tech_now, fund_now, w_tech)

c1, c2, c3, c4 = st.columns(4)
c1.metric("القرار", action)
c2.metric("الدرجة المجمعة", f"{combined:.0f}")
c3.metric("الفني", f"{tech_now:.0f}")
c4.metric("المالي", "—" if fund_now is None else f"{fund_now:.0f}")
if note:
    st.info(note)

tab1, tab2, tab3 = st.tabs(["الرسم الفني", "التحليل المالي", "Backtesting"])

with tab1:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="السعر"), 1, 1)
    for col in ["SMA50", "SMA200", "BB_UP", "BB_LO"]:
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col, line=dict(width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d.RSI, name="RSI"), 2, 1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.MACD, name="MACD"), 3, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d.MACD_SIG, name="Signal"), 3, 1)
    fig.update_layout(height=750, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.dataframe(fund_table, use_container_width=True, hide_index=True)
    st.caption("العتبات عامة وبتختلف من قطاع لقطاع. البنوك مثلاً ليها معايير دين مختلفة.")

with tab3:
    thr = st.slider("حد الشراء (الدرجة الفنية)", -40, 100, 20, 20)
    eq_s, eq_b, table, trades = backtest(d, score, thr)
    st.line_chart(pd.DataFrame({"الاستراتيجية": eq_s, "شراء واحتفاظ": eq_b}))
    st.table(table)
    st.caption(f"عدد الصفقات: {trades}. المحاكاة بتشمل عمولة 0.1% وتنفيذ تاني يوم.")

st.warning("⚠️ ده تحليل آلي مش نصيحة استثمارية. الأداء التاريخي مش ضمان للمستقبل، "
           "والتحليل المالي هنا بيستخدم بيانات مجانية ممكن تكون ناقصة أو متأخرة.")
