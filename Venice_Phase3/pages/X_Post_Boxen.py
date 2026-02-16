
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.colors import qualitative

st.set_page_config(page_title="X Post CO₂-Boxen", layout="centered")

st.markdown(
    """
    <style>
      html, body, .stApp { background:#ffffff !important; color:#111111 !important; }
      .block-container { max-width: 900px; padding-top: 1.2rem; padding-bottom: 1.0rem; }
      header, footer { visibility:hidden; height:0px; }
      #MainMenu { visibility:hidden; }

      /* Make alert boxes readable */
      div[data-testid="stAlert"] p { color:#111111 !important; font-weight:600 !important; }

      /* Slightly stronger labels */
      .stSelectbox label, .stFileUploader label, .stNumberInput label {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #111111 !important;
      }


          /* "Browse files" button */
        section[data-testid="stFileUploaderDropzone"] button{
        background: #111111 !important;     /* button bg */
        color: #ffffff !important;          /* button text */
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("CO₂ vs Vehicular Traffic (Boxen Plot -> X post)")
st.caption("Nested quantile boxes (letter-value).")

# ------------------ robust CSV read (compact) ------------------
def read_csv_robust(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("Empty upload.")
    bio = io.BytesIO(raw)
    encs = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    seps = [None, ";", ",", "\t"]
    last_err = None
    for enc in encs:
        for sep in seps:
            try:
                bio.seek(0)
                df = pd.read_csv(bio, encoding=enc, sep=sep, engine="python" if sep is None else "c", low_memory=False)
                if df is not None and df.shape[1] > 1:
                    df.columns = [str(c).strip() for c in df.columns]
                    return df
            except Exception as e:
                last_err = e
    raise ValueError(f"Could not read CSV ({type(last_err).__name__}).")

def guess_col(cols, preferred, contains=()):
    if preferred in cols:
        return preferred
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in contains):
            return c
    return cols[0]

def hex_rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    a = float(max(0.0, min(1.0, a)))
    return f"rgba({r},{g},{b},{a:.3f})"

uploaded = st.file_uploader("Upload a CSV", type=["csv"])
if not uploaded:
    st.info("Upload your Venice CO₂ CSV to start.")
    st.stop()

try:
    df = read_csv_robust(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

# ------------------ selectors ------------------
default_co2 = guess_col(df.columns, "CO2 ppm", contains=("co2",))
default_cat = guess_col(df.columns, "Air smell - quality", contains=("air", "smell", "air quality", "people"))

c1, c2, c3 = st.columns([1.1, 1.4, 1.0])
with c1:
    CO2 = st.selectbox("Numeric (CO₂)", df.columns, index=df.columns.get_loc(default_co2))
with c2:
    CAT = st.selectbox("Category (context)", df.columns, index=df.columns.get_loc(default_cat))
with c3:
    k_depth = st.slider("Depth", 3, 8, 6)

# ------------------ clean + order ------------------
d = df[[CAT, CO2]].copy()
d[CO2] = pd.to_numeric(d[CO2], errors="coerce")
d[CAT] = d[CAT].astype(str).str.strip()
d = d.dropna(subset=[CAT, CO2])
d = d[(d[CAT] != "") & (~d[CAT].str.lower().isin(["unknown", "nan", "none"]))]

if d.empty:
    st.error("No usable rows after cleaning. Check column selections.")
    st.stop()

order = d[CAT].value_counts().index.tolist()
d[CAT] = pd.Categorical(d[CAT], categories=order, ordered=True)
d = d.sort_values(CAT)
st.markdown(f"**Y:** `{CO2}` · **X:** `{CAT}` · **Rows used:** {len(d):,} · **Categories:** {len(order)}")

# ------------------ boxen geometry ------------------
def quantile_bands(vals: np.ndarray, depth: int):
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    med = float(np.quantile(vals, 0.5))
    bands = [(float(np.quantile(vals, 0.5 - (0.5 / (2**i)))),
              float(np.quantile(vals, 0.5 + (0.5 / (2**i)))))
             for i in range(1, depth + 1)]
    return med, bands

def band_trace(i, w, lo, hi, fill, hover):
    x0, x1 = i - w/2, i + w/2
    return go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[lo, lo, hi, hi, lo],
        mode="lines",
        fill="toself",
        fillcolor=fill,
        line=dict(color="#111111", width=1),
        hoverinfo="text",
        text=hover,
        showlegend=False,
    )

# ------------------ figure ------------------
fig = go.Figure()
palette = qualitative.Plotly
box_w, shrink, jitter_sd = 0.56, 0.085, 0.035
rng = np.random.default_rng(42)

for i, cat in enumerate(order):
    vals = d.loc[d[CAT] == cat, CO2].to_numpy(float)
    out = quantile_bands(vals, int(k_depth))
    if out is None:
        continue
    med, bands = out
    base = palette[i % len(palette)]

    # outer -> inner (widest -> narrowest)
    for j, (lo, hi) in enumerate(bands[::-1]):
        depth_idx = len(bands) - 1 - j
        w = max(0.14, box_w - depth_idx * shrink)
        alpha = min(0.18 + 0.12 * (depth_idx + 1), 0.88)
        hover = f"{CAT}: {cat}<br>{CO2}: [{lo:.1f}, {hi:.1f}]<br>n={len(vals)}"
        fig.add_trace(band_trace(i, w, lo, hi, hex_rgba(base, alpha), hover))

    # median line
    fig.add_trace(go.Scatter(
        x=[i - box_w/2, i + box_w/2],
        y=[med, med],
        mode="lines",
        line=dict(color="#111111", width=2),
        hoverinfo="skip",
        showlegend=False
    ))

    # subtle points
    fig.add_trace(go.Scatter(
        x=i + rng.normal(0, jitter_sd, size=len(vals)),
        y=vals,
        mode="markers",
        marker=dict(size=5, color="rgba(17,17,17,0.25)"),
        hovertemplate=f"{CAT}: {cat}<br>{CO2}: %{{y:.1f}}<extra></extra>",
        showlegend=False,
    ))

fig.update_layout(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",
    height=650,
    margin=dict(l=70, r=30, t=70, b=130),
    title=dict(text=f"CO₂ (ppm) by {CAT} — Boxen", font=dict(size=20, color="#111111")),
    font=dict(size=16, color="#111111"),
)

fig.update_xaxes(
    title_text=CAT,
    title_font=dict(size=18, color="#111111"),
    tickfont=dict(size=14, color="#111111"),
    tickmode="array",
    tickvals=list(range(len(order))),
    ticktext=[str(c) for c in order],
    tickangle=16,
    showgrid=True,
    gridcolor="rgba(0,0,0,0.10)",
    gridwidth=1,
    zeroline=False,
    automargin=True,
)

fig.update_yaxes(
    title_text=CO2,
    title_font=dict(size=18, color="#111111"),
    tickfont=dict(size=14, color="#111111"),
    showgrid=True,
    gridcolor="rgba(0,0,0,0.10)",
    zeroline=False,
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='text-align:center; color:#666666; font-size:12px;'>© Swarnali Mollick, 908523</div>", unsafe_allow_html=True)
