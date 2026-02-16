# Ins_Post_Ridge_v2.py
# Ridgeline (Streamlit + Plotly) — styled to match your main boxplot (white bg, #111 text, soft grid)

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.colors import qualitative

# Optional: smoother KDE if scipy is installed
try:
    from scipy.stats import gaussian_kde
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ------------------ Page ------------------
st.set_page_config(page_title="Insta Post CO₂-Ridgeline", layout="centered")

st.markdown(
    """
    <style>
      html, body, .stApp { background:#ffffff !important; color:#111111 !important; }
      .block-container { padding-top: 1.2rem; padding-bottom: 1.0rem; max-width: 900px; height: 200px;}
      header, footer { visibility:hidden; height:0px; }
      #MainMenu { visibility:hidden; }
      div[data-testid="stAlert"] p { color:#111111 !important; font-weight:600 !important; }


    .stFileUploader label{
        font-size: 16px !important;
        font-weight: 700 !important;
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

st.title("CO₂ vs Connection with Water (Ridgeline Plot -> Instagram post)")

uploaded = st.file_uploader("Upload a CSV", type=["csv"])
if uploaded is None:
    st.info("Upload your Venice CO₂ CSV to start.")
    st.stop()

# ------------------ Robust CSV read ------------------
if hasattr(uploaded, "size") and uploaded.size == 0:
    st.error("Uploaded file is empty (0 bytes). Please upload the CSV again.")
    st.stop()

raw = uploaded.getvalue()
if raw is None or len(raw) == 0:
    st.error("Uploaded file content is empty. Please upload again.")
    st.stop()

bio = io.BytesIO(raw)

df = None
read_errors = []
for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
    try:
        bio.seek(0)
        df = pd.read_csv(bio, encoding=enc, low_memory=False)
        if df is not None and df.shape[1] > 0:
            break
    except Exception as e:
        read_errors.append(f"{enc}: {type(e).__name__}")

if df is None or df.shape[1] == 0:
    st.error("Could not parse any columns. Make sure the file is a valid CSV with a header row.")
    st.caption("Read attempts: " + ", ".join(read_errors[:4]))
    st.stop()

df.columns = [str(c).strip() for c in df.columns]

# ------------------ Column selection ------------------
def guess_default(colnames, preferred, contains_any=()):
    if preferred in colnames:
        return preferred
    for c in colnames:
        cl = c.lower()
        if any(k in cl for k in contains_any):
            return c
    return colnames[0]

default_co2 = guess_default(df.columns, "CO2 ppm", contains_any=("co2",))
default_cat = guess_default(df.columns, "Connection with water", contains_any=("connection", "water", "environment", "people"))

c1, c2, c3 = st.columns([1.1, 1.2, 1.0])
with c1:
    co2_col = st.selectbox("Numeric variable (X)", df.columns, index=df.columns.get_loc(default_co2))
with c2:
    cat_col = st.selectbox("Category (groups)", df.columns, index=df.columns.get_loc(default_cat))
with c3:
    max_groups = st.slider("Max groups (top by frequency)", 3, 12, 7)

# ------------------ Clean data ------------------
d = df[[cat_col, co2_col]].copy()
d[co2_col] = pd.to_numeric(d[co2_col], errors="coerce")
d = d.dropna(subset=[cat_col, co2_col]).copy()

d[cat_col] = d[cat_col].astype(str).str.strip()
d = d[(d[cat_col] != "") & (~d[cat_col].str.lower().isin(["unknown", "nan", "none"]))].copy()

if len(d) == 0:
    st.error("After cleaning, no usable rows remain. Check column selections / missing values.")
    st.stop()

# Keep top groups
topcats = d[cat_col].value_counts().head(max_groups).index.tolist()
d = d[d[cat_col].isin(topcats)].copy()

# Order by median for story (same philosophy as your other plots)
order = topcats  # frequency order (matches your main boxplot)

# X grid
xmin, xmax = np.nanpercentile(d[co2_col], 1), np.nanpercentile(d[co2_col], 99)
if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
    xmin, xmax = float(d[co2_col].min()), float(d[co2_col].max())
xs = np.linspace(xmin, xmax, 260)

def kde_curve(vals):
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return np.zeros_like(xs)

    if HAVE_SCIPY:
        kde = gaussian_kde(vals)
        ys = kde(xs)
    else:
        # histogram density fallback
        hist, edges = np.histogram(vals, bins=30, range=(xmin, xmax), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ys = np.interp(xs, centers, hist)
        # light smoothing
        k = 7
        ys = np.convolve(ys, np.ones(k) / k, mode="same")

    ys = ys / (ys.max() if ys.max() > 0 else 1.0)  # normalize ridge height
    return ys

def hex_to_rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    a = float(max(0.0, min(1.0, a)))
    return f"rgba({r},{g},{b},{a:.3f})"

# ------------------ Build ridgeline ------------------
fig = go.Figure()
gap = 1.15
palette = qualitative.Plotly

for i, cat in enumerate(order):
    vals = d.loc[d[cat_col] == cat, co2_col].to_numpy(dtype=float)
    ys = kde_curve(vals)
    offset = i * gap
    y = ys + offset

    base_hex = palette[i % len(palette)]
    fill = hex_to_rgba(base_hex, 0.35)

    # Filled ridge
    fig.add_trace(go.Scatter(
        x=xs,
        y=y,
        mode="lines",
        line=dict(width=1.6, color=base_hex),
        fill="tozeroy",
        fillcolor=fill,
        name=str(cat),
        hovertemplate=(
            f"<b>{cat_col}</b>: {cat}<br>"
            f"<b>{co2_col}</b>: %{{x:.1f}}<br>"
            f"<b>Density</b>: %{{customdata:.2f}}<extra></extra>"
        ),
        customdata=ys,
        showlegend=False,  # y-axis labels do the job (cleaner for posts)
    ))

    # Median marker (black, consistent with your boxplot borders/text)
    med = float(np.nanmedian(vals)) if np.isfinite(np.nanmedian(vals)) else None
    if med is not None:
        fig.add_trace(go.Scatter(
            x=[med, med],
            y=[offset, offset + 0.25],
            mode="lines",
            line=dict(width=3, color="#111111"),
            showlegend=False,
            hoverinfo="skip"
        ))

# ------------------ Style (MATCHES your px.box styling)
# ------------------
fig.update_layout(
    template="plotly_white",
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=500,
    margin=dict(l=70, r=30, t=70, b=55),
    title=dict(text=f"CO₂ (ppm) distribution by {cat_col}", font=dict(size=20, color="#111111")),
    font=dict(size=16, color="#111111"),
    yaxis=dict(
        tickmode="array",
        tickvals=[i * gap + 0.35 for i in range(len(order))],
        ticktext=[str(c) for c in order],
        title="",
        zeroline=False,
    ),
)

fig.update_xaxes(
    title_text=co2_col,
    title_font=dict(size=18, color="#111111"),
    tickfont=dict(size=14, color="#111111"),
    showgrid=True,
    gridcolor="rgba(0,0,0,0.10)",
    gridwidth=1,
    zeroline=False,
    automargin=True,
)

fig.update_yaxes(
    tickfont=dict(size=14, color="#111111"),
    showgrid=True,
    gridcolor="rgba(0,0,0,0.10)",
    zeroline=False,
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("<div style='text-align:center; color:#666666; font-size:12px;'>© Swarnali Mollick, 908523</div>", unsafe_allow_html=True)
