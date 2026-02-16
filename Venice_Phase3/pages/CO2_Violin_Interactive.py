import io, csv
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.io as pio

st.set_page_config(page_title="Venice CO₂ — Violin (Filtered)", layout="wide")

st.markdown(
    """
<style>
  html, body, .stApp { background:#0b1220 !important; }
  .stApp, .stApp * { color:#f8fafc !important; }
  .block-container { padding-top: 1.4rem; padding-bottom: 1rem; }
  header, footer, #MainMenu { visibility:hidden; height:0px; }

  /* sidebar */
  section[data-testid="stSidebar"] { background:#0b1220 !important; }

  /* File uploader dropzone */
  [data-testid="stFileUploaderDropzone"]{
    background: rgba(255,255,255,0.08) !important;
    border: 1px dashed rgba(255,255,255,0.25) !important;
    border-radius: 10px !important;
    padding: 16px !important;
  }
  [data-testid="stFileUploaderDropzone"] *{ color:#f8fafc !important; }

  /* Inputs */
  div[data-baseweb="select"] > div{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 8px !important;
  }

  /* Plotly card */
  [data-testid="stPlotlyChart"]{
    background:#ffffff !important;
    border-radius: 12px !important;
    padding: 10px !important;
  }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Venice CO2 vs Other Variables — Violin Plot")
st.caption("Web version: pick one context variable (X) + apply sidebar filters. Y is fixed to **CO2 (ppm)**.")

# ---------- robust CSV ----------
ENC = ["utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"]
DELIMS = [",", ";", "\t", "|"]

def _decode(b: bytes):
    for e in ENC:
        try:
            return b.decode(e), e
        except Exception:
            pass
    return b.decode("utf-8", errors="replace"), "utf-8(replace)"

def _sniff(txt: str):
    try:
        return csv.Sniffer().sniff(txt[:5000], delimiters="".join(DELIMS)).delimiter
    except Exception:
        return None

def read_csv_robust(up):
    raw = up.getvalue()
    txt, enc = _decode(raw)
    d0 = _sniff(txt)
    dlist = ([d0] + [d for d in DELIMS if d != d0]) if d0 else DELIMS
    last = None
    for d in dlist:
        try:
            df = pd.read_csv(io.StringIO(txt), sep=d, engine="python")
            if df is not None and df.shape[1] >= 2 and df.shape[0] >= 1:
                return df, enc, d
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not parse CSV. Last error: {last}")

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def find_co2(cols):
    for c in ["CO2 ppm", "CO₂ ppm", "CO2", "co2 ppm", "co2"]:
        if c in cols:
            return c
    return None

# ---------- upload ----------
up = st.file_uploader("Upload CSV", type=["csv"])
if up is None:
    st.info("Upload your Venice CO₂ CSV to start.")
    st.stop()

df, enc, delim = read_csv_robust(up)
df.columns = [str(c).strip() for c in df.columns]

co2 = find_co2(df.columns)
if co2 is None:
    st.error("Could not find a CO₂ column (expected something like `CO2 ppm`).")
    st.write("Columns found:", list(df.columns))
    st.stop()

df[co2] = to_num(df[co2])

# ---------- X candidates ----------
ALLOWED_X = [
    "Type of environment",
    "Wind",
    "Quantity of people",
    "Vehicular traffic (cars, boats)",
    "Connection with water",
    "Environmental noise",
    "Air smell - quality",
    "Main landmark/s",
]
x_candidates = [c for c in ALLOWED_X if c in df.columns]
if not x_candidates:
    fb = []
    for c in df.columns:
        if c == co2: 
            continue
        if to_num(df[c]).notna().mean() >= 0.85:
            continue
        nun = df[c].astype(str).str.strip().replace("nan", np.nan).dropna().nunique()
        if 2 <= nun <= 40:
            fb.append(c)
    x_candidates = fb[:12]
if not x_candidates:
    st.error("No suitable categorical context columns found for X.")
    st.stop()

x_col = st.selectbox("Choose the context variable (X)", x_candidates, index=0)

# ---------- sidebar filters (compact, no Time) ----------
FILTER_COLS = [
    "Type of environment",
    "Quantity of people",
    "Wind",
    "Vehicular traffic (cars, boats)",
    "Connection with water",
    "Environmental noise",
    "Air smell - quality",
    "Main landmark/s",
]
present_filters = [c for c in FILTER_COLS if c in df.columns and c != x_col]

fdf = df.copy()
with st.sidebar:
    st.markdown("### Filters")
    with st.expander("Filters (optional)", expanded=False):
        for col in present_filters:
            s = fdf[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
            opts = sorted(s.dropna().unique().tolist())
            if len(opts) < 2:
                continue
            sel = st.multiselect(col, opts, default=opts)
            if sel:
                fdf = fdf[s.isin(sel)]

# ---------- prep ----------
tmp = fdf[[x_col, co2]].copy()
tmp[x_col] = tmp[x_col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
tmp = tmp.dropna(subset=[x_col, co2])

if tmp.empty:
    st.warning("No rows left after filtering / missing-value removal.")
    st.stop()

# keep top categories (readability)
MAX_CATS = 15
counts = tmp[x_col].value_counts()
keep = counts.head(MAX_CATS).index.tolist()
tmp = tmp[tmp[x_col].isin(keep)].copy()
order = counts.loc[keep].sort_values(ascending=False).index.tolist()
tmp[x_col] = pd.Categorical(tmp[x_col], categories=order, ordered=True)
tmp = tmp.sort_values(x_col)

st.markdown(f"**CO₂ column:** `{co2}`  ·  **Rows used:** {len(tmp):,}  ·  **Categories shown:** {tmp[x_col].nunique()}")

# axis short labels (hover keeps full)
def short_label(s, maxlen=18):
    base = str(s).split("(")[0].strip()
    return (base[:maxlen-1] + "…") if len(base) > maxlen else base

full = list(tmp[x_col].cat.categories)
short = [short_label(v) for v in full]
# ensure unique
seen, uniq = {}, []
for v in short:
    seen[v] = seen.get(v, 0) + 1
    uniq.append(v if seen[v] == 1 else f"{v} {seen[v]}")

# ---------- violin ----------
fig = px.violin(tmp, x=x_col, y=co2, color=x_col, box=True, points="outliers")
fig.update_layout(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",
    showlegend=False,
    margin=dict(l=70, r=30, t=60, b=110),
    title=dict(text=f"CO₂ (ppm) by {x_col}", font=dict(size=20, color="#111111")),
)
fig.update_xaxes(
    title_font=dict(size=18, color="#111111"),
    tickfont=dict(size=14, color="#111111"),
    tickmode="array",
    tickvals=full,
    ticktext=uniq,
    showgrid=True,
    gridcolor="rgba(0,0,0,0.10)",
    gridwidth=1,
    automargin=True,
)
fig.update_yaxes(
    title_font=dict(size=18, color="#111111"),
    tickfont=dict(size=14, color="#111111"),
    showgrid=True,
    gridcolor="rgba(0,0,0,0.10)",
    gridwidth=1,
)
fig.update_traces(hovertemplate=f"<b>{x_col}</b>: %{{x}}<br><b>{co2}</b>: %{{y}}<extra></extra>")

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# downloads (minimal)
c1, c2 = st.columns(2)
with c1:
    st.download_button("Download interactive HTML", pio.to_html(fig, full_html=True, include_plotlyjs="cdn"),
                       file_name="violin_plot.html", mime="text/html")
with c2:
    st.download_button("Download data (CSV)", tmp.to_csv(index=False).encode("utf-8"),
                       file_name="violin_plot_data.csv", mime="text/csv")
