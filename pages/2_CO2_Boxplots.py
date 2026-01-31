
# Run:
#   pip install -r requirements.txt
#   streamlit run co2_boxplots.py


import io
import csv
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Venice CO₂ VS Other Variables", layout="wide")

# --- Force WHITE UI (main + sidebar) + readable labels ---
st.markdown(
    '''
    <style>
      html, body, .stApp {
        background: #ffffff !important;
        color: #111111 !important;
      }
      .block-container { padding-top: 1.2rem; max-width: 1200px; }

      /* Sidebar */
      section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid rgba(0,0,0,0.10) !important;
      }

      /* Labels */
      .stSelectbox label, .stFileUploader label {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #111111 !important;
      }

      /* Markdown text */
      div[data-testid="stMarkdownContainer"] p,
      div[data-testid="stMarkdownContainer"] span {
        color: #111111 !important;
      }

      /* Select background */
      div[data-baseweb="select"] > div {
        background: #ffffff !important;
      }
    
      /* Fix selectbox text/placeholder visibility on white background */
      div[data-baseweb="select"] * { color: #111111 !important; }
      div[data-baseweb="select"] input { color: #111111 !important; }
      div[data-baseweb="select"] input::placeholder { color: #666666 !important; opacity: 1 !important; }
      div[data-baseweb="select"] span { color: #111111 !important; }

    ''',
    unsafe_allow_html=True,
)

st.title("Venice CO₂ vs Other Variables - BoxPlot")
st.caption("Upload a CSV and choose one context variable (X). Y is fixed to **CO₂ (ppm)**.")

# ------------------------
# Robust CSV loader (handles ; , \t | and common encodings)
# ------------------------
ENCODINGS = ["utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"]
DELIMS = [",", ";", "\t", "|"]

def try_decode(data: bytes):
    last_err = None
    for enc in ENCODINGS:
        try:
            return data.decode(enc), enc, None
        except Exception as e:
            last_err = e
    return data.decode("utf-8", errors="replace"), "utf-8 (replacement)", last_err

def sniff_delimiter(text: str):
    sample = text[:5000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join([d if d != "\t" else "\t" for d in DELIMS]))
        return dialect.delimiter
    except Exception:
        return None

def read_csv_robust(uploaded_file):
    data = uploaded_file.getvalue()
    text, enc_used, dec_err = try_decode(data)

    delim_guess = sniff_delimiter(text)
    delim_order = [delim_guess] + [d for d in DELIMS if d != delim_guess] if delim_guess else DELIMS

    last_err = None
    best_df = None
    best_meta = None

    for delim in delim_order:
        try:
            df = pd.read_csv(io.StringIO(text), sep=delim, engine="python")
            if df is not None and df.shape[1] >= 2 and df.shape[0] >= 1:
                best_df = df
                best_meta = {"encoding": enc_used, "delimiter": delim}
                break
            if best_df is None and df is not None and df.shape[0] >= 1:
                best_df = df
                best_meta = {"encoding": enc_used, "delimiter": delim}
        except Exception as e:
            last_err = e

    return best_df, best_meta, (last_err or dec_err), text

def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def find_co2_column(cols):
    candidates = ["CO2 ppm", "CO₂ ppm", "CO2", "co2 ppm", "co2"]
    for c in candidates:
        if c in cols:
            return c
    return None

# ------------------------
# Upload + parse
# ------------------------
uploaded = st.file_uploader("1) Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df, meta, err, raw_text = read_csv_robust(uploaded)
if df is None:
    st.error("Could not read the CSV.")
    if err is not None:
        st.exception(err)
    st.write("Preview of file (first 20 lines):")
    st.code("\n".join(raw_text.splitlines()[:20]))
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
co2_col = find_co2_column(df.columns)

if co2_col is None:
    st.error("Could not find a CO₂ column. Expected something like 'CO2 ppm'.")
    st.write("Columns found:", list(df.columns))
    st.stop()

df[co2_col] = to_numeric_series(df[co2_col])

# ------------------------
# X choices (small + meaningful)
# ------------------------
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
    fallback = []
    for c in df.columns:
        if c == co2_col:
            continue
        s = df[c]
        parsed = to_numeric_series(s)
        if parsed.notna().mean() >= 0.85:
            continue
        nun = s.astype(str).str.strip().replace("nan", np.nan).dropna().nunique()
        if 2 <= nun <= 40:
            fallback.append(c)
    x_candidates = fallback[:8]

if not x_candidates:
    st.error("No suitable categorical context columns found for X.")
    st.write("Columns found:", list(df.columns))
    st.stop()

x_col = st.selectbox("2) Choose the context variable (X)", x_candidates, index=0)

# ------------------------
# Prepare data
# ------------------------
tmp = df[[x_col, co2_col]].copy()
tmp[x_col] = tmp[x_col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
tmp = tmp.dropna(subset=[x_col, co2_col])

if tmp.empty:
    st.warning("No valid rows after removing missing values.")
    st.stop()

max_categories = 15
counts = tmp[x_col].value_counts()
keep = counts.head(max_categories).index.tolist()
tmp = tmp[tmp[x_col].isin(keep)].copy()

order = counts.loc[keep].sort_values(ascending=False).index.tolist()
tmp[x_col] = pd.Categorical(tmp[x_col], categories=order, ordered=True)
tmp = tmp.sort_values(x_col)

st.markdown(
    f"**CO₂ column:** `{co2_col}`  ·  **Rows used:** {len(tmp):,}  ·  **Categories shown:** {tmp[x_col].nunique()}"
)

# ------------------------
# Boxplot + GRID
# ------------------------
fig = px.box(
    tmp,
    x=x_col,
    y=co2_col,
    color=x_col,
    points="outliers",
)

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

st.plotly_chart(fig, use_container_width=True)

# --- Footer ---

st.markdown("<hr style='margin-top: 18px; margin-bottom: 10px; border: none; border-top: 1px solid rgba(0,0,0,0.12);'/>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#666666; font-size:12px;'>&copy; copyright content Swarnali Mollick, 908523</div>",
    unsafe_allow_html=True
)

