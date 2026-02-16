import io, csv
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Venice CO₂ vs Other Variables", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:2rem;padding-bottom:1rem}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

st.title("Venice CO$_2$ vs Other Variables - BoxPlot")
st.caption("Upload a CSV, choose one context variable (X), and optionally filter by other variables. Y is fixed to **CO₂ (ppm)**.")

ENCODINGS = ("utf-8-sig","utf-8","utf-16","cp1252","latin-1")
DELIMS = (",",";","\t","|")


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _read_csv(uploaded):
    b = uploaded.getvalue()
    last = None
    for enc in ENCODINGS:
        try:
            txt = b.decode(enc)
            break
        except Exception as e:
            last = e
    else:
        txt = b.decode("utf-8", errors="replace")

    # Try pandas auto-sep first
    try:
        df = pd.read_csv(io.StringIO(txt), sep=None, engine="python")
        if df.shape[1] >= 2:
            return df, {"encoding": enc, "delimiter": "auto"}, None, txt
    except Exception as e:
        last = e

    # Fallback: sniff + try common delimiters
    try:
        sniff = csv.Sniffer().sniff(txt[:5000], delimiters="".join(DELIMS))
        order = (sniff.delimiter,) + tuple(d for d in DELIMS if d != sniff.delimiter)
    except Exception:
        order = DELIMS

    for d in order:
        try:
            df = pd.read_csv(io.StringIO(txt), sep=d, engine="python")
            if df.shape[1] >= 2:
                return df, {"encoding": enc, "delimiter": d}, None, txt
        except Exception as e:
            last = e
    return None, None, last, txt


def _find_co2(cols):
    prefs = ("CO2 ppm","CO₂ ppm","CO2","co2 ppm","co2")
    for c in prefs:
        if c in cols:
            return c
    # last resort: anything that contains co2
    for c in cols:
        if "co2" in c.lower():
            return c
    return None


uploaded = st.file_uploader("1) Upload CSV", type=["csv"])
if not uploaded:
    st.info("Upload your Venice CO₂ CSV to start.")
    st.stop()

df, meta, err, raw = _read_csv(uploaded)
if df is None:
    st.error("Could not read the CSV.")
    if err:
        st.exception(err)
    st.code("\n".join(raw.splitlines()[:20]))
    st.stop()

# Clean columns
cols = [str(c).strip() for c in df.columns]
df.columns = cols
co2 = _find_co2(cols)
if not co2:
    st.error("Could not find a CO₂ column (e.g., 'CO2 ppm').")
    st.write("Columns found:", cols)
    st.stop()

df[co2] = _to_num(df[co2])

ALLOWED_X = [
    "Type of environment","Wind","Quantity of people","Vehicular traffic (cars, boats)",
    "Connection with water","Environmental noise","Air smell - quality","Main landmark/s",
]

x_candidates = [c for c in ALLOWED_X if c in cols]
if not x_candidates:
    # fallback: categorical-ish columns
    for c in cols:
        if c == co2:
            continue
        if _to_num(df[c]).notna().mean() >= 0.85:
            continue
        nun = df[c].astype(str).str.strip().replace({"nan":np.nan,"":np.nan}).dropna().nunique()
        if 2 <= nun <= 40:
            x_candidates.append(c)
    x_candidates = x_candidates[:8]

if not x_candidates:
    st.error("No suitable categorical context columns found for X.")
    st.stop()

x = st.selectbox("2) Choose the context variable (X)", x_candidates, index=0)

# Sidebar filters (compact, no time/date)
preferred = [
    "Type of environment","Quantity of people","Wind","Vehicular traffic (cars, boats)",
    "Connection with water","Environmental noise","Air smell - quality",
]
flt_cols = [c for c in preferred if c in cols and c != x]

f = df
if flt_cols:
    with st.sidebar.expander("Filters (optional)", expanded=False):
        st.caption("Filter the dataset (web-only richness). Then compare CO₂ distributions across your chosen X.")
        for c in flt_cols:
            s = f[c]
            num = _to_num(s)
            is_num = num.notna().mean() >= 0.85 and num.nunique(dropna=True) > 5
            if is_num:
                vmin, vmax = float(np.nanmin(num)), float(np.nanmax(num))
                if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
                    lo, hi = st.slider(c, vmin, vmax, (vmin, vmax), key=f"n_{c}")
                    f = f[(num >= lo) & (num <= hi)]
                else:
                    st.caption(f"{c}: no usable range")
            else:
                vals = (
                    s.astype(str).str.strip()
                    .replace({"":np.nan,"nan":np.nan,"None":np.nan})
                    .dropna().unique().tolist()
                )
                vals = sorted(vals, key=lambda z: z.lower())
                pick = st.multiselect(c, vals, default=vals, key=f"c_{c}")
                f = f[s.astype(str).str.strip().isin(pick)] if pick else f.iloc[0:0]

# Prepare plot data
p = f[[x, co2]].copy()
p[x] = p[x].astype(str).str.strip().replace({"":np.nan,"nan":np.nan,"None":np.nan})
p = p.dropna(subset=[x, co2])
if p.empty:
    st.warning("No valid rows after filters / missing-value removal.")
    st.stop()

# limit categories for readability
max_cat = 15
counts = p[x].value_counts()
keep = counts.head(max_cat).index.tolist()
p = p[p[x].isin(keep)]
order = counts.loc[keep].sort_values(ascending=False).index.tolist()
p[x] = pd.Categorical(p[x], categories=order, ordered=True)
p = p.sort_values(x)

st.markdown(f"**CO₂ column:** `{co2}`  ·  **Rows used:** {len(p):,}  ·  **Categories shown:** {p[x].nunique()}")

fig = px.box(p, x=x, y=co2, color=x, points="outliers")
fig.update_layout(
    template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
    showlegend=False, margin=dict(l=70, r=30, t=60, b=110),
    title=dict(text=f"CO₂ (ppm) by {x}", font=dict(size=20, color="#111111")),
)
fig.update_xaxes(title_font=dict(size=18, color="#111111"), tickfont=dict(size=14, color="#111111"),
                showgrid=True, gridcolor="rgba(0,0,0,0.10)", gridwidth=1, automargin=True)
fig.update_yaxes(title_font=dict(size=18, color="#111111"), tickfont=dict(size=14, color="#111111"),
                showgrid=True, gridcolor="rgba(0,0,0,0.10)", gridwidth=1)

st.plotly_chart(fig, use_container_width=True)
st.markdown("<hr style='margin-top:18px;margin-bottom:10px;border:none;border-top:1px solid rgba(0,0,0,0.12);'/>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#666;font-size:12px;'>&copy; copyright content Swarnali Mollick, 908523</div>", unsafe_allow_html=True)
