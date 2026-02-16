
# Run:
#   pip install -r requirements.txt
#   streamlit run co2_map.py


from pathlib import Path
import numpy as np
import pandas as pd
import folium
from folium.elements import Element
from branca.colormap import LinearColormap

import streamlit as st
from streamlit_folium import st_folium
#N:\CSIT\_3rd sem\HCI\_project\Venice_Phase3


# -------------------- page setup (NO borders) --------------------
st.set_page_config(page_title="Venice CO2 Anchor Map", layout="wide")

st.markdown(
    '''
    <style>
      html, body, .stApp { margin:0 !important; padding:0 !important; background:#ffffff !important; }
      header, footer { visibility:hidden; height:0px; }
      #MainMenu { visibility:hidden; }

      /* remove all padding so the map is flush */
      .block-container { padding:0 !important; margin:0 !important; max-width:100% !important; }
      section.main > div { padding:0 !important; }
      div[data-testid="stAppViewContainer"] { padding:0 !important; }

      /* remove component borders */
      iframe { border:0 !important; }
      .element-container { margin:0 !important; padding:0 !important; }
    </style>
    ''',
    unsafe_allow_html=True,
)


# -------------------- load data --------------------
def data_dir() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "data").exists():
        return here / "data"
    return Path.cwd() / "data"


def load_co2_wide_to_stop_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    if "Start" in df.columns:
        df = df.rename(columns={"Start": "0.0"})
    if "Location" not in df.columns:
        raise ValueError("raw_venice_co2.csv must contain a column named 'Location'.")

    long = df.melt(id_vars=["Location"], var_name="stop_id", value_name="value")
    wide = long.pivot_table(index="stop_id", columns="Location", values="value", aggfunc="first").reset_index()

    wide["stop_id"] = pd.to_numeric(wide["stop_id"], errors="coerce")
    wide = wide.dropna(subset=["stop_id"]).copy()

    for c in ["CO2 ppm", "Temperature C", "Humidity %"]:
        if c in wide.columns:
            wide[c] = pd.to_numeric(wide[c], errors="coerce")

    return wide


def load_stops(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path, encoding="latin1")
    cols = {c.lower(): c for c in s.columns}
    for k in ["stop_id", "lat", "lon"]:
        if k not in cols:
            raise ValueError(f"Stops file must contain columns: stop_id, lat, lon (missing: {k})")
    s = s.rename(columns={cols["stop_id"]: "stop_id", cols["lat"]: "lat", cols["lon"]: "lon"})
    s["stop_id"] = pd.to_numeric(s["stop_id"], errors="coerce")
    s["lat"] = pd.to_numeric(s["lat"], errors="coerce")
    s["lon"] = pd.to_numeric(s["lon"], errors="coerce")
    return s.dropna(subset=["stop_id", "lat", "lon"]).copy()


def stop_label(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


# -------------------- build folium map --------------------
def build_map(merged: pd.DataFrame, stops: pd.DataFrame) -> folium.Map:
    center = [float(merged["lat"].mean()), float(merged["lon"].mean())]
    m = folium.Map(location=center, zoom_start=14, tiles=None, control_scale=True, prefer_canvas=True)

    # basemaps
    folium.TileLayer(
        tiles="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap &copy; CARTO",
        name="Light",
        show=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors",
        name="OpenStreetMap",
        show=False,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap &copy; CARTO",
        name="Dark",
        show=False,
    ).add_to(m)

    # pane so markers stay above tiles
    m.get_root().header.add_child(Element("<style>.leaflet-pane.leaflet-co2-pane{z-index:650;}</style>"))
    try:
        folium.map.CustomPane("co2-pane", z_index=650).add_to(m)
    except Exception:
        pass

    # colorbar
    vmin = float(np.nanmin(merged["CO2 ppm"].values))
    vmax = float(np.nanmax(merged["CO2 ppm"].values))
    cmap = LinearColormap(["#2ecc71", "#f1c40f", "#e74c3c"], vmin=vmin, vmax=vmax)
    cmap.caption = "CO2 (ppm)"
    cmap.add_to(m)

    # overlay groups
    g_points = folium.FeatureGroup(name="CO2 points", show=True)
    g_route = folium.FeatureGroup(name="Route", show=False)  # OFF by default
    g_points.add_to(m)
    g_route.add_to(m)

    # marker sizes
    r_min, r_max = 8.0, 20.0
    halo_extra = 4.0
    denom = (vmax - vmin) if vmax != vmin else 1.0

    popup_fields = [
        ("Time", "Time"),
        ("CO2", "CO2 ppm"),
        ("Temp", "Temperature C"),
        ("Humidity", "Humidity %"),
        ("Environment", "Type of environment"),
        ("Main landmark/s", "Main landmark/s"),
        ("Connection with water", "Connection with water"),
        ("Wind", "Wind"),
        ("Air smell - quality", "Air smell - quality"),
        ("Environmental noise", "Environmental noise"),
        ("Vehicular traffic (cars, boats)", "Vehicular traffic (cars, boats)"),
        ("Quantity of people", "Quantity of people"),
    ]

    for _, row in merged.iterrows():
        co2 = row.get("CO2 ppm", np.nan)
        if pd.isna(co2):
            continue
        co2 = float(co2)

        sid = float(row["stop_id"])
        sid_txt = stop_label(sid)

        landmark = row.get("Main landmark/s", "")
        landmark = "" if pd.isna(landmark) else str(landmark).strip()

        inner_r = r_min + (co2 - vmin) / denom * (r_max - r_min)
        outer_r = inner_r + halo_extra
        color = cmap(co2)

        # tooltip: stop_id + CO2 + landmark
        tooltip = f"{sid_txt} | CO2 {int(round(co2))} ppm" + (f" | {landmark}" if landmark else "")

        # popup HTML
        lines = [f"<b>Stop:</b> {sid_txt}"]
        for label, col in popup_fields:
            val = row.get(col, "")
            if pd.isna(val):
                val = ""
            if col == "CO2 ppm" and val != "":
                lines.append(f"<b>{label}:</b> {int(round(float(val)))} ppm")
            elif col == "Temperature C" and val != "":
                lines.append(f"<b>{label}:</b> {int(round(float(val)))} °C")
            elif col == "Humidity %" and val != "":
                lines.append(f"<b>{label}:</b> {int(round(float(val)))} %")
            elif val != "":
                lines.append(f"<b>{label}:</b> {val}")
        popup_html = "<div style='width:100%;'><div style='width:100%;'>" + "<br>".join(lines) + "</div></div>"

        lat, lon = float(row["lat"]), float(row["lon"])

        # halo ring
        folium.CircleMarker(
            location=[lat, lon],
            radius=outer_r,
            color="white",
            weight=3,
            fill=True,
            fill_color="white",
            fill_opacity=0.25,
            opacity=0.95,
            pane="co2-pane",
        ).add_to(g_points)

        # colored inner circle
        folium.CircleMarker(
            location=[lat, lon],
            radius=inner_r,
            color="#111111",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.92,
            opacity=0.9,
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=400),
            pane="co2-pane",
        ).add_to(g_points)

    # route polyline
    coords = stops.sort_values("stop_id")[["lat", "lon"]].values.tolist()
    folium.PolyLine(
        coords,
        color="#111111",
        weight=2,
        opacity=0.65,
        tooltip="Walking route (stop order)",
    ).add_to(g_route)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# -------------------- main --------------------
d = data_dir()
co2_file = d / "raw_venice_co2.csv"

stops_file = None
for name in ["venice_lat_lon.csv"]:
    p = d / name
    if p.exists():
        stops_file = p
        break

if not co2_file.exists() or stops_file is None:
    st.error(
        "Missing data files. Put these in a folder named 'data':\n"
        "- raw_venice_co2.csv\n"
        "- venice_lat_lon.csv"
    )
    st.stop()

co2_table = load_co2_wide_to_stop_table(co2_file)
stops = load_stops(stops_file)

merged = co2_table.merge(stops, on="stop_id", how="inner").dropna(subset=["CO2 ppm", "lat", "lon"])

m = build_map(merged, stops)

# map only (no extra UI)
st_folium(m, height=820, use_container_width=True)
