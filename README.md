# Venice_CO2_Phase3
GitHub repo with two Streamlit Phase-3 web representations using the Venice CO₂ dataset: (1) interactive Folium/Leaflet map with CO₂ color/size encoding, tooltips/popups, basemap switch and optional route toggle; (2) simple Plotly boxplot dashboard with CO₂ fixed and selectable context X via CSV upload. 


Venice Phase 3 — CO₂ Web Representations (Streamlit)

This repository contains two connected web-based visual representations built for Phase 3 (Visual Design). Both representations are anchored to the same Venice walking-survey dataset, combining CO₂ sensor readings with contextual observations (environment type, people/traffic, wind, noise, etc.). The goal is to communicate environmental conditions to citizens through clear, interactive, web-friendly visualizations.

Web Representations Included

1) CO₂ Anchor Map (Interactive Leaflet Map)

An interactive map of Venice stops where each point is a sampled location along the walking route.

  Encoding: circle color represents CO₂ concentration (ppm) using a green → yellow → red scale; circle size scales with CO₂ level for visibility.
  Interaction: hover tooltip shows stop information; click opens a detailed popup with additional contextual variables.
  Layers: users can switch basemaps (Light / OpenStreetMap / Dark) and optionally toggle the walking route overlay.
  Purpose: communicates the spatial distribution of CO₂ hotspots across the route and helps users identify where air quality may be worse.

2) CO₂ vs Context Boxplot Dashboard (Interactive)

A simple Streamlit dashboard to explore how CO₂ changes across contextual categories.

  Fixed Y variable: CO₂ (ppm)
  Selectable X variable: user chooses one context variable (e.g., environment type, wind, people/traffic, noise, etc.).
  Output: a boxplot showing CO₂ distribution per category (median, variability, outliers).
  Upload support: users can upload a CSV to generate the plot dynamically.
  Purpose: helps citizens understand which contexts are associated with higher CO₂ levels and more variability, complementing the map’s “where” with “under what conditions”.

Dataset

The visualizations use a Venice dataset collected along a walking route with:
  CO₂ measurements (ppm)
  Stop coordinates (latitude/longitude)
  Contextual annotations (environment, landmark, people, traffic, wind, noise, smell, etc.)
The repository includes the dataset files required to reproduce the results locally.

How to Run Locally

  Install dependencies: pip install -r requirements.txt
  
  Start the Streamlit multipage app: streamlit run Home.py
