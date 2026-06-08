"""
visualize_sacramento.py

Lightweight interactive map showing crosswalk detections clipped to
a manageable area of Sacramento. Opens in any browser.

Requirements:
    pip install geopandas folium

Usage:
    python visualize_sacramento.py
"""

from pathlib import Path
import geopandas as gpd
import folium
from folium.plugins import MeasureControl, MiniMap, Fullscreen
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MERGED_DIR   = Path("tile2net_output/sacramento_pedestrian_network/merged")
NETWORK_SHP  = MERGED_DIR / "sacramento_network.shp"
POLYGONS_SHP = MERGED_DIR / "sacramento_polygons.shp"
OUTPUT_HTML  = Path("sacramento_crosswalks.html")

# Clip to downtown / midtown Sacramento.
# Adjust if you want a different neighborhood.
CLIP_BOUNDS = {
    "lat_min":  38.555,
    "lat_max":  38.580,
    "lon_min": -121.510,
    "lon_max": -121.470,
}

MAP_CENTER = [
    (CLIP_BOUNDS["lat_min"] + CLIP_BOUNDS["lat_max"]) / 2,
    (CLIP_BOUNDS["lon_min"] + CLIP_BOUNDS["lon_max"]) / 2,
]
MAP_ZOOM = 15

# ---------------------------------------------------------------------------
# LOAD, FILTER, CLIP
# ---------------------------------------------------------------------------

print("Loading shapefiles...")
network  = gpd.read_file(NETWORK_SHP)
polygons = gpd.read_file(POLYGONS_SHP)

if network.crs and network.crs.to_epsg() != 4326:
    network = network.to_crs(4326)
if polygons.crs and polygons.crs.to_epsg() != 4326:
    polygons = polygons.to_crs(4326)

net_cw  = network[network["f_type"] == "crosswalk"].copy()
poly_cw = polygons[polygons["f_type"] == "crosswalk"].copy()

print(f"  Crosswalk centerlines (all city): {len(net_cw):,}")
print(f"  Crosswalk polygons    (all city): {len(poly_cw):,}")

def clip_to_bounds(gdf, bounds):
    return gdf.cx[
        bounds["lon_min"]:bounds["lon_max"],
        bounds["lat_min"]:bounds["lat_max"],
    ]

net_cw  = clip_to_bounds(net_cw,  CLIP_BOUNDS)
poly_cw = clip_to_bounds(poly_cw, CLIP_BOUNDS)

print(f"  Crosswalk centerlines (clipped):  {len(net_cw):,}")
print(f"  Crosswalk polygons    (clipped):  {len(poly_cw):,}")

if len(net_cw) == 0 and len(poly_cw) == 0:
    print("\nWARNING: No crosswalks found in clip area.")
    print("Try adjusting CLIP_BOUNDS in the script.")
    raise SystemExit

# ---------------------------------------------------------------------------
# BUILD MAP
# ---------------------------------------------------------------------------

print("Building map...")

m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles=None)

# --- Basemaps (radio toggle — only one active at a time) ---
# Sacramento County 2022 orthos — same imagery tile2net was trained on
folium.TileLayer(
    tiles="https://mapservices.gis.saccounty.net/arcgis/rest/services/Cache/IMAGERY_2022_WEB_MERCATOR/MapServer/tile/{z}/{y}/{x}",
    attr="Sacramento County GIS 2022",
    name="Sacramento County 2022 (inference imagery)",
    overlay=False,
    control=True,
    show=True,
).add_to(m)

# Esri World Imagery — useful fallback if county tiles are slow
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Esri World Imagery",
    overlay=False,
    control=True,
    show=False,
).add_to(m)

# --- Crosswalk polygons (segmentation output, overlay toggle) ---
if len(poly_cw) > 0:
    poly_layer = folium.FeatureGroup(
        name=f"Crosswalk Polygons ({len(poly_cw):,})",
        show=True,
        overlay=True,
    )
    folium.GeoJson(
        poly_cw.__geo_interface__,
        style_function=lambda f: {
            "fillColor":   "#FF6B6B",
            "color":       "#CC0000",
            "weight":      1.0,
            "fillOpacity": 0.45,
        },
    ).add_to(poly_layer)
    poly_layer.add_to(m)

# --- Crosswalk centerlines (overlay toggle) ---
if len(net_cw) > 0:
    net_layer = folium.FeatureGroup(
        name=f"Crosswalk Centerlines ({len(net_cw):,})",
        show=True,
        overlay=True,
    )
    folium.GeoJson(
        net_cw.__geo_interface__,
        style_function=lambda f: {
            "color":   "#00E5FF",
            "weight":  2.5,
            "opacity": 0.9,
        },
    ).add_to(net_layer)
    net_layer.add_to(m)

# --- Study area boundary ---
folium.Rectangle(
    bounds=[
        [CLIP_BOUNDS["lat_min"], CLIP_BOUNDS["lon_min"]],
        [CLIP_BOUNDS["lat_max"], CLIP_BOUNDS["lon_max"]],
    ],
    color="#FFD700",
    weight=2,
    fill=False,
    dash_array="6 4",
    name="Study Area Boundary",
).add_to(m)

# --- Controls ---
folium.LayerControl(collapsed=False).add_to(m)
MeasureControl(
    primary_length_unit="meters",
    secondary_length_unit="feet",
    primary_area_unit="sqmeters",
).add_to(m)
MiniMap(toggle_display=True, tile_layer="CartoDB positron").add_to(m)
Fullscreen().add_to(m)

# --- Title ---
title_html = """
<div style="
    position: fixed;
    top: 12px; left: 50%; transform: translateX(-50%);
    z-index: 9999;
    background: rgba(10,10,20,0.85);
    border: 1px solid rgba(0,229,255,0.4);
    border-radius: 8px;
    padding: 10px 22px;
    font-family: 'Courier New', monospace;
    color: #00E5FF;
    font-size: 15px;
    font-weight: bold;
    letter-spacing: 1px;
    box-shadow: 0 0 18px rgba(0,229,255,0.2);
    pointer-events: none;
    text-align: center;
">
    Sacramento Crosswalk Detection — tile2net
    <span style="color:#aaa; font-size:11px; display:block; margin-top:2px;">
        Downtown / Midtown · zoom 19 · Sacramento County 2022 imagery
    </span>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

# --- Legend ---
legend_html = """
<div style="
    position: fixed;
    bottom: 40px; right: 12px;
    z-index: 9999;
    background: rgba(10,10,20,0.85);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'Courier New', monospace;
    color: #eee;
    font-size: 12px;
    line-height: 2;
">
    <b style="color:#00E5FF; letter-spacing:1px;">LEGEND</b><br>
    <span style="color:#00E5FF;">━━━</span>&nbsp; Crosswalk Centerlines<br>
    <span style="color:#FF6B6B;">▬▬▬</span>&nbsp; Crosswalk Polygons<br>
    <span style="color:#FFD700;">╌╌╌</span>&nbsp; Study Area Boundary
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------

m.save(OUTPUT_HTML)
print(f"\n✓ Saved to: {OUTPUT_HTML.resolve()}")
print("  Open in Chrome or Firefox.")
print(f"\n  Features in view:")
print(f"    Crosswalk centerlines : {len(net_cw):,}")
print(f"    Crosswalk polygons    : {len(poly_cw):,}")
print(f"\n  To show a different area, edit CLIP_BOUNDS in the script.")
