# Sacramento Crossing Extraction

Automated extraction of crosswalks from aerial imagery using [tile2net](https://github.com/VIDA-NYU/tile2net), applied to the City of Sacramento. Sacramento is not in tile2net's built-in supported regions, so this pipeline handles tile downloading manually from Sacramento County's public imagery service and feeds the local tiles into tile2net for inference.

The primary research interest is measuring crosswalk lengths and evaluating how well tile2net performs on a western US city with wider streets and more suburban land use than the east coast cities it was trained on.

---

## Repo structure

```
.
├── run_tile2net_sacramento.py   # Full pipeline: download → inference → merge
├── visualize_sacramento.py      # Interactive HTML map of crosswalk detections
└── README.md
```

---

## Pipeline overview

`run_tile2net_sacramento.py` runs five stages end to end:

1. **Download** — fetches zoom-19 XYZ tiles from Sacramento County's public 2022 ortho imagery service into a local `z/x/y.png` folder structure
2. **Generate** — builds tile2net's inference grid from the local tile directory
3. **Inference** — runs the HRNetV2 segmentation model to produce polygons, then derives a centerline pedestrian network
4. **Cleanup** — optionally deletes raw tiles and stitched images to recover disk space
5. **Merge** — concatenates the four quadrant shapefiles into single city-wide files

The city is split into four quadrants (SW, SE, NW, NE) to keep memory and disk usage manageable at zoom 19. Each quadrant is ~13,000 tiles and takes roughly 20 minutes of inference on an RTX 3060.

---

## Requirements

- Python 3.10 or 3.11 — tile2net breaks on 3.12+
- NVIDIA GPU strongly recommended — CPU inference is extremely slow
- ~15 GB free disk space per quadrant during inference (less with cleanup enabled)

```bash
pip install git+https://github.com/VIDA-NYU/tile2net
pip install geopandas requests folium
```

---

## Usage

### 1. Run the pipeline

```bash
python run_tile2net_sacramento.py
```

All configuration is at the top of the script under the `CONFIG` block — output directory, bounding boxes, zoom level, thread count, and cleanup/merge flags. No arguments needed.

Output shapefiles land in:

```
tile2net_output/sacramento_pedestrian_network/merged/
    sacramento_network.shp     # pedestrian network centerlines (sidewalks, crosswalks, connections)
    sacramento_polygons.shp    # raw segmentation polygons (sidewalk, crosswalk, road)
```

Each feature has an `f_type` attribute (`sidewalk`, `crosswalk`, `sidewalk_connection`, `road`) and a `quadrant` attribute indicating which sub-region it came from.

If the pipeline is interrupted, re-running it will skip any quadrant whose network output directory already exists. To force a re-run of a specific quadrant, delete its output folder and restart.

### 2. Visualize

```bash
python visualize_sacramento.py
```

Generates `sacramento_crosswalks.html` — a self-contained interactive map that opens in any browser with no additional installs. Shows crosswalk detections only (filtered from the full network) clipped to a downtown/midtown study area for performance.

Map features:
- Toggle between Sacramento County 2022 imagery (the same source used for inference) and Esri World Imagery
- Crosswalk polygons and centerlines as independent toggleable layers
- Built-in measurement tool for length/area
- Minimap and fullscreen controls

To change the study area, edit `CLIP_BOUNDS` at the top of `visualize_sacramento.py`.

---

## Known issues and patches

tile2net has several bugs that surface on real-world data outside its supported regions. Two files in the tile2net package need to be patched before running:

### `pednet.py` — Qhull precision crash during centerline generation

Degenerate sidewalk polygons can cause a `scipy.spatial.qhull.QhullError` in `to_cline()`. Wrap the function body in a `try/except` that returns `None` on failure, and add a `None` guard in `create_lines()` before the result is used.

File: `<env>/Lib/site-packages/tile2net/raster/pednet.py`

### `geodata_utils.py` — GEOS TopologyException during polygon union

Negative buffer operations (the erode step in `buffer_union_erode`) can produce self-intersecting geometries that cause `shapely.errors.GEOSException: TopologyException` during the subsequent union. Fix by adding a `buffer(0)` repair pass after the erode step and inside `unary_multi` before dissolve.

File: `<env>/Lib/site-packages/tile2net/raster/tile_utils/geodata_utils.py`

For a more maintainable setup, clone tile2net and install as editable so patches survive reinstalls:

```bash
git clone https://github.com/VIDA-NYU/tile2net
pip install -e ./tile2net
```

---

## Tile source

Sacramento County GIS — 2022 Color 3-Inch Orthos. Public, no API key required.

```
https://mapservices.gis.saccounty.net/arcgis/rest/services/Cache/IMAGERY_2022_WEB_MERCATOR/MapServer
```

Zoom levels 0–24, Web Mercator (EPSG:3857), 256×256 PNG tiles.

---

## Hardware used

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 3060 12GB |
| Inference time per quadrant | ~20 minutes |
| Tiles per quadrant | ~13,000 at zoom 19 |
| Peak disk per quadrant | ~8–12 GB (with cleanup enabled) |
