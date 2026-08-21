# Sacramento Crossing Extraction

Automated extraction of crosswalks from aerial imagery using [tile2net](https://github.com/VIDA-NYU/tile2net), applied to the City of Sacramento. Sacramento is not in tile2net's built-in supported regions, so this pipeline handles tile downloading manually from Sacramento County's public imagery service and feeds the local tiles into tile2net for inference.

The primary research interest is measuring crosswalk lengths and evaluating how well tile2net performs on a western US city with wider streets and more suburban land use than the east coast cities its model was trained on.

> **Handoff note.** This was built and run on a personal machine with an RTX 3060. It has not been run on SACOG hardware. The code is unchanged in substance, but **a CUDA-capable GPU is a hard requirement for the inference stage** — see [Requirements](#requirements). Start with [Start here](#start-here--your-first-run) rather than running the pipeline as-is.

---

## Start here — your first run

A full run downloads **~52,000 tiles** and takes hours. Do not start there. Confirm the whole chain works on a handful of blocks first.

**1. Check you have a usable GPU.**

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If this prints `False`, stop and read [Requirements](#requirements). The download and merge stages work without a GPU; inference effectively will not.

**2. Apply the two tile2net patches.** They are not optional — both crash on Sacramento data. See [Known issues and patches](#known-issues-and-patches).

**3. Run a smoke test.** In `run_tile2net_sacramento.py`, temporarily replace the `SUBREGIONS` list with a single small area:

```python
SUBREGIONS = [
    ("smoke_test", 38.5750, -121.4950, 38.5800, -121.4900),
]
```

That is roughly six by six blocks of downtown Sacramento — **88 tiles instead of 52,000**, a few minutes end to end. Then:

```bash
python run_tile2net_sacramento.py
```

**What success looks like:** the log reaches `✓ Done: smoke_test`, and `tile2net_output/sacramento_pedestrian_network/merged/` contains at least `sacramento_network.shp` with a non-zero feature count.

**4. Only then restore the four real quadrants** and budget several hours.

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
3. **Inference** — runs the model to produce polygons, then derives a centerline pedestrian network
4. **Cleanup** — optionally deletes raw tiles and stitched images to recover disk space
5. **Merge** — concatenates the quadrant shapefiles into single city-wide files

The city is split into four quadrants (SW, SE, NW, NE) to keep memory and disk usage manageable at zoom 19. Each quadrant is ~13,000 tiles; ~52,000 in total.

---

## Requirements

- **Python 3.10 or 3.11.** tile2net breaks on 3.12+.
- **A CUDA-enabled GPU.** tile2net's own documentation states this outright. CPU inference is not a practical fallback at this scale — it is slower by orders of magnitude, not by a factor of two.
- **Disk:** roughly 8–12 GB free per quadrant during inference with cleanup enabled; 30+ GB without it.

```bash
pip install git+https://github.com/VIDA-NYU/tile2net
pip install geopandas requests folium
```

tile2net pins some fairly specific versions upstream (CUDA 11.7, PyTorch 2.0.0, Shapely 2.0.0). If installation fights you, that pinning is usually why.

**If there is no GPU available at SACOG,** the realistic options are: run it on a cloud GPU instance, request a workstation with a discrete NVIDIA card, or treat the existing merged shapefiles as the deliverable and skip re-running inference. The download, merge, and visualization stages all run fine CPU-only.

---

## Usage

### 1. Run the pipeline

```bash
python run_tile2net_sacramento.py
```

All configuration is in the `CONFIG` block at the top of the script — output directory, bounding boxes, zoom level, thread count, and cleanup/merge flags. No arguments needed.

If the pipeline is interrupted, re-running skips any quadrant whose network output directory already exists. To force a re-run of one quadrant, delete its output folder and restart.

### 2. Visualize

```bash
python visualize_sacramento.py
```

Generates `sacramento_crosswalks.html` — a self-contained interactive map that opens in any browser with no additional installs. Shows crosswalk detections only, clipped to a downtown/midtown study area for performance. To change the area, edit `CLIP_BOUNDS` at the top of the script.

Map features: a toggle between Sacramento County 2022 imagery (the imagery inference was run against) and Esri World Imagery; crosswalk polygons and centerlines as independent layers; a measurement tool; minimap and fullscreen controls.

---

## Outputs

Merged shapefiles land in `tile2net_output/sacramento_pedestrian_network/merged/`. The file names come from `LAYER_PATTERNS` in the pipeline script, one output per matched layer:

| File | Contents |
|---|---|
| `sacramento_network.shp` | pedestrian network centerlines |
| `sacramento_sidewalks.shp` | sidewalk features, if tile2net emitted a matching layer |
| `sacramento_crosswalks.shp` | crosswalk features, if emitted |
| `sacramento_footpaths.shp` | footpath features, if emitted |

Layers that tile2net did not produce are skipped with a log line rather than failing the run, so do not be alarmed if fewer than four files appear.

Every feature carries:

- `f_type` — `sidewalk`, `crosswalk`, `sidewalk_connection`, or `road`
- `quadrant` — which sub-region the feature came from, added during the merge step for provenance

---

## Known gaps

Things the next person should know, rather than discover.

- **`sacramento_polygons.shp` is not produced.** `visualize_sacramento.py` will use it if present, but nothing in `merge_outputs()` writes it — `LAYER_PATTERNS` has no polygon entry. The map now degrades to centerlines-only and prints a note instead of crashing on a fresh run. If the raw segmentation polygons are wanted in the merged output, add a pattern for whichever polygon shapefile tile2net actually emits, and confirm the glob against a real run before trusting it.
- **Not run on SACOG hardware.** Every performance figure below is from a personal RTX 3060.
- **The two tile2net patches are applied by hand** to files inside the installed package. A fresh `pip install` silently reverts them. The editable install below is the durable fix.

---

## Known issues and patches

tile2net has several bugs that surface on real-world data outside its supported regions. Two files in the tile2net package need patching before running.

### `pednet.py` — Qhull precision crash during centerline generation

Degenerate sidewalk polygons can cause a `scipy.spatial.qhull.QhullError` in `to_cline()`. Wrap the function body in a `try/except` that returns `None` on failure, and add a `None` guard in `create_lines()` before the result is used.

File: `<env>/Lib/site-packages/tile2net/raster/pednet.py`

### `geodata_utils.py` — GEOS TopologyException during polygon union

Negative buffer operations (the erode step in `buffer_union_erode`) can produce self-intersecting geometries that cause `shapely.errors.GEOSException: TopologyException` during the subsequent union. Fix by adding a `buffer(0)` repair pass after the erode step and inside `unary_multi` before dissolve.

File: `<env>/Lib/site-packages/tile2net/raster/tile_utils/geodata_utils.py`

To make the patches survive reinstalls, clone tile2net and install it as editable:

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

Zoom levels 0–24, Web Mercator (EPSG:3857), 256×256 PNG tiles. Note that the URL template is `/tile/{z}/{y}/{x}` — row before column, the opposite of the usual `{z}/{x}/{y}` convention, and an easy thing to get wrong when pointing this at a different service.

---

## tile2net reference

Upstream project: <https://github.com/VIDA-NYU/tile2net>

The two documents worth reading before modifying this pipeline:

- **`DATA_PREPARE.md`** — how tile2net expects local image tiles to be laid out. This is the contract the download stage is written against, and the reason `input_dir` is passed the literal `.../19/x/y.png` pattern rather than a real path.
- **`BASICS.md`** — core concepts and the list of built-in supported regions. Sacramento is not among them, which is the entire reason this repo exists.

Also useful: `examples/inference.ipynb` for the Python API surface, since `input_dir` is not exposed through tile2net's CLI and has to be driven from Python.

Paper: Hosseini, M., Sevtsuk, A., Miranda, F., Cesar Jr, R. M., & Silva, C. T. (2023). *Mapping the walk: A scalable computer vision approach for generating sidewalk network datasets from aerial imagery.* Computers, Environment and Urban Systems, 101, 101950.

---

## Performance reference

Measured on the machine this was developed on. Treat as a rough scaling guide, not a spec.

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 3060 12GB |
| Inference time per quadrant | ~20 minutes |
| Tiles per quadrant | ~13,000 at zoom 19 |
| Peak disk per quadrant | ~8–12 GB (with cleanup enabled) |

**Scaling down.** With less VRAM, lower the stitch step passed to `raster.generate(2)`, or split the region into more, smaller entries in `SUBREGIONS` — the quadrant split exists precisely so this is a config change rather than a code change. Dropping to zoom 18 cuts tile count roughly fourfold but coarsens the imagery, and crosswalk detection depends on that resolution; earlier project notes already flag 50 cm imagery as marginal for this task, so trading resolution away is not free.
