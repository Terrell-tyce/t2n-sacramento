"""
run_tile2net_sacramento.py

Runs the full tile2net pedestrian network extraction pipeline over the
Sacramento region using an NVIDIA GPU (tested on RTX 3060 12GB).

Since Sacramento is not in tile2net's built-in supported regions list,
this script handles tile downloading manually (via requests) from the
USGS National Map XYZ tile service, then feeds the local tile directory
to tile2net using the input_dir argument.

Pipeline:
  1. download  — fetches XYZ tiles from USGS into z/x/y.png folder structure
  2. generate  — builds tile2net's inference grid from the local tile dir
  3. inference — runs segmentation model → polygons → centerline network
  4. cleanup   — deletes intermediate files to save disk space
  5. merge     — combines sub-region shapefiles into single region-wide files

Requirements:
    Python 3.10 or 3.11  (tile2net breaks on 3.12+)
    pip install git+https://github.com/VIDA-NYU/tile2net
    pip install geopandas requests
"""

import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

import requests

# =============================================================================
# CONFIG — edit this block, don't touch anything below
# =============================================================================

# Where to write all outputs
OUTPUT_ROOT = Path("tile2net_output/sacramento_pedestrian_network")

# Zoom level 19 — confirmed available on Sacramento County's tile service below.
ZOOM = 19

# Sacramento County GIS — 2022 Color 3-Inch Orthos (public, no key needed).
# 24 zoom levels, Web Mercator, 256x256 PNG tiles.
# Source: https://mapservices.gis.saccounty.net/arcgis/rest/services/Cache/IMAGERY_2022_WEB_MERCATOR/MapServer
TILE_URL = "https://mapservices.gis.saccounty.net/arcgis/rest/services/Cache/IMAGERY_2022_WEB_MERCATOR/MapServer/tile/{z}/{y}/{x}"

# Seconds to wait between tile requests (be a polite downloader)
REQUEST_DELAY = 0.05

# Parallel tile download threads
DOWNLOAD_WORKERS = 8

# Run all sub-regions (True) or the full bbox as a single job (False).
USE_SUBREGIONS = True

# Used when USE_SUBREGIONS = False — City of Sacramento proper
FULL_REGION = ("sacramento_city", 38.4950, -121.5600, 38.6250, -121.4200)

# Sub-region grid split for City of Sacramento proper (USE_SUBREGIONS = True).
# At zoom 19 this is ~2,200 tiles per quadrant — manageable with cleanup on.
# Comment out rows to skip specific quadrants.
# Format: (name, lat_min, lon_min, lat_max, lon_max)
SUBREGIONS = [
    ("city_sw", 38.4950, -121.5600, 38.5600, -121.4870),  # extends slightly past midpoint
    ("city_se", 38.4950, -121.4930, 38.5600, -121.4200),  # starts slightly before midpoint
    ("city_nw", 38.5600, -121.5600, 38.6250, -121.4870),
    ("city_ne", 38.5600, -121.4930, 38.6250, -121.4200),
]

# Delete raw tiles and stitched images after each region finishes inference.
# Recommended: keeps peak disk usage around 8-12 GB instead of 30+ GB.
CLEANUP_AFTER_INFERENCE = True

# Merge all sub-region shapefiles into single region-wide files at the end.
MERGE_OUTPUTS = True

# =============================================================================
# END CONFIG
# =============================================================================

LAYER_PATTERNS = {
    "sidewalks":  "**/sidewalk*.shp",
    "crosswalks": "**/crosswalk*.shp",
    "footpaths":  "**/footpath*.shp",
    "network":    "**/network*.shp",
}

CLEANUP_DIRS = ["tiles", "stitched", "predictions"]


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def check_dependencies():
    missing = []
    for pkg in ("tile2net", "geopandas", "requests"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("\nERROR: Missing packages:", ", ".join(missing))
        print("Install with:")
        print("  pip install git+https://github.com/VIDA-NYU/tile2net geopandas requests")
        sys.exit(1)

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            log(f"GPU: {name} ({vram:.1f} GB VRAM)")
        else:
            log("WARNING: No CUDA GPU — inference will be very slow on CPU.")
    except ImportError:
        log("WARNING: torch not importable, cannot verify GPU.")


# ---------------------------------------------------------------------------
# XYZ tile math
# ---------------------------------------------------------------------------

def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon to XYZ tile x/y at a given zoom level."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def tiles_for_bbox(lat_min, lon_min, lat_max, lon_max, zoom):
    """Return all (x, y) tile coords covering a bounding box at zoom."""
    x_min, y_max = lat_lon_to_tile(lat_min, lon_min, zoom)  # SW → top-left in tile coords
    x_max, y_min = lat_lon_to_tile(lat_max, lon_max, zoom)  # NE → bottom-right
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            yield x, y


# ---------------------------------------------------------------------------
# Tile downloader
# ---------------------------------------------------------------------------

def download_tile(args):
    """Download a single tile. Returns (x, y, success)."""
    x, y, zoom, tile_dir, url_template = args
    out_path = tile_dir / str(zoom) / str(x) / f"{y}.png"
    if out_path.exists():
        return x, y, True  # already downloaded

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = url_template.format(z=zoom, x=x, y=y)  # template uses {y}/{x} = row/col

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 tile2net-pipeline/1.0"})
            if resp.status_code == 200:
                out_path.write_bytes(resp.content)
                time.sleep(REQUEST_DELAY)
                return x, y, True
            else:
                time.sleep(1)
        except Exception:
            time.sleep(2)

    return x, y, False


def download_tiles(lat_min, lon_min, lat_max, lon_max, tile_dir: Path):
    """Download all tiles for the bbox into tile_dir/z/x/y.png structure."""
    tile_list = list(tiles_for_bbox(lat_min, lon_min, lat_max, lon_max, ZOOM))
    total = len(tile_list)
    log(f"  Downloading {total} tiles at zoom {ZOOM}...")

    args = [(x, y, ZOOM, tile_dir, TILE_URL) for x, y in tile_list]
    failed = []

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(download_tile, a): a for a in args}
        done = 0
        for future in as_completed(futures):
            x, y, ok = future.result()
            done += 1
            if not ok:
                failed.append((x, y))
            if done % 500 == 0 or done == total:
                log(f"  Progress: {done}/{total} tiles ({len(failed)} failed)")

    if failed:
        log(f"  WARNING: {len(failed)} tiles failed to download — results may have gaps.")

    log(f"  Download complete. Tiles in: {tile_dir}")


# ---------------------------------------------------------------------------
# tile2net stages
# ---------------------------------------------------------------------------

def run_generate(name, lat_min, lon_min, lat_max, lon_max, out_dir, tile_dir):
    """Use the Python API directly — input_dir is not exposed in the CLI."""
    from tile2net import Raster
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    # tile_dir/z/x/y.png — tile2net expects the pattern with literal x/y as placeholders
    input_dir_pattern = str(tile_dir / str(ZOOM) / "x" / "y.png")
    log(f"  Building tile2net grid (input_dir={input_dir_pattern})...")
    raster = Raster(
        location=bbox,
        name=name,
        output_dir=str(out_dir),
        input_dir=input_dir_pattern,
        zoom=ZOOM,
    )
    raster.generate(2)   # stitch_step=2 balances memory vs speed
    return raster


def run_inference(name, out_dir, raster=None):
    """Run inference. Reuses raster object if available, else reconstructs."""
    if raster is None:
        from tile2net import Raster
        bbox_stub = f"{name}"  # fallback — raster should always be passed
        raster = Raster.from_info(str(out_dir / name / f"{name}.json"))
    log(f"  Running segmentation inference...")
    raster.project.num_workers = 0  # disable multiprocessing to prevent immediate gridlock
    raster.inference()


def cleanup_intermediates(out_dir: Path):
    freed = 0
    for dirname in CLEANUP_DIRS:
        target = out_dir / dirname
        if target.exists():
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            shutil.rmtree(target)
            freed += size
            log(f"  Cleaned: {dirname}/ ({size / 1e6:.0f} MB freed)")
    if freed:
        log(f"  Total freed: {freed / 1e6:.0f} MB")
    else:
        log("  Nothing to clean up.")


def run_region(name, lat_min, lon_min, lat_max, lon_max):
    out_dir  = OUTPUT_ROOT / name
    tile_dir = out_dir / "tiles_raw"
    # Check for existing completed output before doing any work
    network_dir = out_dir / name / "network"
    if network_dir.exists() and any(network_dir.iterdir()):
        log(f"  Skipping {name} — network output already exists at {network_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    log("=" * 56)
    log(f"  Region : {name}")
    log(f"  BBox   : ({lat_min}, {lon_min}) → ({lat_max}, {lon_max})")
    log(f"  Output : {out_dir}")
    log("=" * 56)

    log("[1/3] Downloading tiles...")
    download_tiles(lat_min, lon_min, lat_max, lon_max, tile_dir)

    log("[2/3] Building tile2net grid...")
    raster = run_generate(name, lat_min, lon_min, lat_max, lon_max, out_dir, tile_dir)

    log("[3/3] Running segmentation inference...")
    run_inference(name, out_dir, raster=raster)

    if CLEANUP_AFTER_INFERENCE:
        log("[+] Cleaning up intermediate files...")
        cleanup_intermediates(out_dir)

    log(f"✓ Done: {name}")


def merge_outputs():
    import geopandas as gpd
    import pandas as pd

    merged_dir = OUTPUT_ROOT / "merged"
    merged_dir.mkdir(exist_ok=True)

    print()
    log("=" * 56)
    log("  Merging sub-region shapefiles...")
    log("=" * 56)

    for layer_name, pattern in LAYER_PATTERNS.items():
        files = [f for f in OUTPUT_ROOT.glob(pattern) if "merged" not in f.parts]
        if not files:
            log(f"  {layer_name}: no files found, skipping.")
            continue

        gdfs = []
        for f in files:
            try:
                gdfs.append(gpd.read_file(f))
            except Exception as e:
                log(f"  WARNING: could not read {f.name}: {e}")

        if not gdfs:
            continue

        merged = pd.concat(gdfs, ignore_index=True)
        out_path = merged_dir / f"sacramento_{layer_name}.shp"
        merged.to_file(out_path)
        log(f"  ✓ {layer_name}: {len(merged):,} features → {out_path.name}")

    log(f"Merge complete. Files in: {merged_dir}")


def main():
    print()
    log("=" * 56)
    log("  tile2net — Sacramento Pedestrian Network Pipeline")
    log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 56)

    check_dependencies()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"Output root: {OUTPUT_ROOT.resolve()}")

    if USE_SUBREGIONS:
        log(f"Mode: sub-regions ({len(SUBREGIONS)} quadrants)")
        for region in SUBREGIONS:
            run_region(*region)
    else:
        log("Mode: full region (single job)")
        run_region(*FULL_REGION)

    if MERGE_OUTPUTS:
        merge_outputs()

    print()
    log("=" * 56)
    log(f"  Pipeline complete: {datetime.now().strftime('%H:%M:%S')}")
    log(f"  Outputs in: {OUTPUT_ROOT.resolve()}")
    log("=" * 56)


if __name__ == "__main__":
    main()