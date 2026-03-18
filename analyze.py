#!/usr/bin/env python3
"""
Seedling Growth Analyzer
========================

Fixed 4x7 grid — each cell is one plant slot, consistent across captures.
No GUI required. Camera is fixed overhead.

Usage:
    python3 analyze.py                    # Analyze all new images
    python3 analyze.py --image FILE       # Analyze one image
    python3 analyze.py --reprocess        # Redo all
    python3 analyze.py --tune --image FILE  # Save debug images for HSV tuning
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ==================== CONFIG ====================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "analysis")
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")
TUNE_DIR = os.path.join(OUTPUT_DIR, "tune")
CSV_FILE = os.path.join(OUTPUT_DIR, "growth_log.csv")
STATE_FILE = os.path.join(OUTPUT_DIR, ".state.json")

# Grid: 4 rows x 7 columns = 28 plant slots
GRID_ROWS = 4
GRID_COLS = 6

# HSV thresholds for yellow-green cotyledons under grow light
HSV_LOWER = (25, 30, 100)
HSV_UPPER = (80, 255, 255)

# Morphological cleanup
MORPH_KERNEL = 5
MORPH_ITER = 2

# Margin inside each cell to avoid counting edge pixels (% of cell size)
CELL_MARGIN_PCT = 5

# Crop region to exclude tray edges: (x, y, w, h) or None
# Applied before grid — grid boundaries are relative to cropped image
CROP_REGION = (70, 50, 860, 660)

# Custom column boundaries (x positions). Set to None for even spacing.
# Adjust these to align grid lines with actual plant columns.
# 7 values for 6 columns: [left_edge, col1|col2, col2|col3, ..., right_edge]
CUSTOM_COL_BOUNDS = [0, 123, 267, 410, 553, 697, 860]

# Custom row boundaries. Set to None for even spacing.
CUSTOM_ROW_BOUNDS = None


def ensure_dirs():
    for d in [OUTPUT_DIR, ANNOTATED_DIR, TUNE_DIR]:
        os.makedirs(d, exist_ok=True)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"processed": []}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def parse_timestamp(filepath):
    stem = Path(filepath).stem
    for fmt in ["%Y-%m-%d_%H-%M", "%Y-%m-%d_%H-%M-%S", "%Y-%m-%d_%H%M%S"]:
        try:
            return datetime.strptime(stem, fmt)
        except ValueError:
            continue
    return datetime.fromtimestamp(os.path.getmtime(filepath))


def segment_green(img):
    """HSV threshold + morphological cleanup → binary mask."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(HSV_LOWER), np.array(HSV_UPPER))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER)

    return mask


def get_grid_cells(img_h, img_w):
    """
    Compute fixed grid cells with optional custom boundaries.
    Each cell has a consistent ID (1-N).
    """
    # Column boundaries
    if CUSTOM_COL_BOUNDS:
        col_bounds = CUSTOM_COL_BOUNDS
    else:
        col_bounds = [round(c * img_w / GRID_COLS) for c in range(GRID_COLS + 1)]

    # Row boundaries
    if CUSTOM_ROW_BOUNDS:
        row_bounds = CUSTOM_ROW_BOUNDS
    else:
        row_bounds = [round(r * img_h / GRID_ROWS) for r in range(GRID_ROWS + 1)]

    cells = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell_id = r * GRID_COLS + c + 1
            x1 = col_bounds[c]
            x2 = col_bounds[c + 1]
            y1 = row_bounds[r]
            y2 = row_bounds[r + 1]
            cw = x2 - x1
            ch = y2 - y1
            mx = int(cw * CELL_MARGIN_PCT / 100)
            my = int(ch * CELL_MARGIN_PCT / 100)
            cells.append({
                "id": cell_id,
                "row": r,
                "col": c,
                "x": x1 + mx, "y": y1 + my,
                "w": cw - 2 * mx, "h": ch - 2 * my,
            })
    return cells


def measure_cells(mask, cells):
    """Measure green pixel coverage in each grid cell."""
    results = []
    for cell in cells:
        x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
        cell_mask = mask[y:y+h, x:x+w]
        green_px = int(np.sum(cell_mask > 0))
        total_px = w * h
        coverage = round(green_px / total_px * 100, 3) if total_px > 0 else 0

        results.append({
            "id": cell["id"],
            "row": cell["row"],
            "col": cell["col"],
            "green_pixels": green_px,
            "total_pixels": total_px,
            "coverage_pct": coverage,
        })
    return results


def check_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ok = brightness > 40 and sharpness > 15
    return ok, brightness, sharpness


def draw_annotated(img, mask, cells, measurements, stats):
    """Draw grid overlay with per-cell coverage labels."""
    out = img.copy()
    h, w = out.shape[:2]

    # Green tint on detected pixels
    overlay = np.zeros_like(out)
    overlay[mask > 0] = (0, 220, 0)
    out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)

    # Grid boundaries
    if CUSTOM_COL_BOUNDS:
        col_bounds = CUSTOM_COL_BOUNDS
    else:
        col_bounds = [round(c * w / GRID_COLS) for c in range(GRID_COLS + 1)]
    if CUSTOM_ROW_BOUNDS:
        row_bounds = CUSTOM_ROW_BOUNDS
    else:
        row_bounds = [round(r * h / GRID_ROWS) for r in range(GRID_ROWS + 1)]

    # Grid lines
    for x in col_bounds:
        cv2.line(out, (x, 0), (x, h), (100, 100, 100), 1)
    for y in row_bounds:
        cv2.line(out, (0, y), (w, y), (100, 100, 100), 1)

    # Per-cell labels
    for m in measurements:
        cx = (col_bounds[m["col"]] + col_bounds[m["col"] + 1]) // 2
        cy = (row_bounds[m["row"]] + row_bounds[m["row"] + 1]) // 2

        # Color code: white if low coverage, green if high
        color = (0, 255, 255) if m["green_pixels"] > 100 else (150, 150, 150)

        cv2.putText(out, f"#{m['id']}",
                    (cx - 12, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(out, f"{m['green_pixels']}px",
                    (cx - 18, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Summary bar
    cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
    active = sum(1 for m in measurements if m["coverage_pct"] > 0.5)
    summary = (f"Active plants: {active}/{GRID_ROWS * GRID_COLS} | "
               f"Avg coverage: {stats['avg_coverage']:.2f}% | "
               f"Total green: {stats['total_green_px']}px | "
               f"Brightness: {stats['brightness']:.0f}")
    cv2.putText(out, summary, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    return out


def analyze_image(filepath):
    """Full analysis pipeline for one image."""
    img = cv2.imread(filepath)
    if img is None:
        print(f"  ERROR: can't load {filepath}")
        return None

    ts = parse_timestamp(filepath)
    ts_str = ts.strftime("%Y-%m-%d_%H-%M-%S")

    # Crop to tray interior
    if CROP_REGION:
        cx, cy, cw, ch = CROP_REGION
        img = img[cy:cy+ch, cx:cx+cw]

    h, w = img.shape[:2]

    # Quality check
    quality_ok, brightness, sharpness = check_quality(img)

    # Segment
    mask = segment_green(img)

    # Measure fixed grid cells
    cells = get_grid_cells(h, w)
    measurements = measure_cells(mask, cells)

    # Aggregate stats
    total_px = h * w
    green_px = int(np.sum(mask > 0))
    coverages = [m["coverage_pct"] for m in measurements]
    active_cells = [m for m in measurements if m["coverage_pct"] > 0.5]

    stats = {
        "timestamp": ts_str,
        "total_green_px": green_px,
        "total_coverage_pct": round(green_px / total_px * 100, 3),
        "avg_coverage": round(np.mean(coverages), 3),
        "active_plants": len(active_cells),
        "brightness": brightness,
        "sharpness": sharpness,
        "quality_ok": quality_ok,
        "measurements": measurements,
    }

    if not quality_ok:
        flag = "DARK" if brightness < 40 else "BLUR"
        print(f"  {ts_str} | {flag} (brightness={brightness:.0f}) — skipping annotation")
    else:
        annotated = draw_annotated(img, mask, cells, measurements, stats)
        ann_path = os.path.join(ANNOTATED_DIR, f"{ts_str}.jpg")
        cv2.imwrite(ann_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    q = "OK" if quality_ok else "WARN"
    print(f"  {ts_str} | active={len(active_cells)}/{GRID_ROWS * GRID_COLS} | "
          f"avg={stats['avg_coverage']:.2f}% | "
          f"total_green={stats['total_coverage_pct']:.2f}% | "
          f"quality={q}")

    return stats


def log_to_csv(stats):
    """Append one row per cell per capture."""
    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "cell_id", "row", "col",
                "green_pixels", "total_pixels", "coverage_pct",
                "brightness", "quality_ok",
            ])
        for m in stats["measurements"]:
            writer.writerow([
                stats["timestamp"],
                m["id"], m["row"], m["col"],
                m["green_pixels"], m["total_pixels"], m["coverage_pct"],
                round(stats["brightness"], 1),
                stats["quality_ok"],
            ])


def cmd_analyze(image_path=None, reprocess=False):
    ensure_dirs()

    if image_path:
        files = [image_path]
    else:
        files = sorted([
            os.path.join(IMAGES_DIR, f)
            for f in os.listdir(IMAGES_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    if not files:
        print("No images found.")
        return

    state = load_state()
    if reprocess:
        if os.path.exists(CSV_FILE):
            os.remove(CSV_FILE)
        state = {"processed": []}

    new_files = [f for f in files if os.path.abspath(f) not in state["processed"]]

    if not new_files:
        print("All images already analyzed. Use --reprocess to redo.")
        return

    print(f"Analyzing {len(new_files)} image(s)...\n")

    for filepath in new_files:
        stats = analyze_image(filepath)
        if stats:
            log_to_csv(stats)
            state["processed"].append(os.path.abspath(filepath))

    save_state(state)
    print(f"\nDone. Results in {OUTPUT_DIR}/")


def cmd_tune(image_path):
    """Save debug images for HSV tuning. No GUI."""
    ensure_dirs()

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: can't load {image_path}")
        sys.exit(1)

    if CROP_REGION:
        cx, cy, cw, ch = CROP_REGION
        img = img[cy:cy+ch, cx:cx+cw]

    h, w = img.shape[:2]
    print(f"Tuning on: {image_path} ({w}x{h}, cropped={CROP_REGION is not None})")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV channels
    for i, name in enumerate(["hue", "saturation", "value"]):
        ch = hsv[:, :, i]
        path = os.path.join(TUNE_DIR, f"channel_{name}.jpg")
        cv2.imwrite(path, ch)
        print(f"  {name}: range {ch.min()}-{ch.max()}")

    # Masks
    raw_mask = cv2.inRange(hsv, np.array(HSV_LOWER), np.array(HSV_UPPER))
    cv2.imwrite(os.path.join(TUNE_DIR, "mask_raw.jpg"), raw_mask)

    mask = segment_green(img)
    cv2.imwrite(os.path.join(TUNE_DIR, "mask_clean.jpg"), mask)

    # Overlay
    overlay = img.copy()
    overlay[mask > 0] = (0, 255, 0)
    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    cv2.imwrite(os.path.join(TUNE_DIR, "overlay.jpg"), blended)

    # Grid measurement
    cells = get_grid_cells(h, w)
    measurements = measure_cells(mask, cells)
    active = [m for m in measurements if m["coverage_pct"] > 0.5]

    print(f"\n  HSV: {HSV_LOWER} - {HSV_UPPER}")
    print(f"  Green pixels: {np.sum(mask > 0)} / {mask.size} "
          f"({np.sum(mask > 0) / mask.size * 100:.2f}%)")
    print(f"  Active cells: {len(active)}/{GRID_ROWS * GRID_COLS}")
    print(f"\n  Per-cell coverage:")
    for m in measurements:
        marker = "█" if m["coverage_pct"] > 0.5 else "·"
        print(f"    #{m['id']:2d} (r{m['row']}c{m['col']}): "
              f"{m['coverage_pct']:6.2f}% {marker}")

    # Annotated with grid
    stats = {
        "avg_coverage": np.mean([m["coverage_pct"] for m in measurements]),
        "total_green_px": int(np.sum(mask > 0)),
        "brightness": float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()),
    }
    annotated = draw_annotated(img, mask, cells, measurements, stats)
    cv2.imwrite(os.path.join(TUNE_DIR, "annotated.jpg"), annotated)

    print(f"\n  Debug images saved to {TUNE_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Seedling Growth Analyzer")
    parser.add_argument("--image", type=str, help="Analyze a specific image")
    parser.add_argument("--reprocess", action="store_true", help="Reprocess all images")
    parser.add_argument("--tune", action="store_true", help="Save debug images for HSV tuning")
    args = parser.parse_args()

    if args.tune:
        if not args.image:
            print("ERROR: --tune requires --image")
            sys.exit(1)
        cmd_tune(args.image)
    else:
        cmd_analyze(args.image, args.reprocess)


if __name__ == "__main__":
    main()
