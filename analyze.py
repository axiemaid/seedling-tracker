#!/usr/bin/env python3
"""
Seedling Growth Analyzer
========================

Processes captured images to measure green plant coverage.
No GUI required. No perspective correction needed (camera is fixed overhead).

Finds individual plant blobs via HSV segmentation, measures area,
logs to CSV, saves annotated images.

Usage:
    # Analyze all new images
    python3 analyze.py

    # Analyze a specific image
    python3 analyze.py --image images/2026-03-18_17-59-38.jpg

    # Reprocess everything
    python3 analyze.py --reprocess

    # Tune HSV thresholds on a sample (saves debug images)
    python3 analyze.py --tune --image images/2026-03-18_17-59-38.jpg
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

# HSV thresholds for bright yellow-green cotyledons under grow light
# Hue 25-80 covers yellow-green through green
# Sat 30+ excludes white/gray reflections
# Val 100+ excludes dark shadows
HSV_LOWER = (25, 30, 100)
HSV_UPPER = (80, 255, 255)

# Minimum blob area in pixels to count as a plant (filters noise)
MIN_BLOB_AREA = 80

# Morphological cleanup
MORPH_KERNEL = 5
MORPH_ITER = 2

# Image is 1024x768 (XGA). Define a crop region to exclude tray edges.
# Set to None to use the full image.
# Format: (x, y, w, h) or None
CROP_REGION = None


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
    """Extract timestamp from filename like 2026-03-18_17-59-38.jpg"""
    stem = Path(filepath).stem
    for fmt in ["%Y-%m-%d_%H-%M-%S", "%Y-%m-%d_%H%M%S"]:
        try:
            return datetime.strptime(stem, fmt)
        except ValueError:
            continue
    return datetime.fromtimestamp(os.path.getmtime(filepath))


def segment_green(img):
    """HSV threshold + morphological cleanup → binary mask of green pixels."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(HSV_LOWER), np.array(HSV_UPPER))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER)

    return mask


def find_blobs(mask):
    """Find contours above minimum area. Returns list of contour info dicts."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BLOB_AREA:
            continue
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else 0
        x, y, w, h = cv2.boundingRect(cnt)

        blobs.append({
            "area": int(area),
            "center_x": cx,
            "center_y": cy,
            "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
            "contour": cnt,
        })

    # Sort top-left to bottom-right (row-major by center position)
    blobs.sort(key=lambda b: (b["center_y"] // 80, b["center_x"]))

    return blobs


def check_quality(img):
    """Quick brightness/blur check. Returns (ok, brightness, sharpness)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ok = brightness > 40 and sharpness > 15
    return ok, brightness, sharpness


def draw_annotated(img, mask, blobs, stats):
    """Draw green overlay + blob labels on image."""
    out = img.copy()

    # Green tint on detected pixels
    overlay = np.zeros_like(out)
    overlay[mask > 0] = (0, 220, 0)
    out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)

    # Label each blob
    for i, b in enumerate(blobs):
        # Bounding box
        cv2.rectangle(out,
                      (b["bbox_x"], b["bbox_y"]),
                      (b["bbox_x"] + b["bbox_w"], b["bbox_y"] + b["bbox_h"]),
                      (255, 255, 255), 1)

        # ID + area
        label = f"{i+1}"
        cv2.putText(out, label,
                    (b["center_x"] - 6, b["center_y"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        area_label = f"{b['area']}px"
        cv2.putText(out, area_label,
                    (b["center_x"] - 15, b["center_y"] + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

    # Summary bar at top
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
    summary = (f"Plants: {stats['blob_count']} | "
               f"Green: {stats['coverage_pct']:.2f}% | "
               f"Total area: {stats['total_green_px']}px | "
               f"Brightness: {stats['brightness']:.0f}")
    cv2.putText(out, summary, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return out


def analyze_image(filepath):
    """Full analysis pipeline for one image. Returns stats dict or None."""
    img = cv2.imread(filepath)
    if img is None:
        print(f"  ERROR: can't load {filepath}")
        return None

    ts = parse_timestamp(filepath)
    ts_str = ts.strftime("%Y-%m-%d_%H-%M-%S")

    # Crop if configured
    if CROP_REGION:
        x, y, w, h = CROP_REGION
        img = img[y:y+h, x:x+w]

    # Quality check
    quality_ok, brightness, sharpness = check_quality(img)

    # Segment
    mask = segment_green(img)

    # Find blobs
    blobs = find_blobs(mask)

    # Stats
    total_px = img.shape[0] * img.shape[1]
    green_px = int(np.sum(mask > 0))
    coverage = round(green_px / total_px * 100, 3) if total_px > 0 else 0
    blob_areas = [b["area"] for b in blobs]

    stats = {
        "timestamp": ts_str,
        "blob_count": len(blobs),
        "total_green_px": green_px,
        "total_pixels": total_px,
        "coverage_pct": coverage,
        "mean_blob_area": round(np.mean(blob_areas), 1) if blob_areas else 0,
        "max_blob_area": max(blob_areas) if blob_areas else 0,
        "min_blob_area": min(blob_areas) if blob_areas else 0,
        "brightness": brightness,
        "sharpness": sharpness,
        "quality_ok": quality_ok,
    }

    # Skip dark/bad images from annotation but still log them
    if not quality_ok:
        flag = "DARK" if brightness < 40 else "BLUR"
        print(f"  {ts_str} | {flag} (brightness={brightness:.0f}, "
              f"sharpness={sharpness:.0f}) — skipping annotation")
    else:
        # Save annotated image
        annotated = draw_annotated(img, mask, blobs, stats)
        ann_path = os.path.join(ANNOTATED_DIR, f"{ts_str}.jpg")
        cv2.imwrite(ann_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Print summary
    q = "OK" if quality_ok else "WARN"
    print(f"  {ts_str} | plants={len(blobs)} | "
          f"green={coverage:.2f}% | "
          f"avg_area={stats['mean_blob_area']:.0f}px | "
          f"quality={q}")

    # Per-blob data for CSV
    stats["blobs"] = [{
        "id": i + 1,
        "area": b["area"],
        "cx": b["center_x"],
        "cy": b["center_y"],
    } for i, b in enumerate(blobs)]

    return stats


def log_to_csv(stats):
    """Append one capture's data to the CSV."""
    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "blob_count", "total_green_px", "coverage_pct",
                "mean_blob_area", "max_blob_area", "min_blob_area",
                "brightness", "sharpness", "quality_ok",
            ])
        writer.writerow([
            stats["timestamp"],
            stats["blob_count"],
            stats["total_green_px"],
            stats["coverage_pct"],
            stats["mean_blob_area"],
            stats["max_blob_area"],
            stats["min_blob_area"],
            stats["brightness"],
            stats["sharpness"],
            stats["quality_ok"],
        ])


def log_blobs_csv(stats):
    """Append per-blob data to a separate CSV for tracking individual plants."""
    blob_csv = os.path.join(OUTPUT_DIR, "blob_log.csv")
    file_exists = os.path.exists(blob_csv)

    with open(blob_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "blob_id", "area", "center_x", "center_y"])
        for b in stats.get("blobs", []):
            writer.writerow([
                stats["timestamp"], b["id"], b["area"], b["cx"], b["cy"],
            ])


def cmd_analyze(image_path=None, reprocess=False):
    """Process images."""
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
        # Wipe previous results
        for f in [CSV_FILE, os.path.join(OUTPUT_DIR, "blob_log.csv")]:
            if os.path.exists(f):
                os.remove(f)
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
            log_blobs_csv(stats)
            state["processed"].append(os.path.abspath(filepath))

    save_state(state)
    print(f"\nDone. Results in {OUTPUT_DIR}/")


def cmd_tune(image_path):
    """Save debug images showing each segmentation step. No GUI needed."""
    ensure_dirs()

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: can't load {image_path}")
        sys.exit(1)

    print(f"Tuning on: {image_path} ({img.shape[1]}x{img.shape[0]})")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Save HSV channels separately
    for i, name in enumerate(["hue", "saturation", "value"]):
        ch = hsv[:, :, i]
        path = os.path.join(TUNE_DIR, f"channel_{name}.jpg")
        cv2.imwrite(path, ch)
        print(f"  Saved {name} channel: {path} (range {ch.min()}-{ch.max()})")

    # Raw mask before cleanup
    raw_mask = cv2.inRange(hsv, np.array(HSV_LOWER), np.array(HSV_UPPER))
    cv2.imwrite(os.path.join(TUNE_DIR, "mask_raw.jpg"), raw_mask)

    # Cleaned mask
    mask = segment_green(img)
    cv2.imwrite(os.path.join(TUNE_DIR, "mask_clean.jpg"), mask)

    # Overlay
    overlay = img.copy()
    overlay[mask > 0] = (0, 255, 0)
    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    cv2.imwrite(os.path.join(TUNE_DIR, "overlay.jpg"), blended)

    # Blobs
    blobs = find_blobs(mask)
    print(f"\n  Found {len(blobs)} blobs with HSV {HSV_LOWER} - {HSV_UPPER}")
    print(f"  Green pixels: {np.sum(mask > 0)} / {mask.size} "
          f"({np.sum(mask > 0) / mask.size * 100:.2f}%)")

    if blobs:
        areas = [b["area"] for b in blobs]
        print(f"  Blob areas: min={min(areas)}, max={max(areas)}, "
              f"mean={np.mean(areas):.0f}, median={np.median(areas):.0f}")

    # Full annotated
    stats = {
        "blob_count": len(blobs),
        "total_green_px": int(np.sum(mask > 0)),
        "coverage_pct": round(np.sum(mask > 0) / mask.size * 100, 3),
        "brightness": float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()),
    }
    annotated = draw_annotated(img, mask, blobs, stats)
    cv2.imwrite(os.path.join(TUNE_DIR, "annotated.jpg"), annotated)

    print(f"\n  All debug images saved to {TUNE_DIR}/")
    print(f"  Check overlay.jpg — if plants aren't highlighted, adjust HSV_LOWER/HSV_UPPER in analyze.py")


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
