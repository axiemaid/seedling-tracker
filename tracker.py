#!/usr/bin/env python3
"""
Seedling Tracker - Mac Mini Application
========================================

Captures images from ESP32-CAM, performs perspective correction,
segments green plant area, and logs growth data for 28 hydroponic seedlings.

Usage:
    First run (calibration):
        python3 tracker.py --calibrate

    Scheduled capture:
        python3 tracker.py --capture

    Generate growth report:
        python3 tracker.py --report

    Live preview (for aiming / focusing):
        python3 tracker.py --preview
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ==================== CONFIG ====================

CONFIG_FILE = "config.json"
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANNOTATED_DIR = os.path.join(DATA_DIR, "annotated")
CSV_FILE = os.path.join(DATA_DIR, "growth_log.csv")

DEFAULT_CONFIG = {
    "esp32_ip": "192.168.151.219",
    "esp32_port": 80,
    "grid_rows": 4,
    "grid_cols": 7,
    "perspective_corners": None,       # Set during calibration
    "warped_width": 1400,              # Output width after perspective fix
    "warped_height": 800,              # Output height after perspective fix
    "hsv_lower": [25, 30, 30],         # Lower HSV bound for green detection
    "hsv_upper": [95, 255, 255],       # Upper HSV bound for green detection
    "cell_margin_pct": 10,             # % margin inside each cell to avoid edges
    "capture_timeout_s": 10,
    "retry_count": 3,
    "retry_delay_s": 2,
}


def ensure_dirs():
    """Create data directories if they don't exist."""
    for d in [DATA_DIR, IMAGES_DIR, ANNOTATED_DIR]:
        os.makedirs(d, exist_ok=True)


def load_config():
    """Load config from file, or create default."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        # Merge with defaults for any new keys
        for k, v in DEFAULT_CONFIG.items():
            if k not in cfg:
                cfg[k] = v
        return cfg
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    """Save config to file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config saved to {CONFIG_FILE}")


def capture_image(cfg):
    """Fetch a JPEG from the ESP32-CAM."""
    url = f"http://{cfg['esp32_ip']}:{cfg['esp32_port']}/capture"

    for attempt in range(cfg["retry_count"]):
        try:
            import urllib.request
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=cfg["capture_timeout_s"]) as resp:
                img_bytes = resp.read()
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    print(f"Captured image: {img.shape[1]}x{img.shape[0]}")
                    return img
                print(f"Attempt {attempt+1}: decode failed")
        except Exception as e:
            print(f"Attempt {attempt+1}: {e}")
        if attempt < cfg["retry_count"] - 1:
            time.sleep(cfg["retry_delay_s"])

    print("ERROR: Failed to capture image from ESP32-CAM")
    return None


def warp_perspective(img, cfg):
    """Apply perspective correction using calibrated corner points."""
    corners = cfg.get("perspective_corners")
    if corners is None:
        print("WARNING: No perspective calibration. Using raw image.")
        return img

    src = np.array(corners, dtype=np.float32)
    w, h = cfg["warped_width"], cfg["warped_height"]
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped


def segment_green(img, cfg):
    """
    Segment green plant pixels from background using HSV thresholding.
    Returns binary mask (255 = plant, 0 = background).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg["hsv_lower"], dtype=np.uint8)
    upper = np.array(cfg["hsv_upper"], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup: remove noise, fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def get_cell_regions(cfg):
    """
    Compute the bounding box for each seedling cell in the warped image.
    Returns list of (row, col, x, y, w, h) for each of 28 cells.
    """
    rows, cols = cfg["grid_rows"], cfg["grid_cols"]
    img_w, img_h = cfg["warped_width"], cfg["warped_height"]
    margin_pct = cfg["cell_margin_pct"] / 100.0

    cell_w = img_w / cols
    cell_h = img_h / rows

    mx = int(cell_w * margin_pct)
    my = int(cell_h * margin_pct)

    regions = []
    for r in range(rows):
        for c in range(cols):
            x = int(c * cell_w) + mx
            y = int(r * cell_h) + my
            w = int(cell_w) - 2 * mx
            h = int(cell_h) - 2 * my
            regions.append((r, c, x, y, w, h))
    return regions


def measure_cells(mask, cfg):
    """
    Measure green pixel count in each cell region.
    Returns list of dicts with row, col, cell_id, green_pixels, cell_pixels, coverage_pct.
    """
    regions = get_cell_regions(cfg)
    results = []

    for r, c, x, y, w, h in regions:
        cell_mask = mask[y:y+h, x:x+w]
        green_px = int(np.sum(cell_mask > 0))
        total_px = w * h
        coverage = round(green_px / total_px * 100, 2) if total_px > 0 else 0

        cell_id = r * cfg["grid_cols"] + c + 1  # 1-indexed
        results.append({
            "cell_id": cell_id,
            "row": r,
            "col": c,
            "green_pixels": green_px,
            "total_pixels": total_px,
            "coverage_pct": coverage,
        })

    return results


def draw_annotated(warped, mask, measurements, cfg):
    """Draw grid overlay and measurements on the warped image."""
    annotated = warped.copy()
    regions = get_cell_regions(cfg)

    # Green overlay for detected plant area
    green_overlay = np.zeros_like(annotated)
    green_overlay[mask > 0] = (0, 255, 0)
    annotated = cv2.addWeighted(annotated, 0.7, green_overlay, 0.3, 0)

    # Draw grid lines
    rows, cols = cfg["grid_rows"], cfg["grid_cols"]
    img_w, img_h = cfg["warped_width"], cfg["warped_height"]
    cell_w = img_w // cols
    cell_h = img_h // rows

    for c in range(cols + 1):
        x = c * cell_w
        cv2.line(annotated, (x, 0), (x, img_h), (200, 200, 200), 1)
    for r in range(rows + 1):
        y = r * cell_h
        cv2.line(annotated, (0, y), (img_w, y), (200, 200, 200), 1)

    # Label each cell with ID and coverage
    for m in measurements:
        r, c = m["row"], m["col"]
        cx = int((c + 0.5) * cell_w)
        cy = int((r + 0.5) * cell_h)

        label = f"#{m['cell_id']}"
        pct_label = f"{m['coverage_pct']:.1f}%"

        cv2.putText(annotated, label, (cx - 20, cy - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(annotated, pct_label, (cx - 25, cy + 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return annotated


def log_to_csv(measurements, timestamp):
    """Append measurements to the CSV growth log."""
    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "cell_id", "row", "col",
                             "green_pixels", "total_pixels", "coverage_pct"])
        for m in measurements:
            writer.writerow([
                timestamp,
                m["cell_id"], m["row"], m["col"],
                m["green_pixels"], m["total_pixels"], m["coverage_pct"]
            ])

    print(f"Logged {len(measurements)} cells to {CSV_FILE}")


# ==================== COMMANDS ====================

def cmd_calibrate(cfg):
    """Interactive calibration: set ESP32 IP, pick grid corners, tune HSV thresholds."""
    print("=" * 50)
    print("SEEDLING TRACKER CALIBRATION")
    print("=" * 50)

    # Step 1: ESP32 IP
    current_ip = cfg["esp32_ip"]
    new_ip = input(f"\nESP32-CAM IP address [{current_ip}]: ").strip()
    if new_ip:
        cfg["esp32_ip"] = new_ip

    # Step 2: Grid dimensions
    print(f"\nCurrent grid: {cfg['grid_rows']} rows x {cfg['grid_cols']} cols = {cfg['grid_rows']*cfg['grid_cols']} cells")
    change_grid = input("Change grid dimensions? [y/N]: ").strip().lower()
    if change_grid == "y":
        cfg["grid_rows"] = int(input("  Rows: "))
        cfg["grid_cols"] = int(input("  Cols: "))

    # Step 3: Capture a test image
    print("\nCapturing test image from ESP32-CAM...")
    img = capture_image(cfg)
    if img is None:
        print("Could not capture image. Check ESP32 IP and connection.")
        save_config(cfg)
        return

    # Save raw test image
    test_path = os.path.join(DATA_DIR, "calibration_raw.jpg")
    cv2.imwrite(test_path, img)
    print(f"Raw image saved: {test_path}")

    # Step 4: Pick perspective corners
    print("\n--- PERSPECTIVE CALIBRATION ---")
    print("A window will open showing the camera image.")
    print("Click the 4 corners of your seedling grid in this order:")
    print("  1. Top-left")
    print("  2. Top-right")
    print("  3. Bottom-right")
    print("  4. Bottom-left")
    print("Press 'r' to reset, 'q' to accept corners.\n")

    corners = []
    display_img = img.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append([x, y])
            cv2.circle(display_img, (x, y), 8, (0, 0, 255), -1)
            if len(corners) > 1:
                cv2.line(display_img, tuple(corners[-2]), tuple(corners[-1]), (0, 255, 0), 2)
            if len(corners) == 4:
                cv2.line(display_img, tuple(corners[3]), tuple(corners[0]), (0, 255, 0), 2)
                print("4 corners selected. Press 'q' to accept or 'r' to reset.")
            cv2.imshow("Calibration", display_img)

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", 1024, 768)
    cv2.setMouseCallback("Calibration", mouse_callback)
    cv2.imshow("Calibration", display_img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") and len(corners) == 4:
            break
        elif key == ord("r"):
            corners.clear()
            display_img = img.copy()
            cv2.imshow("Calibration", display_img)
            print("Reset. Click 4 corners again.")
        elif key == 27:  # ESC
            print("Cancelled.")
            cv2.destroyAllWindows()
            save_config(cfg)
            return

    cfg["perspective_corners"] = corners
    cv2.destroyAllWindows()

    # Show warped result
    warped = warp_perspective(img, cfg)
    warp_path = os.path.join(DATA_DIR, "calibration_warped.jpg")
    cv2.imwrite(warp_path, warped)
    print(f"Warped image saved: {warp_path}")

    # Step 5: HSV threshold tuning
    print("\n--- HSV THRESHOLD TUNING ---")
    print("Adjust the sliders until green seedlings are highlighted white")
    print("and background is black. Press 'q' when satisfied.\n")

    cv2.namedWindow("HSV Tuning", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Tuning", 1024, 768)
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)

    cv2.createTrackbar("H Low",  "Controls", cfg["hsv_lower"][0], 179, lambda x: None)
    cv2.createTrackbar("S Low",  "Controls", cfg["hsv_lower"][1], 255, lambda x: None)
    cv2.createTrackbar("V Low",  "Controls", cfg["hsv_lower"][2], 255, lambda x: None)
    cv2.createTrackbar("H High", "Controls", cfg["hsv_upper"][0], 179, lambda x: None)
    cv2.createTrackbar("S High", "Controls", cfg["hsv_upper"][1], 255, lambda x: None)
    cv2.createTrackbar("V High", "Controls", cfg["hsv_upper"][2], 255, lambda x: None)

    hsv_img = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    while True:
        hl = cv2.getTrackbarPos("H Low",  "Controls")
        sl = cv2.getTrackbarPos("S Low",  "Controls")
        vl = cv2.getTrackbarPos("V Low",  "Controls")
        hh = cv2.getTrackbarPos("H High", "Controls")
        sh = cv2.getTrackbarPos("S High", "Controls")
        vh = cv2.getTrackbarPos("V High", "Controls")

        mask = cv2.inRange(hsv_img, np.array([hl, sl, vl]), np.array([hh, sh, vh]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Show mask and overlay side by side
        overlay = warped.copy()
        overlay[mask > 0] = (0, 255, 0)
        blended = cv2.addWeighted(warped, 0.6, overlay, 0.4, 0)

        display = np.hstack([blended, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow("HSV Tuning", display)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            cfg["hsv_lower"] = [hl, sl, vl]
            cfg["hsv_upper"] = [hh, sh, vh]
            break

    cv2.destroyAllWindows()

    save_config(cfg)
    print("\nCalibration complete!")
    print(f"  Grid: {cfg['grid_rows']}x{cfg['grid_cols']}")
    print(f"  HSV range: {cfg['hsv_lower']} - {cfg['hsv_upper']}")
    print(f"\nRun 'python3 tracker.py --capture' to start tracking.")


def cmd_capture(cfg):
    """Capture an image, process it, and log measurements."""
    if cfg.get("perspective_corners") is None:
        print("ERROR: Run --calibrate first to set up perspective corners.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n[{timestamp}] Capturing...")

    # Capture
    img = capture_image(cfg)
    if img is None:
        sys.exit(1)

    # Save raw
    raw_path = os.path.join(IMAGES_DIR, f"{timestamp}_raw.jpg")
    cv2.imwrite(raw_path, img)

    # Warp
    warped = warp_perspective(img, cfg)

    # Segment
    mask = segment_green(warped, cfg)

    # Measure
    measurements = measure_cells(mask, cfg)

    # Log
    log_to_csv(measurements, timestamp)

    # Annotated image
    annotated = draw_annotated(warped, mask, measurements, cfg)
    ann_path = os.path.join(ANNOTATED_DIR, f"{timestamp}_annotated.jpg")
    cv2.imwrite(ann_path, annotated)

    # Summary
    avg_coverage = np.mean([m["coverage_pct"] for m in measurements])
    max_cell = max(measurements, key=lambda m: m["coverage_pct"])
    min_cell = min(measurements, key=lambda m: m["coverage_pct"])

    print(f"  Avg coverage: {avg_coverage:.1f}%")
    print(f"  Best: cell #{max_cell['cell_id']} ({max_cell['coverage_pct']:.1f}%)")
    print(f"  Weakest: cell #{min_cell['cell_id']} ({min_cell['coverage_pct']:.1f}%)")
    print(f"  Annotated: {ann_path}")


def cmd_report(cfg):
    """Generate a growth report with charts from logged data."""
    if not os.path.exists(CSV_FILE):
        print("No data yet. Run --capture first.")
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("Install matplotlib: pip3 install matplotlib")
        return

    # Read CSV
    timestamps = []
    cell_data = {}  # cell_id -> [(timestamp, coverage_pct)]

    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = datetime.strptime(row["timestamp"], "%Y-%m-%d_%H-%M-%S")
            cid = int(row["cell_id"])
            cov = float(row["coverage_pct"])
            if cid not in cell_data:
                cell_data[cid] = []
            cell_data[cid].append((ts, cov))

    if not cell_data:
        print("No data in CSV.")
        return

    num_cells = len(cell_data)
    rows, cols = cfg["grid_rows"], cfg["grid_cols"]

    # Plot 1: All cells growth curves
    fig, ax = plt.subplots(figsize=(14, 6))
    for cid in sorted(cell_data.keys()):
        data = sorted(cell_data[cid])
        times = [d[0] for d in data]
        covs = [d[1] for d in data]
        ax.plot(times, covs, marker=".", markersize=3, label=f"#{cid}", alpha=0.7)

    ax.set_xlabel("Time")
    ax.set_ylabel("Green Coverage (%)")
    ax.set_title("Seedling Growth - All Cells")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.xticks(rotation=45)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()

    report_path = os.path.join(DATA_DIR, "growth_curves.png")
    plt.savefig(report_path, dpi=150)
    print(f"Growth curves saved: {report_path}")

    # Plot 2: Grid heatmap of latest readings
    latest = {}
    for cid, data in cell_data.items():
        latest[cid] = sorted(data)[-1][1]  # most recent coverage

    grid = np.zeros((rows, cols))
    for cid, cov in latest.items():
        r = (cid - 1) // cols
        c = (cid - 1) % cols
        grid[r][c] = cov

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    im = ax2.imshow(grid, cmap="YlGn", vmin=0, vmax=max(100, grid.max()))
    for r in range(rows):
        for c in range(cols):
            cid = r * cols + c + 1
            ax2.text(c, r, f"#{cid}\n{grid[r][c]:.1f}%",
                     ha="center", va="center", fontsize=8)
    ax2.set_title("Latest Coverage Heatmap")
    ax2.set_xticks(range(cols))
    ax2.set_yticks(range(rows))
    plt.colorbar(im, label="Coverage %")
    plt.tight_layout()

    heatmap_path = os.path.join(DATA_DIR, "heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    print(f"Heatmap saved: {heatmap_path}")

    plt.show()


def cmd_preview(cfg):
    """Show live MJPEG stream from ESP32-CAM for aiming/focusing."""
    url = f"http://{cfg['esp32_ip']}:{cfg['esp32_port']}/stream"
    print(f"Opening stream: {url}")
    print("Press 'q' to quit, 's' to save a snapshot.\n")

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        # Fallback: poll /capture repeatedly
        print("MJPEG stream failed, falling back to polling /capture...")
        while True:
            img = capture_image(cfg)
            if img is not None:
                cv2.imshow("Preview", img)
            key = cv2.waitKey(500) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and img is not None:
                snap = os.path.join(DATA_DIR, f"snapshot_{int(time.time())}.jpg")
                cv2.imwrite(snap, img)
                print(f"Saved: {snap}")
        cv2.destroyAllWindows()
        return

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Preview", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and ret:
            snap = os.path.join(DATA_DIR, f"snapshot_{int(time.time())}.jpg")
            cv2.imwrite(snap, frame)
            print(f"Saved: {snap}")

    cap.release()
    cv2.destroyAllWindows()


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="Seedling Growth Tracker")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--calibrate", action="store_true", help="Interactive calibration")
    group.add_argument("--capture", action="store_true", help="Capture and measure")
    group.add_argument("--report", action="store_true", help="Generate growth report")
    group.add_argument("--preview", action="store_true", help="Live camera preview")
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_config()

    if args.calibrate:
        cmd_calibrate(cfg)
    elif args.capture:
        cmd_capture(cfg)
    elif args.report:
        cmd_report(cfg)
    elif args.preview:
        cmd_preview(cfg)


if __name__ == "__main__":
    main()
