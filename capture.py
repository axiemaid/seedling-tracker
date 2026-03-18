#!/usr/bin/env python3
"""Fetch a JPEG from the ESP32-CAM and save it with a timestamp."""

import os
import sys
import urllib.request
from datetime import datetime

ESP32_URL = "http://192.168.151.219/capture"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
TIMEOUT = 10
RETRIES = 3

def capture():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(ESP32_URL, timeout=TIMEOUT) as resp:
                data = resp.read()
                if len(data) < 1000:
                    print(f"Attempt {attempt}: image too small ({len(data)} bytes)")
                    continue
                
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                path = os.path.join(SAVE_DIR, f"{ts}.jpg")
                with open(path, "wb") as f:
                    f.write(data)
                
                print(f"Saved {path} ({len(data)} bytes)")
                return 0
        except Exception as e:
            print(f"Attempt {attempt}: {e}")
    
    print("Failed after all retries")
    return 1

if __name__ == "__main__":
    sys.exit(capture())
