"""Capture README screenshots from the live deployment.

Usage:  python scripts/capture_screenshots.py [BASE_URL]
Writes docs/screenshots/*.png (1440x900 viewport).
Requires: pip install playwright && playwright install chromium
"""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://quant-ai-ui.vercel.app"
OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

# Dismiss first-visit overlays so page shots show the page itself.
DISMISS_OVERLAYS = """
localStorage.setItem('quant-ai:tour-done', '1');
localStorage.setItem('quant-ai:demo-banner-dismissed', '1');
"""

# (filename, route, settle seconds)
PAGES = [
    ("dashboard", "/dashboard?ticker=AAPL", 10),
    ("portfolio", "/portfolio", 10),
    ("screener", "/screener", 8),
    ("leaderboard", "/leaderboard", 8),
]


def shoot(page, name, route, settle):
    page.goto(f"{BASE}{route}", wait_until="load", timeout=60_000)
    time.sleep(settle)  # let react-query fill in real data
    page.screenshot(path=str(OUT / f"{name}.png"))
    print(f"saved {name}.png")


with sync_playwright() as p:
    browser = p.chromium.launch()

    # Clean page shots (overlays pre-dismissed)
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    ctx.add_init_script(DISMISS_OVERLAYS)
    page = ctx.new_page()
    for name, route, settle in PAGES:
        shoot(page, name, route, settle)
    ctx.close()

    # One shot WITH the first-visit tour visible (it's a selling point)
    ctx2 = browser.new_context(viewport={"width": 1440, "height": 900})
    page2 = ctx2.new_page()
    shoot(page2, "tour", "/screener", 8)
    ctx2.close()

    browser.close()

print(f"done → {OUT}")
