"""WCAG contrast of the console's colour tokens in every theme (light / dark / military).

Text tokens must reach 4.5:1 on every surface they are drawn on; on-* colours on their filled backgrounds as well.
Usage: python scripts/check_contrast.py   (exit 1 on any failure; run in CI)
"""
import re
import sys

CSS = open("web/src/styles.css", encoding="utf-8").read()
BLOCKS = {"light": r":root\{(.*?)\}", "dark": r':root\[data-theme="dark"\]\{(.*?)\}', "military": r':root\[data-theme="military"\]\{(.*?)\}'}
TEXT = ["ink", "ink-2", "ink-3", "obs", "pred", "danger", "ok", "ekf", "ukf"]
SURF = ["paper", "sheet", "hl"]
FILLED = [("on-pred", "pred"), ("on-ink", "ink"), ("on-danger", "danger"), ("side-ink", "side"), ("side-mute", "side"), ("side-acc", "side")] + [("on-lv", f"lv{i}") for i in range(1, 6)]


def tokens(name):
    t = dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", re.search(BLOCKS[name], CSS, re.S).group(1)))
    if name != "light":                     # unspecified tokens fall back to :root
        t = {**tokens("light"), **t}
    return t


def lum(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


fails = []
for th in BLOCKS:
    t = tokens(th)
    pairs = [(f, s) for f in TEXT for s in SURF] + FILLED
    worst = min(((ratio(t[f], t[s]), f, s) for f, s in pairs))
    for f, s in pairs:
        r = ratio(t[f], t[s])
        if r < 4.5:
            fails.append(f"{th}: --{f} {t[f]} on --{s} {t[s]} = {r:.2f}")
    print(f"{th}: worst {worst[0]:.2f} (--{worst[1]} on --{worst[2]})")
print("\n".join(fails) if fails else "all contrast checks passed (>= 4.5:1)")
sys.exit(1 if fails else 0)
