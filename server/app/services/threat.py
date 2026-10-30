"""Threat assessment against the assets table (wraps trajai.threat)."""
from __future__ import annotations

import numpy as np

from trajai import threat


def assess(ip, sigma, trajectory: dict, assets) -> dict:
    lst = [dict(name=a.name, x=a.x_km, y=a.y_km, radius=a.radius_km) for a in assets]
    probs = threat.impact_probabilities(np.asarray(ip, float), np.maximum(np.asarray(sigma, float), 1.0), assets=lst) if lst else {}
    level = threat.threat_level(probs)
    top = max(probs, key=probs.get) if probs else None
    windows = threat.engagement_windows(np.asarray(trajectory["t"]), np.asarray(trajectory["z"]))
    top_id = next((a.id for a in assets if a.name == top), None)
    return dict(level=level, max_prob=float(probs[top]) if top else 0.0, top_asset_id=top_id, probabilities=probs, windows=windows)
