"""Read a TrajAI-Bench split with Stone Soup's generic CSV readers (pip install stonesoup).
Each track is placed on its own time window (TIME0 + track_id * 1000 s), so the readers can stream all tracks
of a split in one pass; 'track_id' is kept as detection metadata.
Usage: python stonesoup_example.py [split]
"""
import sys
from pathlib import Path

from stonesoup.reader.generic import CSVDetectionReader, CSVGroundTruthReader

split = sys.argv[1] if len(sys.argv) > 1 else "test_id_f10"
root = Path(__file__).resolve().parent / "stonesoup"

detections = CSVDetectionReader(root / f"{split}_detections.csv", state_vector_fields=("x", "y", "z"),
                                time_field="time", timestamp=True, metadata_fields=("track_id", "range_m", "az_rad", "el_rad"))
truth = CSVGroundTruthReader(root / f"{split}_groundtruth.csv", state_vector_fields=("x", "y", "z"),
                             time_field="time", path_id_field="track_id", timestamp=True)

n_scans = n_det = 0
for _, dets in detections:
    n_scans += 1
    n_det += len(dets)
paths = set()
for _, gts in truth:
    paths |= {p.id for p in gts}
print(f"{split}: {n_det} detections in {n_scans} scans, {len(paths)} ground-truth paths")
