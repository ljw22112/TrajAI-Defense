"""Generate train / val / test splits into data/."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajai.dataset import make_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=20_000)
    ap.add_argument("--n-val", type=int, default=2_000)
    ap.add_argument("--only", nargs="*", default=None, help="generate only these splits")
    a = ap.parse_args()
    make_all(a.out, a.n_train, a.n_val, only=a.only)
