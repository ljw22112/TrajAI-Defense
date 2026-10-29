#!/usr/bin/env bash
# Two parallel lanes of training runs (1 CPU thread each), then evaluation.
set -e
cd "$(dirname "$0")/.."
STEPS=${STEPS:-1500}
lane_a() {
  python3 scripts/train.py --name data_tf     --encoder transformer --lambda-p 0 --steps $STEPS
  python3 scripts/train.py --name pinn_tf     --encoder transformer --lambda-p 1.0 --steps $STEPS
  python3 scripts/train.py --name data_tf_2k  --encoder transformer --lambda-p 0 --n-train 2000 --steps $STEPS
}
lane_b() {
  while pgrep -f scripts/prepare.py >/dev/null; do sleep 20; done
  python3 scripts/train.py --name pinn_tf_hybrid --encoder transformer --lambda-p 1.0 --hybrid --steps $STEPS
  python3 scripts/train.py --name pinn_lstm      --encoder lstm        --lambda-p 1.0 --steps $STEPS
  python3 scripts/train.py --name pinn_tf_2k     --encoder transformer --lambda-p 1.0 --n-train 2000 --steps $STEPS
}
lane_a > results/lane_a.log 2>&1 &
A=$!
lane_b > results/lane_b.log 2>&1 &
B=$!
wait $A $B
python3 scripts/evaluate.py data_tf pinn_tf pinn_lstm pinn_tf_hybrid data_tf_2k pinn_tf_2k > results/evaluate.log 2>&1
echo ALL_DONE
