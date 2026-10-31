#!/usr/bin/env bash
# Extra seeds for seed-variance (§9.6) and the deep ensemble (hybrid seeds 0-2). Resumable: rerun the same lane.
cd "$(dirname "$0")/.."
T="python3 scripts/train.py --resume --encoder transformer --steps 4500 --train-split train_aug --val-split val_aug"
case "$1" in
  A) $T --name pinn_tf_hybrid_v2_s1 --seed 1 --lambda-p 1.0 --hybrid >> results/train_pinn_tf_hybrid_v2_s1.log 2>&1 || exit 1
     $T --name pinn_tf_v2_s1 --seed 1 --lambda-p 1.0 >> results/train_pinn_tf_v2_s1.log 2>&1 ;;
  B) $T --name pinn_tf_hybrid_v2_s2 --seed 2 --lambda-p 1.0 --hybrid >> results/train_pinn_tf_hybrid_v2_s2.log 2>&1 || exit 1
     $T --name data_tf_v2_s1 --seed 1 --lambda-p 0 >> results/train_data_tf_v2_s1.log 2>&1 ;;
esac
