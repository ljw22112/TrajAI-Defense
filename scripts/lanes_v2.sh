#!/usr/bin/env bash
# Resumable training lanes for v2 (noise augmentation) and the weather experiment.
cd "$(dirname "$0")/.."
T="python3 scripts/train.py --resume --encoder transformer"
case "$1" in
  A)
    $T --name pinn_tf_hybrid_v2 --lambda-p 1.0 --hybrid --steps 4500 --train-split train_aug --val-split val_aug >> results/train_pinn_tf_hybrid_v2.log 2>&1 || exit 1
    $T --name wx_hybrid_date   --lambda-p 1.0 --hybrid --wx --global-key g_date --phys-atm date --steps 4500 --train-split train_wx --val-split val_wx >> results/train_wx_hybrid_date.log 2>&1 || exit 1
    $T --name pinn_tf_v2       --lambda-p 1.0 --steps 4500 --train-split train_aug --val-split val_aug >> results/train_pinn_tf_v2.log 2>&1 ;;
  B)
    $T --name data_tf_v2       --lambda-p 0 --steps 4500 --train-split train_aug --val-split val_aug >> results/train_data_tf_v2.log 2>&1 || exit 1
    $T --name wx_hybrid_clim   --lambda-p 1.0 --hybrid --phys-atm clim --steps 4500 --train-split train_wx --val-split val_wx >> results/train_wx_hybrid_clim.log 2>&1 || exit 1
    $T --name data_tf_2k_v2    --lambda-p 0 --n-train 2000 --steps 3000 --train-split train_aug --val-split val_aug >> results/train_data_tf_2k_v2.log 2>&1 || exit 1
    $T --name pinn_tf_2k_v2    --lambda-p 1.0 --n-train 2000 --steps 3000 --train-split train_aug --val-split val_aug >> results/train_pinn_tf_2k_v2.log 2>&1 ;;
esac
