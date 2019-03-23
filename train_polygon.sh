#!/bin/bash

PROJECT_ROOT=$(dirname $(realpath "$0"))

cd "$PROJECT_ROOT"
source ".venv/bin/activate"

PYTHONPATH=. python3 ./pix2pix.py --mode train --max_epochs 40 --input_dir inputs/train/polygon_10000_colsur_v6_s256 --summary_freq 100 --scale_size 256 --lr_gen 0.0001 --lr_dis 0.0000001 --decay_rate 0.98 --no_flip --l1_weight 1.0 --gan_weight 0.05 --checkpoint trainings/polygon_10000_colsur_v6_s256_20_0.0001_5e-07_1.0_0.05_0.98
