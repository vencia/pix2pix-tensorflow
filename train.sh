#!/bin/bash

PROJECT_ROOT=$(dirname $(realpath "$0"))

cd "$PROJECT_ROOT"
source ".venv/bin/activate"

PYTHONPATH=. python3 ./pix2pix.py --mode train --output_dir summaries/train --max_epochs 20 --input_dir inputs/train_pattern --summary_freq 100 --scale_size 256 --lr_gen 0.0002 --lr_dis 0.00001 --decay_rate 0.98 --no_flip --l1_weight 1.0 --gan_weight 0.1
