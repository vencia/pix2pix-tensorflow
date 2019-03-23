#!/bin/bash

PROJECT_ROOT=$(dirname $(realpath "$0"))

cd "$PROJECT_ROOT"
source ".venv/bin/activate"

for lr_gen in 0.0005 0.0001 0.00005
do
for lr_dis in 0.000001 0.0000005 0.0000001
do
    for decay_rate in 0.98 0.99
    do
        for gan_weight in 0.05 0.01
        do
            PYTHONPATH=. python3 ./pix2pix.py --mode train --output_dir summaries/polygon --max_epochs 30 --input_dir inputs/train_polygon --summary_freq 1000 --scale_size 256 --lr_gen $lr_gen --lr_dis $lr_dis --decay_rate $decay_rate --no_flip --l1_weight 1.0 --gan_weight $gan_weight
        done
    done
done
done
