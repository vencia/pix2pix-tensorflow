#!/bin/bash

PROJECT_ROOT=$(dirname $(realpath "$0"))

cd "$PROJECT_ROOT"
source ".venv/bin/activate"

for lr_dis in 0.00005 0.00001 0.000005 0.000001
do
    for decay_rate in 0.97 0.98 0.99 1.0
    do
        for gan_weight in 0.2 0.1 0.05
        do
            PYTHONPATH=. python3 ./pix2pix.py --mode train --output_dir summaries/train --max_epochs 5 --input_dir inputs/train_polygon --summary_freq 100 --scale_size 256 --lr_gen 0.0002 --lr_dis $lr_dis --decay_rate $decay_rate --no_flip --l1_weight 1.0 --gan_weight $gan_weight
        done
    done
done

