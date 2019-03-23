#!/bin/bash

PROJECT_ROOT=$(dirname $(realpath "$0"))

cd "$PROJECT_ROOT"
source ".venv/bin/activate"

PYTHONPATH=. python3 ./pix2pix.py --mode test --output_dir predictions/polygon_64 --input_dir inputs/val/polygon_v3_s64_input_nodes_and_edges_output_nodes_and_edges --scale_size 256 --no_flip --checkpoint trainings/polygon_64_20_0.0001_1e-06_1.0_0.01_0.98
