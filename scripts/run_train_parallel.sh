#!/bin/bash

python scripts/run_train_parallel.py \
    --config train_configs/gelu-1l-asae.yaml \
    --commands experiments/gelu-1l-asae/all_train_ensemble_cmds.txt \
    --config-params train_configs/gelu-1l-param-list.yaml
