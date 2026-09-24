#!/bin/bash
selector_config=${1:-cfg/selector/selector_config.yaml}

bin/train_colmap \
    cfg/colmap/gaussian_splatting.yaml \
    ./data/tandt_db/db/drjohnson \
    results/colmap/drjohnson \
    no_viewer \
    --selector-config "$selector_config"
