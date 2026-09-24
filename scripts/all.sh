#!/bin/bash
exp=$1
selector_config=${2:-cfg/selector/selector_config.yaml}

scripts/replica_mono.sh $exp "$selector_config"
scripts/replica_rgbd.sh $exp "$selector_config"

scripts/tum_mono.sh $exp "$selector_config"
scripts/tum_rgbd.sh $exp "$selector_config"
