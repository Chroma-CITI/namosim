#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

# Deactivate GUIs to go faster
export NAMO_NO_DISPLAY_WINDOW=TRUE
export NAMO_DEACTIVATE_RVIZ=TRUE

for n_robots in $(seq 1 10); do
  echo "Starting simulations for ${n_robots} robots."

  variants=("namo" "namo_ncr" "namo_ndr" "snamo" "snamo_ncr" "snamo_ndr")
  for alg in "${variants[@]}"; do

    index=0
    for filename in ./tests/experiments/scenarios/willow_garage/generated/${n_robots}_robots_50_goals_${alg}/*.svg; do
      echo "Running simulation for scenario $filename"
      python -m namosim.main run $filename --logs-dir "namo_logs/willow_garage/${n_robots}_robots_50_goals_${alg}/${index}" &
      ((index++))
    done

    wait
  done
done
