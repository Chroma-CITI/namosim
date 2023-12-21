#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

# Deactivate GUIs to go faster
export NAMO_NO_DISPLAY_WINDOW=TRUE
export NAMO_DEACTIVATE_RVIZ=TRUE

# 2_robots_50_goals_namo
index=0
for filename in ./tests/experiments/scenarios/intersections/generated/2_robots_50_goals_namo/*.svg; do
  echo "Running simulation for scenario $filename"
  python -m namosim.main run $filename --logs-dir "namo_logs/intersections/2_robots_50_goals_namo/${index}" &
  ((index++))
done

# 2_robots_50_goals_snamo
index=0
for filename in ./tests/experiments/scenarios/intersections/generated/2_robots_50_goals_snamo/*.svg; do
  echo "Running simulation for scenario $filename"
  python -m namosim.main run $filename --logs-dir "namo_logs/intersections/2_robots_50_goals_snamo/${index}" &
  ((index++))
done

# 4_robots_50_goals_namo
index=0
for filename in ./tests/experiments/scenarios/intersections/generated/2_robots_50_goals_namo/*.svg; do
  echo "Running simulation for scenario $filename"
  python -m namosim.main run $filename --logs-dir "namo_logs/intersections/2_robots_50_goals_namo/${index}" &
  ((index++))
done

wait