#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

python -m namosim.main compare-results \
  --results-a "namo_logs/2_robots_50_goals_snamo/stats.json" \
  --title-a "snamo" \
  --results-b "namo_logs/2_robots_50_goals_namo/stats.json" \
  --title-b "namo"
