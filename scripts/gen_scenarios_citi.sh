#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

out=tests/experiments/scenarios/citi_lab/generated

python -m namosim.main gen-alt-scenarios \
  --base-scenario "tests/experiments/scenarios/citi_lab/citi_lab_2_robots_50_goals_snamo.svg" \
  --n-robots 2 \
  --goals-per-robot 50 \
  --n-scenarios 10 \
  --out-dir $out
python -m namosim.main gen-alt-scenarios \
  --base-scenario "tests/experiments/scenarios/citi_lab/citi_lab_2_robots_50_goals_snamo.svg" \
  --n-robots 2 \
  --goals-per-robot 50 \
  --n-scenarios 10 \
  --use-social-cost \
  --out-dir $out
