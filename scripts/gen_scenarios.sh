#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

out=tests/experiments/scenarios/intersections/generated

python -m namosim.main gen-alt-scenarios \
  --base-scenario "tests/experiments/scenarios/intersections/2_robots_50_goals_snamo.svg" \
  --n-robots 1 \
  --goals-per-robot 50 \
  --n-scenarios 10 \
  --out-dir $out
python -m namosim.main gen-alt-scenarios \
  --base-scenario "tests/experiments/scenarios/intersections/2_robots_50_goals_snamo.svg" \
  --n-robots 1 \
  --goals-per-robot 50 \
  --n-scenarios 10 \
  --use-social-cost \
  --out-dir $out

python -m namosim.main gen-alt-scenarios \
  --base-scenario "tests/experiments/scenarios/intersections/2_robots_50_goals_snamo.svg" \
  --n-robots 4 \
  --goals-per-robot 50 \
  --n-scenarios 10 \
  --out-dir $out
python -m namosim.main gen-alt-scenarios \
  --base-scenario "tests/experiments/scenarios/intersections/2_robots_50_goals_snamo.svg" \
  --n-robots 4 \
  --goals-per-robot 50 \
  --n-scenarios 10 \
  --use-social-cost \
  --out-dir $out

python -m namosim.main gen-alt-scenarios \
  --base-scenario "tests/experiments/scenarios/intersections/2_robots_50_goals_snamo.svg" \
  --n-robots 2 \
  --goals-per-robot 50 \
  --n-scenarios 10 \
  --out-dir $out
python -m namosim.main gen-alt-scenarios \
  --base-scenario "tests/experiments/scenarios/intersections/2_robots_50_goals_snamo.svg" \
  --n-robots 2 \
  --goals-per-robot 50 \
  --n-scenarios 10 \
  --use-social-cost \
  --out-dir $out

python -m namosim.main gen-alt-scenarios \
  --base-scenario "tests/experiments/scenarios/intersections/2_robots_50_goals_snamo.svg" \
  --n-robots 10 \
  --goals-per-robot 50 \
  --n-scenarios 10 \
  --out-dir $out
python -m namosim.main gen-alt-scenarios \
  --base-scenario "tests/experiments/scenarios/intersections/2_robots_50_goals_snamo.svg" \
  --n-robots 10 \
  --goals-per-robot 50 \
  --n-scenarios 10 \
  --use-social-cost \
  --out-dir $out