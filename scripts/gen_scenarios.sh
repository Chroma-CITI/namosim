#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

out=tests/experiments/scenarios/intersections/generated

for i in $(seq 1 20); do
  echo "Generating Intersections scenarios with ${i} robots."
  # namo_ncr
  python -m namosim.main gen-alt-scenarios \
    --base-scenario "tests/experiments/scenarios/intersections/2_robots_50_goals_snamo.svg" \
    --n-robots $i \
    --goals-per-robot 50 \
    --n-scenarios 20 \
    --no-resolve-conflicts \
    --out-dir $out

  # namo
  python -m namosim.main gen-alt-scenarios \
    --base-scenario "tests/experiments/scenarios/intersections/2_robots_50_goals_snamo.svg" \
    --n-robots $i \
    --goals-per-robot 50 \
    --n-scenarios 20 \
    --out-dir $out

  # snamo_ncr
  python -m namosim.main gen-alt-scenarios \
    --base-scenario "tests/experiments/scenarios/intersections/2_robots_50_goals_snamo.svg" \
    --n-robots $i \
    --goals-per-robot 50 \
    --n-scenarios 20 \
    --use-social-cost \
    --no-resolve-conflicts \
    --no-resolve-deadlocks \
    --out-dir $out

  # snamo
  python -m namosim.main gen-alt-scenarios \
    --base-scenario "tests/experiments/scenarios/intersections/2_robots_50_goals_snamo.svg" \
    --n-robots $i \
    --goals-per-robot 50 \
    --n-scenarios 20 \
    --use-social-cost \
    --out-dir $out
  
  # snamo_ndr
  python -m namosim.main gen-alt-scenarios \
    --base-scenario "tests/experiments/scenarios/intersections/2_robots_50_goals_snamo.svg" \
    --n-robots $i \
    --goals-per-robot 50 \
    --n-scenarios 20 \
    --use-social-cost \
    --no-resolve-deadlocks \
    --out-dir $out
done