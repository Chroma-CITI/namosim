#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

out=tests/experiments/scenarios/citi_lab/generated
base_scenario="tests/experiments/scenarios/citi_lab/citi_lab_2_robots_50_goals_snamo.svg"
n_scenarios=10
n_goals=50

for i in $(seq 1 10); do

  echo "Generating Citi Lab scenarios with ${i} robots."

  # namo_ncr
  python -m namosim.main gen-alt-scenarios \
    --base-scenario ${base_scenario} \
    --n-robots $i \
    --goals-per-robot ${n_goals} \
    --n-scenarios ${n_scenarios} \
    --no-resolve-conflicts \
    --no-resolve-deadlocks \
    --out-dir $out

  # namo_ndr
  python -m namosim.main gen-alt-scenarios \
    --base-scenario ${base_scenario} \
    --n-robots $i \
    --goals-per-robot ${n_goals} \
    --n-scenarios ${n_scenarios} \
    --no-resolve-deadlocks \
    --out-dir $out

  # namo
  python -m namosim.main gen-alt-scenarios \
    --base-scenario ${base_scenario} \
    --n-robots $i \
    --goals-per-robot ${n_goals} \
    --n-scenarios ${n_scenarios} \
    --out-dir $out

  # snamo_ncr
  python -m namosim.main gen-alt-scenarios \
    --base-scenario ${base_scenario} \
    --n-robots $i \
    --goals-per-robot ${n_goals} \
    --n-scenarios ${n_scenarios} \
    --use-social-cost \
    --no-resolve-conflicts \
    --no-resolve-deadlocks \
    --out-dir $out

  # snamo_ndr
  python -m namosim.main gen-alt-scenarios \
    --base-scenario ${base_scenario} \
    --n-robots $i \
    --goals-per-robot ${n_goals} \
    --n-scenarios ${n_scenarios} \
    --use-social-cost \
    --no-resolve-deadlocks \
    --out-dir $out

  # snamo
  python -m namosim.main gen-alt-scenarios \
    --base-scenario ${base_scenario} \
    --n-robots $i \
    --goals-per-robot ${n_goals} \
    --n-scenarios ${n_scenarios} \
    --use-social-cost \
    --out-dir $out

done