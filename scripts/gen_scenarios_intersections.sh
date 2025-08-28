#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

# Deactivate GUIs to go faster
export NAMO_DEACTIVATE_TKINTER=TRUE
export NAMO_DEACTIVATE_RVIZ=TRUE

scenario=intersections
out=tests/experiments/scenarios/${scenario}/generated
base_scenario="tests/experiments/scenarios/${scenario}/${scenario}_base.svg"
n_scenarios=20
n_goals=50

for i in $(seq 1 10); do

  echo "Generating ${scenario} scenarios with ${i} robots."

  # namo_ncr
  python -m namosim.main gen-alt-scenarios \
    --base-scenario ${base_scenario} \
    --n-robots $i \
    --goals-per-robot ${n_goals} \
    --n-scenarios ${n_scenarios} \
    --no-resolve-conflicts \
    --no-resolve-deadlocks \
    --out-dir "$out/${i}_robots_${n_goals}_goals_namo_ncr"

  # namo_ndr
  python -m namosim.main gen-alt-scenarios \
    --base-scenario ${base_scenario} \
    --n-robots $i \
    --goals-per-robot ${n_goals} \
    --n-scenarios ${n_scenarios} \
    --no-resolve-deadlocks \
    --out-dir "$out/${i}_robots_${n_goals}_goals_namo_ndr"

  # namo
  python -m namosim.main gen-alt-scenarios \
    --base-scenario ${base_scenario} \
    --n-robots $i \
    --goals-per-robot ${n_goals} \
    --n-scenarios ${n_scenarios} \
    --out-dir "$out/${i}_robots_${n_goals}_goals_namo"

  # snamo_ncr
  python -m namosim.main gen-alt-scenarios \
    --base-scenario ${base_scenario} \
    --n-robots $i \
    --goals-per-robot ${n_goals} \
    --n-scenarios ${n_scenarios} \
    --use-social-cost \
    --no-resolve-conflicts \
    --no-resolve-deadlocks \
    --out-dir "$out/${i}_robots_${n_goals}_goals_snamo_ncr"

  # snamo_ndr
  python -m namosim.main gen-alt-scenarios \
    --base-scenario ${base_scenario} \
    --n-robots $i \
    --goals-per-robot ${n_goals} \
    --n-scenarios ${n_scenarios} \
    --use-social-cost \
    --no-resolve-deadlocks \
    --out-dir "$out/${i}_robots_${n_goals}_goals_snamo_ndr"

  # snamo
  python -m namosim.main gen-alt-scenarios \
    --base-scenario ${base_scenario} \
    --n-robots $i \
    --goals-per-robot ${n_goals} \
    --n-scenarios ${n_scenarios} \
    --use-social-cost \
    --out-dir "$out/${i}_robots_${n_goals}_goals_snamo"

  # snamo_distance_dr
  python -m namosim.main gen-alt-scenarios \
    --base-scenario ${base_scenario} \
    --n-robots $i \
    --goals-per-robot ${n_goals} \
    --n-scenarios ${n_scenarios} \
    --use-social-cost \
    --deadlock-strategy DISTANCE \
    --out-dir "$out/${i}_robots_${n_goals}_goals_snamo_distance_dr"

done