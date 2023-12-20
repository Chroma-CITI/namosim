#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

python -m namosim.main gen-alt-scenarios \
  --base-scenario "tests/experiments/scenarios/intersections/1_robot_50_goals_snamo.svg"
