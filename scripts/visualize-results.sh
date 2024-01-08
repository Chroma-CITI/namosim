#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

python -m namosim.main compare-results \
  --result-dirs "namo_logs/intersections/1_robots_50_goals_snamo,namo_logs/intersections/1_robots_50_goals_snamo,namo_logs/intersections/2_robots_50_goals_snamo,namo_logs/intersections/2_robots_50_goals_namo,namo_logs/intersections/4_robots_50_goals_snamo,namo_logs/intersections/4_robots_50_goals_namo" \
  --titles "1-Robot Social,1-Robot,2-Robot Social,2-Robot,4-Robot Social,4-Robot"

