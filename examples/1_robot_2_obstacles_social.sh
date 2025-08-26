#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

python3 -m namosim.main run tests/scenarios/1_robot_2_obstacles_social.svg
