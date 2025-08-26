#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

python3 -m namosim.main run tests/scenarios/minimal_stilman_rrt_star.svg
