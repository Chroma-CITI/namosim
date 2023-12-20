#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

python -m namosim.main gen-alt-scenarios --base-scenario "tests/unit/scenarios/minimal_stilman_2005.svg"