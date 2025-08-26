#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

python3 -m namosim.main run tests/scenarios/citi_full/citi_full_namo.yaml
