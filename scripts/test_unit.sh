#!/bin/bash

set -eo pipefail

DIR=$(dirname "$0")
cd $DIR/..

export NAMO_NO_DISPLAY_WINDOW="TRUE"
python3 -m pytest -s --cov=namosim --cov-fail-under=27 tests/unit