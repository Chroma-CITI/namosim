#!/bin/bash

set -eo pipefail

DIR=$(dirname "$0")
cd $DIR/..

export NAMO_NO_DISPLAY_WINDOW="TRUE"
export NAMO_DEACTIVATE_RVIZ="TRUE"
python3 -m pytest -s tests/e2e