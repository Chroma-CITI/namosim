#!/bin/bash

set -eo pipefail

DIR=$(dirname "$0")
cd $DIR/..

export NAMO_DEACTIVATE_TKINTER="TRUE"
export NAMO_DEACTIVATE_RVIZ="TRUE"

./examples/compute_plan.sh