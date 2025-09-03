#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

set -eo pipefail
python3 -m examples.compute_plan