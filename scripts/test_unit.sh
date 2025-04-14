#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

set -eo pipefail

./namosim/scripts/test_unit.sh