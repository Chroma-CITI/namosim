#!/bin/bash

DIR=$(dirname "$0")
cd $DIR

set -eo pipefail

./format.sh
./test_types.sh
./test_unit.sh
./test_e2e.sh