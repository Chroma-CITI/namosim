#!/bin/bash

set -eo pipefail

DIR=$(dirname "$0")
cd $DIR

#./format.sh
#./test_types.sh
./test_unit.sh
./test_e2e.sh