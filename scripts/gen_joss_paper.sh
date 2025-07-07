#!/bin/bash

set -eo pipefail

DIR=$(dirname "$0")
cd $DIR/..

docker run --rm \
    --volume $PWD:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/inara