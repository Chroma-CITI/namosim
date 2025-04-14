#!/bin/bash

DIR="$(dirname "$(readlink -f "$0")")"
cd $DIR

mkdir -p gz_plugin/build
cd gz_plugin/build && cmake .. && make && cd ../..
