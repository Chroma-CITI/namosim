#!/bin/bash

DIR=$(dirname "$0")
cd $DIR/..

cd docs
make clean
make html
