#!/usr/bin/env bash

# Check args
if [ "$#" -ne 1 ]; then
  echo "usage: ./build.sh IMAGE_NAME"
  exit
fi

sudo docker build\
  --network=host\
  -t $1 ./
