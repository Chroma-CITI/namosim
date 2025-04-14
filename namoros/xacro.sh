#!/bin/bash

DIR="$(dirname "$(readlink -f "$0")")"

cd ${DIR}/models/turtlebot_description/robots
xacro kobuki_hexagons_astra.urdf.xacro > kobuki_hexagons_astra.urdf
