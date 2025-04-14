#!/bin/bash

DIR="$(dirname "$(readlink -f "$0")")"

cd ${DIR}/..
colcon build
source ./install/setup.bash 
ros2 run namoros scenario2sdf --svg-file=${DIR}/config/space_conflict.svg --out-dir=${DIR}/config
colcon build
ros2 launch namoros launch.multi.py \
    scenario_file:=${DIR}/config/space_conflict.svg \
    config_file:=${DIR}/config/namoros_config.yaml \
    sdf_file:=${DIR}/config/namo_world.sdf \
    map_yaml:=${DIR}/config/space_conflict.yaml