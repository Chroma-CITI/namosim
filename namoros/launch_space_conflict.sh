#!/bin/bash

DIR="$(dirname "$(readlink -f "$0")")"

cd ${DIR}
colcon build
source ./install/setup.bash 
ros2 run namoros scenario2sdf --svg-file=namoros/config/space_conflict.svg --out-dir=namoros/config
colcon build
ros2 launch namoros launch.multi.py \
    scenario_file:=namoros/config/space_conflict.svg \
    config_file:=namoros/config/namoros_config.yaml \
    sdf_file:=namoros/config/namo_world.sdf \
    map_yaml:=${DIR}/namoros/config/space_conflict.yaml