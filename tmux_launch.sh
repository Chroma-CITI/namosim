#!/bin/sh
tmux new-session -d 'roscore'
tmux split-window -h 'ptpython'
tmux select-pane -t !
tmux split-window -v 'rviz -d rviz/two_robots_config.rviz'
tmux select-pane -t !
tmux split-window -v 'pycharm .'
tmux select-pane -t bottom
tmux split-window -v
tmux -2 attach-session -d
