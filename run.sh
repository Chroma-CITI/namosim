#!/usr/bin/env bash

# Check args
if [ "$#" -ne 2 ]; then
  echo "usage: ./run.sh IMAGE_NAME CONTAINER_NAME"
  return 1
fi

XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

sudo docker run -it \
		--workdir="/home/$USER" \
		--volume="/home/$USER:/home/$USER" \
        --volume=$XSOCK:$XSOCK:rw \
        --volume=$XAUTH:$XAUTH:rw \
    	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    	--env="QT_X11_NO_MITSHM=1" \
        --env="XAUTHORITY=${XAUTH}" \
        --env="DISPLAY=$DISPLAY" \
        --user=$UID \
		--runtime=nvidia \
		--name $2\
		$1 \
    	$SHELL

