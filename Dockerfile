FROM xia0ben/ros:melodic-desktop-full-nvidia-x

# Install system dependencies:
RUN apt-get update && \
	apt-get install -y \
		ros-melodic-grid-map \
		ros-melodic-jsk-visualization \
		python-pip && \
	rm -rf /var/lib/apt/lists/*

# Install dev depencies
## Install JRE+JDK to be able to run Pycharm within the container
# RUN apt-get update && \
# 	apt-get install -y \
# 		default-jre \
# 		default-jdk \
# 		libcanberra-gtk-module \
# 		nano \
# 		tmux && \
# 	rm -rf /var/lib/apt/lists/*

# Copy all files
COPY . s-namo-sim/

# Allow modification by any users (it's ok since we only run this locally)
RUN chmod -R u+rw,g+rw,o+rw s-namo-sim 

# Add folder to python path

ENV PYTHONPATH="${PYTHONPATH}:/s-namo-sim"

# Install Python dependencies
RUN pip install -r /s-namo-sim/requirements.txt
