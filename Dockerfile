FROM osrf/ros:humble-desktop-full

# Install dependencies
RUN apt-get update -y && \
    apt-get install -y curl python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /workspace

# Copy necessary files
COPY . /workspace

# Install ROS dependencies
RUN rosdep install -ry --from-paths . || true

# Install Python dependencies
RUN pip install -r namosim/requirements.txt --ignore-installed && \
    pip install -r namoros/requirements.txt --ignore-installed

# Build plugins
RUN ./namoros/build_plugins.sh

# Build the project
RUN colcon build

# Source the setup script
RUN echo "source /workspace/install/setup.bash" >> ~/.bashrc

# Run bash shell
CMD ["/bin/bash"]