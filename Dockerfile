FROM osrf/ros:humble-desktop-full

# Install dependencies
RUN apt-get update -y && \
    apt-get install -y curl python3-pip ros-humble-grid-map && \
    rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /workspace

# Copy necessary files
COPY . .

# Install ROS dependencies

RUN rosdep install -ry --from-paths . || true

# Install Python dependencies
RUN pip install -r requirements.txt

# Build the project
# RUN colcon build

# Source the setup script
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Run bash shell
CMD ["/bin/bash"]