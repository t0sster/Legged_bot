ARG ROS_DISTRO=jazzy
FROM osrf/ros:${ROS_DISTRO}-desktop

ARG ROS_DISTRO=jazzy
ENV ROS_DISTRO=${ROS_DISTRO}
ENV ROS_DOMAIN_ID=5

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    python3-catkin-pkg-modules \
    python3-colcon-common-extensions \
    python3-empy \
    python3-lark \
    iputils-ping \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    ros-${ROS_DISTRO}-ament-cmake \
    ros-${ROS_DISTRO}-ros-workspace \
    ros-${ROS_DISTRO}-pinocchio \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    empy==3.3.4 \
    torch==2.3.0 \
    torchvision==0.18.0 \
    numpy \
    pynput \
    onnxruntime \
    scipy \
    pyyaml

WORKDIR /workspace

COPY . /workspace/

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    cd /workspace/tinker_msgs && \
    colcon build --symlink-install

ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

CMD ["/bin/bash", "-c", "source /opt/ros/$ROS_DISTRO/setup.bash && source /workspace/tinker_msgs/install/setup.bash && exec python3 -m src.gait_controller"]