# Inference RL Controller

## About


## Installation and launch

- `xhost +local:docker`

**Build container** from **/software/docker/controller/** path:
```bash
docker build -t gait_controller:jazzy .
```

then **run it**:
```bash
docker run -it --net host --ipc host --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/logs:/workspace/logs \
  gait_controller:jazzy

```

## Run logs

Each run writes `commands.csv`, `motor_positions.csv`, `imu.csv` and a few
debug plots (`.png`) to `logs/<run_timestamp>/`. The `-v $(pwd)/logs:/workspace/logs`
mount above is what makes these land on the host machine instead of being
lost inside the container.
