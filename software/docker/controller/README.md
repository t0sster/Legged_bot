# Inference RL Controller

## About


## Installation and launch

- `xhost +local:docker`

**Build container** from **/software/docker/controller/** path:
```bash
docker build -t gait-controller:jazzy .
```

then **run it**:
```bash
docker run -it --net host --ipc host --gpus all \
  -e DISPLAY=$DISPLAY \
  -e CYCLONEDDS_URI=file:///tmp/cyclone.xml \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp/cyclone.xml:/tmp/cyclone.xml \
  gait_controller:jazzy

```

### Fast run with local changes

Use the helper script to reuse the existing image and mount your local code into the container:

```bash
./run.sh
```

To pass controller args (e.g. device or model path):

```bash
./run.sh --gait lip_play --device keyboard --path ./src/model/tinker
```

