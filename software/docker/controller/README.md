# Inference RL Controller

## About


## Installation and launch

- `xhost +local:docker`

**Build container** from **/software/docker/controller/** path:
```bash
docker build -t controller:jazzy .
```

then **run it**:
```bash
docker run -it --net host --ipc host --gpus all \
  -e DISPLAY=$DISPLAY \
  -e CYCLONEDDS_URI=file:///tmp/cyclone.xml \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp/cyclone.xml:/tmp/cyclone.xml \
  controller:jazzy
```

For the LIPM-based policy, launch `inference_controller_setup.py` with `--gait bd_lip`
and the default model path `./src/model/BipeD`.
