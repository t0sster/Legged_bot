# Симуляция в MuJoCo

Файл `sim_mujoco.py` запускает симуляцию в MuJoCo.

```bash
docker run -it --network host --ipc host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/dri \
  --group-add video \
  sim2sim:mujoco

```