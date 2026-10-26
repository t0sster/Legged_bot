# Симуляция в MuJoCo

Файл `sim_mujoco.py` запускает симуляцию в MuJoCo.


```bash
docker build -t sim2sim:mujoco .
```

```bash
docker run -it --network host --ipc host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/dri --group-add video \
  sim2sim:mujoco

```
## Смещение центра масс базы

Параметр `base_com_offset` — смещение ЦМ тела `torso` в метрах, в системе `base_link`
(x вперёд, y влево, z вверх). Масса и инерция относительно ЦМ не меняются, как в
`base_pseudo_inertia` обучения (там ±0.10 м по каждой оси). Значения пишутся как
float (`0.0`, не `0`):

```bash
docker run -it --network host --ipc host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/dri --group-add video \
  sim2sim:mujoco \
  bash -c "source /opt/ros/jazzy/setup.bash && source /workspace/tinker_msgs/install/setup.bash && \
    python3 /workspace/sim_mujoco.py --ros-args -p base_com_offset:='[0.05, 0.0, 0.0]'"
```
