# Inference RL Controller

## About


## Installation and launch

- `xhost +local:docker`

**Build container** from **/software/docker/controller/** path:
```bash
docker build -t gait_controller:jazzy .
```

then **run it**, choosing the mode with `--sim` (MuJoCo sim2sim) or `--real` (robot).
Arguments after the image name go to the controller, so switching modes or changing the
velocity does not need a rebuild:
```bash
docker run -it --net host --ipc host --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/logs:/workspace/logs \
  gait_controller:jazzy --sim
```

| Flag | State topic | `use_sim_time` |
|---|---|---|
| `--sim` | `/low_level_state` (`sim_mujoco.py`) | `True` (`/clock` from the sim) |
| `--real` | `/low_level_state_real` (`hardware_node`) | `False` |

One of them is required.

## Velocity command

`--vel VX VY WZ` — forward and lateral velocity in m/s, yaw rate in rad/s (body frame:
x forward, y left, z up). Default `0.1 0 0`. Training ranges: vx ±0.15, vy ±0.1, wz ±0.25.
Below 0.02 m/s and 0.05 rad/s the command is treated as zero and the robot stands still.
```bash
docker run -it --net host --ipc host --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/logs:/workspace/logs \
  gait_controller:jazzy --real --vel 0.1 0.0 0.2
```

## Policy without sending commands (`--no-cmd`)

Works with both `--sim` and `--real`. The policy runs on the incoming state and everything
is logged as usual, but `LowCmd` is never published, so the robot (or the simulator) only
does what other nodes tell it (e.g. the pose held by `init_pose_node`). `commands.csv` and
the `commanded` curve in `joint_tracking.png` show what the policy would have sent.
The log directory gets a `_nocmd` mark (`sim_nocmd_`, `real_nocmd_`, `real_and_sim_nocmd_`,
see [Run logs](#run-logs)).
```bash
docker run -it --net host --ipc host --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/logs:/workspace/logs \
  gait_controller:jazzy --real --no-cmd
```

## kp / kd

Gains sent in `LowCmd` are ROS parameters `kp` and `kd` (default 15.0 / 0.65; the
policy was trained with 40.0 / 1.0). To set them at launch, add ROS arguments after the
controller flags. Values must be floats (`15.0`, not `15`):
```bash
docker run -it --net host --ipc host --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/logs:/workspace/logs \
  gait_controller:jazzy --real --vel 0.1 0.0 0.0 --ros-args -p kp:=40.0 -p kd:=1.0
```

They can also be changed while the controller is running (applied on the next step),
from the host with ROS 2 Jazzy installed:
```bash
export ROS_DOMAIN_ID=5
ros2 param set /gait_controller kp 15.0
ros2 param set /gait_controller kd 1.0
```
or from inside the running container:
```bash
docker exec -it <container> bash -c "source /opt/ros/jazzy/setup.bash && ros2 param set /gait_controller kp 15.0"
```

## Run logs

Each run writes CSV files and debug plots to `logs/<prefix>_<timestamp>/`. The
`-v $(pwd)/logs:/workspace/logs` mount above is what makes these land on the host
machine instead of being lost inside the container.

| Prefix | When |
|---|---|
| `sim_` | `--sim` |
| `real_` | `--real`, simulator not running |
| `real_and_sim_` | `--real` while the `sim2sim:mujoco` container is running |
| `sim_nocmd_`, `real_nocmd_`, `real_and_sim_nocmd_` | the same with `--no-cmd` |

Example: `logs/real_and_sim_nocmd_20261026_171500/`.

Files: `commands.csv` (targets sent in `LowCmd`; with `--no-cmd` — targets the policy
would have sent), `motor_positions.csv`,
`motor_velocities.csv`, `motor_torques.csv`, `imu.csv`; plots `joint_tracking.png`,
`motor_velocities.png`, `motor_torques.png`, `imu_orientation.png`, `imu_rates.png`.

Logging starts only after every logged source has sent its first state, so a run
never begins with empty rows; the controller itself runs from the first robot state.

### Real robot mirrored in the simulator (`real_and_sim_`)

Both `hardware_node` and `sim_mujoco.py` subscribe to `/low_level_cmd`, so with `--real`
the simulator executes the same commands as the robot. The policy still runs only on the
robot's state (`/low_level_state_real`); the simulator's state (`/low_level_state`) is
logged next to it at the same controller steps:

1. start the `sim2sim:mujoco` container;
2. start the controller with `--real`.

With `--real --no-cmd` neither the robot nor the simulator gets commands from the policy,
so the comparison shows how both behave under the other nodes' commands (or none).

The simulator is detected on the first control step, so it must already be running.
Its data goes to the same files with a `sim_` prefix (`sim_motor_positions.csv`, ...),
and every plot overlays real (solid) and sim (dashed).

The simulator is open loop here: it gets commands computed from the robot's state, not
its own, so any difference between them grows and the simulated robot drifts away or falls.
Joint-level curves (how the motors track the same targets) stay comparable; IMU curves only
until the two diverge.

To get a shell in the image instead of the controller: `docker run -it --entrypoint bash gait_controller:jazzy`.
