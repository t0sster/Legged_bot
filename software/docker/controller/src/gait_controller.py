import argparse
import sys

import numpy as np
import rclpy
import rclpy.parameter
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from scipy.spatial.transform import Rotation as R

from .factories import InputDeviceFactory
from .inference_controller import InferenceController
from .data_logger import RunLogger, state_from_msg
from tinker_msgs.msg import LowState, LowCmd, MotorCmd


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class GaitController(Node):

    # SmoothVelocityCommandCfg (tinker_upd/mdp/commands.py).
    _LINEAR_DEADBAND = 0.02  # m/s
    _LINEAR_WALK = 0.05
    _YAW_DEADBAND = 0.05     # rad/s
    _YAW_WALK = 0.15
    _TRANSITION_TIME = 0.2   # s

    # TinkerRobotCfg.nominal_gait_period; training picks offset 0 or 0.5 per episode.
    _GAIT_PERIOD = 0.7
    _GAIT_PHASE_OFFSET = 0.0

    # Rotation imu site -> base_link (torso quat "0 0 0 1" * imu quat in tinker.xml).
    _R_BASE_IMU = np.array([[0., 1., 0.],
                            [-1., 0., 0.],
                            [0., 0., 1.]], dtype=np.float32)

    # sim_mujoco.py publishes /low_level_state, hardware_node /low_level_state_real.
    _SIM_STATE_TOPIC = '/low_level_state'
    _REAL_STATE_TOPIC = '/low_level_state_real'

    # def __init__(self, device_type: str, model_path: str):
    def __init__(self, model_path: str, use_sim: bool, velocity_command, send_commands: bool = True):
        super().__init__(
            'gait_controller',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time',
                    rclpy.parameter.Parameter.Type.BOOL,
                    use_sim)
            ]
        )

        # vx, vy (m/s), wz (rad/s); training ranges: ±0.15, ±0.1, ±0.25.
        self.velocity_command = np.array(velocity_command, dtype=np.float32)
        # False (--no-cmd): the policy still runs and everything is logged
        # (commands.csv holds what it would send), but LowCmd is never published.
        self.send_commands = send_commands
        self.use_sim = use_sim
        state_topic = self._SIM_STATE_TOPIC if use_sim else self._REAL_STATE_TOPIC

        # Training: stiffness 40, damping 1.0; default 15 is the Raspberry driver kp limit.
        # Read every step, so `ros2 param set /gait_controller kp 40.0` works at runtime.
        self.declare_parameter('kp', 15.0)
        self.declare_parameter('kd', 0.65)

        # self.device = InputDeviceFactory.get_device(device_type, node=self)
        # self.device.initialize()

        self.inference_controller = InferenceController(
            node=self, model_dir=model_path)

        loop_freq = float(self.inference_controller.loop_frequency)

        # Sensor state
        self.imu_quat  = np.array([1., 0., 0., 0.], dtype=np.float32)  # wxyz
        self.ang_vel   = np.zeros(3, dtype=np.float32)
        self.accel     = np.zeros(3, dtype=np.float32)
        self.rpy       = np.zeros(3, dtype=np.float32)
        self.positions = np.zeros(10, dtype=np.float32)
        self.velocities = np.zeros(10, dtype=np.float32)
        self.commands  = np.zeros(3, dtype=np.float32)

        # Command filter state
        self._cmd_target = np.zeros(3, dtype=np.float32)
        self._cmd_start = np.zeros(3, dtype=np.float32)
        self._cmd_elapsed = 0.0
        self._activity_start = 0.0
        self.gait_activity = 0.0

        self.first_state_received = False
        self._step_count = 0

        # Sources to log are chosen on the first control step (is the simulator
        # running alongside the robot?); the logger is created once every one of
        # them has sent a state, so a run never starts with empty rows.
        self._log_prefix = None
        self._log_sources = None
        self.logger = None
        self._state_msg = None
        self._sim_state_msg = None

        self.lowstate_subscriber = self.create_subscription(
            LowState, state_topic, self.lowstate_callback, 10)
        if not use_sim:
            # sim_mujoco.py also subscribes to /low_level_cmd, so with --real it
            # executes the same commands; its state is only logged for comparison.
            self.sim_state_subscriber = self.create_subscription(
                LowState, self._SIM_STATE_TOPIC, self.sim_state_callback, 10)
        self.lowcmd_publisher = self.create_publisher(
            LowCmd, '/low_level_cmd', 10)

        self._control_dt = 1.0 / loop_freq
        self.control_timer = self.create_timer(self._control_dt, self.control_loop)

    def lowstate_callback(self, msg: LowState):
        self._state_msg = msg
        imu = msg.imu_state
        self.imu_quat  = np.array(imu.quaternion,    dtype=np.float32)  # wxyz
        self.ang_vel   = np.array(imu.gyroscope,     dtype=np.float32)
        self.accel     = np.array(imu.accelerometer, dtype=np.float32)
        self.rpy       = np.array(imu.rpy,           dtype=np.float32)
        self.positions = np.array([m.position for m in msg.motor_state], dtype=np.float32)
        self.velocities = np.array([m.velocity for m in msg.motor_state], dtype=np.float32)

        if not self.first_state_received:
            self.get_logger().info('First state received — control loop active.')
            self.first_state_received = True

    def sim_state_callback(self, msg: LowState):
        self._sim_state_msg = msg

    def choose_log_sources(self):
        if self.use_sim:
            prefix, sources = 'sim', ['sim']
        elif self.count_publishers(self._SIM_STATE_TOPIC) > 0:
            prefix, sources = 'real_and_sim', ['real', 'sim']
        else:
            prefix, sources = 'real', ['real']
        if not self.send_commands:
            prefix += '_nocmd'
        return prefix, sources

    def log_states(self):
        """{source: state} for the logger, or None while a source is still silent."""
        msgs = {self._log_sources[0]: self._state_msg}
        if 'sim' in self._log_sources and self._log_sources[0] != 'sim':
            msgs['sim'] = self._sim_state_msg
        if any(msg is None for msg in msgs.values()):
            return None
        return {source: state_from_msg(msg) for source, msg in msgs.items()}

    def control_loop(self):
        try:
            if not self.first_state_received:
                return

            # self.commands = np.array(self.device.get_commands(), dtype=np.float32)
            self.update_command(self.velocity_command)

            self.inference_controller.compute_observation(
                base_ang_vel=self._R_BASE_IMU @ self.ang_vel,
                projected_gravity=self.projected_gravity(),
                velocity_command=self.commands,
                joint_pos=self.positions - self.inference_controller.init_joint_angles,
                joint_vel=self.velocities,
                last_action=self.inference_controller.actions,
                gait_phase=self.gait_phase_obs(),
            )

            self._step_count += 1

            # Policy inference
            self.inference_controller.compute_actions()

            q_des = (self.inference_controller.actions * self.inference_controller.action_scale
                        + self.inference_controller.init_joint_angles)

            if self._log_sources is None:
                self._log_prefix, self._log_sources = self.choose_log_sources()
                self.get_logger().info(f'Logging sources: {self._log_sources}, '
                                       'waiting for a state from each of them')

            states = self.log_states()
            if states is not None:
                if self.logger is None:
                    self.logger = RunLogger(self._log_prefix, self._log_sources)
                    self.get_logger().info(f'Logging to {self.logger.run_dir}')
                self.logger.log_step(step=self._step_count, sent_command=q_des, states=states)

            if self.send_commands:
                self.publish_lowcmd_action(q_des)

        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')

    def projected_gravity(self):
        # Unit gravity in the imu frame, then rotated into base_link.
        w, x, y, z = self.imu_quat
        gravity_imu = R.from_quat([x, y, z, w]).inv().apply([0.0, 0.0, -1.0])
        return (self._R_BASE_IMU @ gravity_imu).astype(np.float32)

    def update_command(self, raw: np.ndarray):
        """Port of SmoothVelocityCommand.compute for a single robot.

        Deadbands attenuate translation and yaw independently; a new target is
        reached in _TRANSITION_TIME along a smoothstep. self.commands is the
        command the policy observes, self.gait_activity scales the gait clock.
        """
        linear = smoothstep(
            (np.linalg.norm(raw[:2]) - self._LINEAR_DEADBAND)
            / (self._LINEAR_WALK - self._LINEAR_DEADBAND))
        yaw = smoothstep(
            (abs(raw[2]) - self._YAW_DEADBAND) / (self._YAW_WALK - self._YAW_DEADBAND))
        target = raw.copy()
        target[:2] *= linear
        target[2] *= yaw
        activity = max(linear, yaw)

        if np.any(target != self._cmd_target):
            self._cmd_start = self.commands.copy()
            self._activity_start = self.gait_activity
            self._cmd_elapsed = 0.0
        self._cmd_target = target
        self._cmd_elapsed = min(self._cmd_elapsed + self._control_dt, self._TRANSITION_TIME)

        blend = smoothstep(self._cmd_elapsed / self._TRANSITION_TIME)
        self.commands = (self._cmd_start + (target - self._cmd_start) * blend).astype(np.float32)
        self.gait_activity = float(self._activity_start + (activity - self._activity_start) * blend)

    def gait_phase_obs(self):
        time_s = self._step_count * self._control_dt
        phase = 2.0 * np.pi * ((time_s / self._GAIT_PERIOD + self._GAIT_PHASE_OFFSET) % 1.0)
        return self.gait_activity * np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

    def publish_lowcmd_action(self, action: np.ndarray):
        kp = self.get_parameter('kp').value
        kd = self.get_parameter('kd').value

        msg = LowCmd()
        msg.motor_cmd = [MotorCmd() for _ in range(10)]

        for i in range(10):
            msg.motor_cmd[i].position = float(action[i])
            msg.motor_cmd[i].kp = float(kp)
            msg.motor_cmd[i].kd = float(kd)

        self.lowcmd_publisher.publish(msg)

    def shutdown(self):
        # self.device.shutdown()
        if self.logger is not None:
            self.logger.finalize()
        self.destroy_node()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--sim', action='store_true', help='MuJoCo sim2sim: /low_level_state, sim time')
    mode.add_argument('--real', action='store_true', help='robot: /low_level_state_real, wall time')
    parser.add_argument('--vel', nargs=3, type=float, default=[0.1, 0.0, 0.0],
                        metavar=('VX', 'VY', 'WZ'), help='velocity command: m/s, m/s, rad/s')
    parser.add_argument('--no-cmd', action='store_true',
                        help='run the policy and log everything, but do not publish LowCmd')
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = GaitController(model_path='/workspace/models/params',
                          use_sim=args.sim, velocity_command=args.vel,
                          send_commands=not args.no_cmd)
    node.get_logger().info(
        f"Mode: {'sim' if args.sim else 'real'}, velocity command (vx, vy, wz): {args.vel}, "
        f"{'LowCmd NOT published (--no-cmd)' if args.no_cmd else 'LowCmd published'}")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.shutdown()
