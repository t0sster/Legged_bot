import os
import numpy as np
import rclpy
import rclpy.parameter
from abc import ABC, abstractmethod
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from .factories import InputDeviceFactory
from .inference_controller import InferenceController
from .bd_kinematics import BDKinematics
from .lipm_planner import LIPMStepPlanner
from tinker_msgs.msg import LowState, LowCmd, MotorCmd


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


LOWCMD_JOINT_NAMES = [
    "J_L0",
    "J_L1",
    "J_L2",
    "J_L3",
    "J_L4_ankle",
    "J_R0",
    "J_R1",
    "J_R2",
    "J_R3",
    "J_R4_ankle",
]


class BaseGaitAdapter(ABC):
    def __init__(self, node: Node, kinematics: BDKinematics, dt: float):
        self.node = node
        self.kinematics = kinematics
        self.dt = dt
        self.obs_layout = 'legacy'
        self.init_stance_half_width = 0.054
        self.nominal_com_height = 0.24

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def build_observation_payload(
        self,
        base_pos: np.ndarray,
        base_vel: np.ndarray,
        base_heading: float,
        imu_quat: np.ndarray,
        commands: np.ndarray,
        joint_positions: np.ndarray,
    ) -> dict:
        pass


class LegacyLipAdapter(BaseGaitAdapter):
    def __init__(self, node: Node, kinematics: BDKinematics, dt: float):
        super().__init__(node, kinematics, dt)
        self.init_stance_half_width = 0.054
        self.nominal_com_height = 0.34

        step_period = 25   # half-cycle in controller steps = 0.25 s at 100 Hz
        self.lipm = LIPMStepPlanner(
            dt=dt,
            step_period=step_period,
            dstep_width=0.24,
            dstep_length=0.05,
            stride_compensation_gain=0.0,
            stride_compensation_max_ratio=0.0,
            use_cmd_heading=False,
        )
        self.lipm.reset(init_stance_half_width=self.init_stance_half_width)

    def reset(self) -> None:
        self.lipm.reset(init_stance_half_width=self.init_stance_half_width)

    def build_observation_payload(
        self,
        base_pos: np.ndarray,
        base_vel: np.ndarray,
        base_heading: float,
        imu_quat: np.ndarray,
        commands: np.ndarray,
        joint_positions: np.ndarray,
    ) -> dict:
        foot_right, foot_left, _, _ = self.kinematics.compute(joint_positions)

        q_xyzw = np.array(
            [imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]], dtype=np.float64
        )
        R_base = R.from_quat(q_xyzw).as_matrix()

        foot_pos_right_world = base_pos + R_base @ foot_right[:3]
        foot_pos_left_world = base_pos + R_base @ foot_left[:3]
        foot_head_right_world = float(foot_right[3]) + base_heading
        foot_head_left_world = float(foot_left[3]) + base_heading

        com_body = self.kinematics.compute_com(joint_positions).astype(np.float64)
        com_world = base_pos + R_base @ com_body

        self.lipm.update(
            base_pos=base_pos.astype(float),
            base_vel=base_vel.astype(float),
            base_heading=base_heading,
            commands=commands.astype(float),
            foot_pos_right_world=foot_pos_right_world.astype(float),
            foot_pos_left_world=foot_pos_left_world.astype(float),
            foot_heading_right=foot_head_right_world,
            foot_heading_left=foot_head_left_world,
            com=com_world,
        )

        step_cmd_right, step_cmd_left = self.lipm.get_step_commands_body(
            base_pos.astype(float), q_xyzw
        )
        phase_sin, phase_cos = self.lipm.get_phase_obs()

        return {
            'foot_states_right': foot_right,
            'foot_states_left': foot_left,
            'step_cmd_right': step_cmd_right,
            'step_cmd_left': step_cmd_left,
            'phase_sin': phase_sin,
            'phase_cos': phase_cos,
            'obs_layout': 'legacy',
        }


class GaitCommandSampler:
    def __init__(
        self,
        freq_range: tuple[float, float],
        offset_range: tuple[float, float],
        duration_range: tuple[float, float],
        resample_time_s: float,
        randomize: bool = True,
    ):
        self.freq_range = freq_range
        self.offset_range = offset_range
        self.duration_range = duration_range
        self.resample_time_s = resample_time_s
        self.randomize = randomize
        self._elapsed = 0.0

        self.frequency = 0.5 * sum(freq_range)
        self.offset = 0.5 * sum(offset_range)
        self.duration = 0.5 * sum(duration_range)

    def update(self, dt: float) -> bool:
        self._elapsed += dt
        if self._elapsed < self.resample_time_s:
            return False
        self._elapsed = 0.0
        self._resample()
        return True

    def _resample(self) -> None:
        if self.randomize:
            self.frequency = np.random.uniform(*self.freq_range)
            self.offset = np.random.uniform(*self.offset_range)
            self.duration = np.random.uniform(*self.duration_range)
        else:
            self.frequency = 0.5 * sum(self.freq_range)
            self.offset = 0.5 * sum(self.offset_range)
            self.duration = 0.5 * sum(self.duration_range)


class BDLipAdapter(BaseGaitAdapter):
    def __init__(self, node: Node, kinematics: BDKinematics, dt: float):
        super().__init__(node, kinematics, dt)
        self.obs_layout = 'bd_lip'

        self.init_stance_half_width = 0.054
        self.nominal_com_height = 0.36
        self._base_height_command = 0.25
        self._dstep_width = 0.20

        self.gait_cmd = GaitCommandSampler(
            freq_range=(1.0, 2.0),
            offset_range=(0.5, 0.5),
            duration_range=(0.5, 0.5),
            resample_time_s=5.0,
            randomize=True,
        )

        initial_period_s = 0.5 / max(self.gait_cmd.frequency, 1e-3)
        step_period_steps = int(round(initial_period_s / dt))
        self.lipm = LIPMStepPlanner(
            dt=dt,
            step_period=step_period_steps,
            dstep_width=self._dstep_width,
            dstep_length=0.05,
            stride_compensation_gain=0.5,
            stride_compensation_max_ratio=0.5,
            use_cmd_heading=True,
            heading_speed_eps=1e-3,
        )
        self.lipm.reset(init_stance_half_width=self.init_stance_half_width)

    def reset(self) -> None:
        self.lipm.reset(init_stance_half_width=self.init_stance_half_width)

    def build_observation_payload(
        self,
        base_pos: np.ndarray,
        base_vel: np.ndarray,
        base_heading: float,
        imu_quat: np.ndarray,
        commands: np.ndarray,
        joint_positions: np.ndarray,
    ) -> dict:
        if self.gait_cmd.update(self.dt):
            period_s = 0.5 / max(self.gait_cmd.frequency, 1e-3)
            period_steps = int(round(period_s / self.dt))
            self.lipm.set_step_period_steps(period_steps)

        foot_right, foot_left, _, _ = self.kinematics.compute(joint_positions)

        q_xyzw = np.array(
            [imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]], dtype=np.float64
        )
        R_base = R.from_quat(q_xyzw).as_matrix()

        foot_pos_right_world = base_pos + R_base @ foot_right[:3]
        foot_pos_left_world = base_pos + R_base @ foot_left[:3]
        foot_head_right_world = float(foot_right[3]) + base_heading
        foot_head_left_world = float(foot_left[3]) + base_heading

        com_body = self.kinematics.compute_com(joint_positions).astype(np.float64)
        com_world = base_pos + R_base @ com_body

        self.lipm.update(
            base_pos=base_pos.astype(float),
            base_vel=base_vel.astype(float),
            base_heading=base_heading,
            commands=commands.astype(float),
            foot_pos_right_world=foot_pos_right_world.astype(float),
            foot_pos_left_world=foot_pos_left_world.astype(float),
            foot_heading_right=foot_head_right_world,
            foot_heading_left=foot_head_left_world,
            com=com_world,
        )

        step_cmd_right_world = self.lipm.step_commands[0].copy()
        step_cmd_left_world = self.lipm.step_commands[1].copy()

        def _step_command_rel(step_cmd_world: np.ndarray) -> np.ndarray:
            rel_pos = R.from_quat(q_xyzw).inv().apply(
                step_cmd_world[:3].astype(np.float64) - base_pos.astype(np.float64)
            )
            rel_yaw = _wrap_to_pi(float(step_cmd_world[2]) - base_heading)
            return np.array([rel_pos[0], rel_pos[1], rel_pos[2], rel_yaw], dtype=np.float32)

        foot_target_right = _step_command_rel(step_cmd_right_world)
        foot_target_left = _step_command_rel(step_cmd_left_world)
        phase_sin, phase_cos = self.lipm.get_phase_obs()

        return {
            'foot_states_right': foot_right,
            'foot_states_left': foot_left,
            'foot_target_right': foot_target_right,
            'foot_target_left': foot_target_left,
            'phase_sin': phase_sin,
            'phase_cos': phase_cos,
            'base_height_command': self._base_height_command,
            'gait_phase': np.array([phase_sin, phase_cos], dtype=np.float32),
            'obs_layout': 'bd_lip',
        }


# Backwards-compatible alias for older launch scripts/configs.
LipPlayAdapter = BDLipAdapter


class GaitAdapterFactory:
    _ADAPTERS = {
        'legacy': LegacyLipAdapter,
        'bd_lip': BDLipAdapter,
        'lip_play': BDLipAdapter,
    }

    @staticmethod
    def create(adapter_type: str, node: Node, kinematics: BDKinematics, dt: float) -> BaseGaitAdapter:
        adapter_class = GaitAdapterFactory._ADAPTERS.get(adapter_type.lower())
        if adapter_class is None:
            raise ValueError(f'Unknown gait adapter type: {adapter_type}')
        return adapter_class(node=node, kinematics=kinematics, dt=dt)


class GaitController(Node):
    # Velocity estimation: low-pass decay on integrated accelerometer velocity.
    # Reduces drift while preserving short-term dynamics.
    _VEL_DECAY             = 0.98

    def __init__(self, device_type: str, model_path: str, gait_mode: str = 'legacy'):
        super().__init__(
            'gait_controller',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time',
                    rclpy.parameter.Parameter.Type.BOOL,
                    True)
            ]
        )

        self.device = InputDeviceFactory.get_device(device_type, node=self)
        self.device.initialize()

        self.inference_controller = InferenceController(
            node=self, model_dir=model_path, robot_type='tinker')
        self._action_joint_names = list(self.inference_controller.joint_names)
        self._lowcmd_action_index = self._build_lowcmd_action_index()

        # Forward kinematics (Pinocchio)
        self.kinematics = BDKinematics()

        # Gait adapter (legacy or bd_lip)
        loop_freq = float(self.inference_controller.loop_frequency)
        dt = 1.0 / loop_freq
        self.gait_adapter = GaitAdapterFactory.create(
            gait_mode, node=self, kinematics=self.kinematics, dt=dt
        )
        self.inference_controller.set_observation_layout(self.gait_adapter.obs_layout)

        # Sensor state
        self.imu_quat  = np.array([1., 0., 0., 0.], dtype=np.float32)  # wxyz
        self.ang_vel   = np.zeros(3, dtype=np.float32)
        self.accel     = np.zeros(3, dtype=np.float32)
        self.rpy       = np.zeros(3, dtype=np.float32)
        self.positions = np.zeros(10, dtype=np.float32)
        self.velocities = np.zeros(10, dtype=np.float32)
        self.commands  = np.zeros(3, dtype=np.float32)

        # Velocity / position estimate (world frame, relative to start)
        self._dt            = dt
        self._base_vel_world = np.zeros(3, dtype=np.float64)
        self._base_pos_world = np.zeros(3, dtype=np.float64)
        self._base_pos_world[2] = self.gait_adapter.nominal_com_height  # start at nominal height

        self.first_state_received = False
        self._step_count = 0
        self._LOG_EVERY = 50  # print every N control steps (1 s at 50 Hz)

        self.lowstate_subscriber = self.create_subscription(
            LowState, '/low_level_state', self.lowstate_callback, 10)
        self.lowcmd_publisher = self.create_publisher(
            LowCmd, '/low_level_cmd', 10)

        control_dt = 1.0 / loop_freq
        self.control_timer = self.create_timer(control_dt, self.control_loop)

    def _build_lowcmd_action_index(self) -> list[int]:
        """Map policy action order to LowCmd motor order by joint name."""
        index_by_name = {name: idx for idx, name in enumerate(self._action_joint_names)}
        missing = [name for name in LOWCMD_JOINT_NAMES if name not in index_by_name]
        if missing:
            raise ValueError(
                f"Policy joint list does not cover LowCmd joints: missing={missing}, "
                f"policy_order={self._action_joint_names}"
            )

        lowcmd_action_index = [index_by_name[name] for name in LOWCMD_JOINT_NAMES]
        if self._action_joint_names != LOWCMD_JOINT_NAMES:
            self.get_logger().info(
                f"Policy action order {self._action_joint_names} -> LowCmd order {LOWCMD_JOINT_NAMES}"
            )
        return lowcmd_action_index

    def lowstate_callback(self, msg: LowState):
        imu = msg.imu_state
        self.ang_vel   = np.array(imu.gyroscope,    dtype=np.float32)
        self.imu_quat  = np.array(imu.quaternion,   dtype=np.float32)  # wxyz
        self.accel     = np.array(imu.accelerometer, dtype=np.float32)
        self.rpy       = np.array(imu.rpy,          dtype=np.float32)
        self.positions = np.array([m.position for m in msg.motor_state], dtype=np.float32)
        self.velocities = np.array([m.velocity for m in msg.motor_state], dtype=np.float32)

        if not self.first_state_received:
            self.get_logger().info('First state received — control loop active.')
            self.first_state_received = True

        if not self.first_state_received:
            print(f'first state: positions={[m.position for m in msg.motor_state]}')


    def control_loop(self):
        
        try:
            if not self.first_state_received:
                return

            self.commands = np.array(self.device.get_commands(), dtype=np.float32)

            # State estimation
            self._update_velocity_estimate()

            base_pos = self._base_pos_world.astype(np.float32)
            base_vel = self._base_vel_world.astype(np.float32)

            # Quaternion xyzw for scipy
            q_xyzw = np.array([
                self.imu_quat[1], self.imu_quat[2],
                self.imu_quat[3], self.imu_quat[0]], dtype=np.float64)

            # Base heading (yaw from IMU RPY, published by sim)
            base_heading = float(self.rpy[2])

            obs_payload = self.gait_adapter.build_observation_payload(
                base_pos=base_pos,
                base_vel=base_vel,
                base_heading=base_heading,
                imu_quat=self.imu_quat,
                commands=self.commands,
                joint_positions=self.positions,
            )

            # Observations
            self.inference_controller.compute_observation(
                imu_quat=self.imu_quat,
                imu_rpy=self.rpy,
                base_ang_vel=self.ang_vel,
                joint_positions=self.positions,
                joint_velocities=self.velocities,
                commands=self.commands,
                **obs_payload,
            )

            # self._log_observations(
            #     base_heading, foot_right, foot_left,
            #     step_cmd_right, step_cmd_left,
            #     phase_sin, phase_cos,
            # )

            self._step_count += 1

            # Policy inference
            self.inference_controller.compute_actions()
            actions = self.inference_controller.actions.copy()

            clip_act = float(
                self.inference_controller.rl_cfg['clip_scales']['clip_actions'])
            actions = np.clip(actions, -clip_act, clip_act)

            q_des = (actions * self.inference_controller.control_cfg['action_scale_pos']
                        + self.inference_controller.init_joint_angles)

            self.publish_lowcmd_action(q_des)

        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')

    # Just function for logging
    def _log_observations(
        self,
        base_heading,
        foot_right, foot_left,
        step_cmd_right, step_cmd_left,
        phase_sin, phase_cos
    ):
        ic = self.inference_controller
        dof_pos = self.positions * ic.obs_scales['dof_pos']
        dof_vel = self.velocities * ic.obs_scales['dof_vel']
        ang_vel = self.ang_vel * ic.obs_scales['ang_vel']

        q_xyzw = np.array([
            self.imu_quat[1], self.imu_quat[2],
            self.imu_quat[3], self.imu_quat[0]], dtype=np.float64)
        from scipy.spatial.transform import Rotation as _R
        proj_grav = _R.from_quat(q_xyzw).inv().apply([0., 0., -1.])

        lines = [
            f'=== step {self._step_count} ===',
            f'  base_heading:      {base_heading:.4f}',
            f'  base_ang_vel:      {ang_vel}',
            f'  projected_gravity: {proj_grav}',
            f'  foot_states_right: {foot_right}',
            f'  foot_states_left:  {foot_left}',
            f'  step_cmd_right:    {step_cmd_right}',
            f'  step_cmd_left:     {step_cmd_left}',
            f'  commands:          {self.commands}',
            f'  phase_sin:         {phase_sin:.4f}',
            f'  phase_cos:         {phase_cos:.4f}',
            f'  dof_pos: {dof_pos}',
            f'  dof_vel: {dof_vel}',
        ]
        print('\n'.join(lines), flush=True)


    def _update_velocity_estimate(self):

        q_xyzw = np.array([
            self.imu_quat[1], self.imu_quat[2],
            self.imu_quat[3], self.imu_quat[0]], dtype=np.float64)
        R_mat = R.from_quat(q_xyzw).as_matrix()

        # Rotate accelerometer reading to world frame, then remove gravity
        acc_world = R_mat @ self.accel.astype(np.float64)
        acc_world[2] -= 9.81

        # Integrate with decay to limit long-term drift
        self._base_vel_world = (
            self._base_vel_world * self._VEL_DECAY + acc_world * self._dt)
        self._base_pos_world += self._base_vel_world * self._dt

        # Keep z at nominal CoM height (flat ground assumption)
        self._base_pos_world[2] = self.gait_adapter.nominal_com_height


    def publish_lowcmd_action(self, action: np.ndarray):
        msg = LowCmd()
        msg.motor_cmd = [MotorCmd() for _ in range(10)]
        kp = float(self.inference_controller.control_cfg['stiffness'])
        kd = float(self.inference_controller.control_cfg['damping'])

        for low_idx, action_idx in enumerate(self._lowcmd_action_index):
            msg.motor_cmd[low_idx].position = float(action[action_idx])
            msg.motor_cmd[low_idx].velocity = 0.0
            msg.motor_cmd[low_idx].torque = 0.0
            msg.motor_cmd[low_idx].kp = kp
            msg.motor_cmd[low_idx].kd = kd

        self.lowcmd_publisher.publish(msg)

    def shutdown(self):
        self.device.shutdown()
        self.destroy_node()
