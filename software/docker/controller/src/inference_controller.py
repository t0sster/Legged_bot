import os
import numpy as np
import yaml
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node


class InferenceController:

    def __init__(self, node: Node, model_dir: str, robot_type: str):
        self.node = node

        self.config_file = os.path.join(model_dir, 'params.yaml')
        self.model_file  = os.path.join(model_dir, 'policy', 'policy.onnx')

        self.load_config(self.config_file)

        self.policy_session      = ort.InferenceSession(self.model_file)
        self.policy_input_name   = self.policy_session.get_inputs()[0].name
        self.policy_output_name  = self.policy_session.get_outputs()[0].name

        inp = self.policy_session.get_inputs()[0]
        self.node.get_logger().info(
            f'ONNX loaded: input={self.policy_input_name} shape={inp.shape}')

        # Infer expected observation size from the ONNX model
        self._obs_needs_batch = (len(inp.shape) == 2)
        if self._obs_needs_batch:
            self._expected_obs_size = int(inp.shape[1])
        else:
            self._expected_obs_size = int(inp.shape[0])

        self.actions      = np.zeros(self.actions_size)
        self.observations = np.zeros(self._expected_obs_size)
        self._first_obs_logged = False
        self._first_act_logged = False
        self._pending_debug_lines = []
        self.observation_layout = 'legacy'
        self._default_base_height = 0.25

        self.node.get_logger().info('Inference controller initialised')


    def load_config(self, config_file: str):
        with open(config_file) as f:
            config = yaml.safe_load(f)

        cfg = config['TinkerCfg']
        self.joint_names      = cfg['joint_names']
        self.init_state       = cfg['init_state']['default_joint_angle']
        self.stand_duration   = cfg['stand_mode']['stand_duration']
        self.control_cfg      = cfg['control']
        self.rl_cfg           = cfg['normalization']
        self.obs_scales       = cfg['normalization']['obs_scales']
        self.actions_size     = cfg['size']['actions_size']
        self.observations_size = cfg['size']['observations_size']
        self.imu_orientation_offset = np.array(
            list(cfg['imu_orientation_offset'].values()))
        self.user_cmd_cfg     = cfg['user_cmd_scales']
        self.loop_frequency   = cfg['loop_frequency']

        # Action order follows the policy output / LowCmd packing order.
        self.init_joint_angles = np.array(
            [self.init_state[n] for n in self.joint_names], dtype=np.float32)

        # Observation order follows the simulator / IsaacLab state order.
        self.obs_joint_names = [
            "J_L0", "J_L1", "J_L2", "J_L3", "J_L4_ankle",
            "J_R0", "J_R1", "J_R2", "J_R3", "J_R4_ankle",
        ]
        self.init_joint_angles_obs = np.array(
            [self.init_state[n] for n in self.obs_joint_names], dtype=np.float32)

        self.node.get_logger().info(
            f'Config loaded: obs_size={self.observations_size} '
            f'actions_size={self.actions_size}')

    def set_observation_layout(self, layout: str) -> None:
        self.observation_layout = layout

    def set_default_base_height(self, height: float) -> None:
        self._default_base_height = float(height)

    def compute_observation(
        self,
        imu_quat:            np.ndarray,   # (4,) wxyz
        imu_rpy:             np.ndarray,   # (3,) [roll, pitch, yaw] from sim
        base_ang_vel:        np.ndarray,   # (3,)
        joint_positions:     np.ndarray,   # (10,)
        joint_velocities:    np.ndarray,   # (10,)
        commands:            np.ndarray,   # (3,) [vx, vy, yaw_rate]
        foot_states_right:   np.ndarray,   # (4,) from BDKinematics
        foot_states_left:    np.ndarray,   # (4,)
        foot_target_right:   np.ndarray,   # (4,) from LIPMStepPlanner
        foot_target_left:    np.ndarray,   # (4,)
        phase_sin:           float,
        phase_cos:           float,
        base_height_command: float | None = None,
        gait_phase:          np.ndarray | None = None,
        obs_layout:          str | None = None,
    ):
        try:
            layout = (obs_layout or self.observation_layout).lower()
            if layout == 'lip_play':
                layout = 'bd_lip'
            # Quaternion wxyz -> xyzw for scipy
            q_xyzw = np.array(
                [imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]],
                dtype=np.float64)
            rot = R.from_quat(q_xyzw)


            # base_heading
            base_heading = np.float32(imu_rpy[2])

            # projected_gravity
            projected_gravity = (rot.inv().apply(np.array([0., 0., -1.]))
                                 + np.random.uniform(-0.05, 0.05, 3)).astype(np.float32)

            # command scaling
            cmd_scale = np.array([
                self.user_cmd_cfg['lin_vel_x'],
                self.user_cmd_cfg['lin_vel_y'],
                self.user_cmd_cfg['ang_vel_yaw'],
            ], dtype=np.float32)
            scaled_commands = (commands * cmd_scale).astype(np.float32)

            # Observation noise matching Isaac Lab training distribution (uniform, add)
            noisy_ang_vel  = base_ang_vel    + np.random.uniform(-0.2,  0.2,  3).astype(np.float32)
            # IsaacLab trains on joint_pos_rel, i.e. positions relative to the default pose.
            joint_pos_rel = joint_positions - self.init_joint_angles_obs
            noisy_dof_pos  = joint_pos_rel + np.random.uniform(-0.01, 0.01, 10).astype(np.float32)
            noisy_dof_vel  = joint_velocities + np.random.uniform(-1.5,  1.5,  10).astype(np.float32)

            # joint state
            scaled_dof_pos = (noisy_dof_pos * self.obs_scales['dof_pos']).astype(np.float32)
            scaled_dof_vel = (noisy_dof_vel * self.obs_scales['dof_vel']).astype(np.float32)
            scaled_ang_vel = (noisy_ang_vel  * self.obs_scales['ang_vel']).astype(np.float32)

            if layout == 'legacy':
                obs = np.concatenate([
                    [base_heading],         # 1
                    scaled_ang_vel,         # 3
                    projected_gravity,      # 3
                    foot_states_right,      # 4
                    foot_states_left,       # 4
                    foot_target_right,      # 4
                    foot_target_left,       # 4
                    scaled_commands,        # 3
                    [phase_sin],            # 1
                    [phase_cos],            # 1
                    scaled_dof_pos,         # 10
                    scaled_dof_vel,         # 10
                ]).astype(np.float32)
            elif layout == 'bd_lip':
                if gait_phase is None:
                    gait_phase = np.array([phase_sin, phase_cos], dtype=np.float32)
                if base_height_command is None:
                    base_height_command = self._default_base_height
                obs = np.concatenate([
                    [base_heading],         # 1
                    scaled_ang_vel,         # 3
                    projected_gravity,      # 3
                    foot_states_right,      # 4
                    foot_states_left,       # 4
                    foot_target_right,      # 4
                    foot_target_left,       # 4
                    scaled_commands,        # 3
                    [base_height_command],  # 1
                    gait_phase,             # 2
                    scaled_dof_pos,         # 10
                    scaled_dof_vel,         # 10
                ]).astype(np.float32)
            else:
                raise ValueError(f'Unknown observation layout: {layout}')

            self.observations = obs

        except Exception as e:
            self.node.get_logger().error(
                f'[Inference] compute_observation error: {e}')

    def compute_actions(self):
        try:
            clip = float(self.rl_cfg['clip_scales']['clip_observations'])
            obs  = np.clip(self.observations, -clip, clip).astype(np.float32)


            if self._obs_needs_batch:
                inp = obs.reshape(1, -1)
            else:
                inp = obs.flatten()

            output = self.policy_session.run(
                [self.policy_output_name],
                {self.policy_input_name: inp})

            self.actions = np.asarray(output[0], dtype=np.float32).flatten()

            if not self._first_act_logged:
                lines = getattr(self, '_pending_debug_lines', [])
                lines += ['=== First actions ===']
                for i, val in enumerate(self.actions):
                    lines.append(f'  [{i:2d}] action[{i}] = {val:.6f}')
                print('\n'.join(lines), flush=True)
                self._first_act_logged = True

        except Exception as e:
            self.node.get_logger().error(
                f'[Inference] compute_actions error: {e}')
