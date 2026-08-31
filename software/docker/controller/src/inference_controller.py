import os
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node


class InferenceController:

    def __init__(self, node: Node, model_dir: str):
        self.node = node

        self.model_file = os.path.join(model_dir, 'policy.onnx')

        self.load_config()

        self.policy_session = ort.InferenceSession(self.model_file)
        self.policy_input_name = self.policy_session.get_inputs()[0].name
        self.policy_output_name = self.policy_session.get_outputs()[0].name

        input_info = self.policy_session.get_inputs()[0]
        self.node.get_logger().info(
            f'ONNX loaded: input={self.policy_input_name} shape={input_info.shape}')

        if len(input_info.shape) == 2:
            self.obs_needs_batch = True
        else:
            self.obs_needs_batch = False

        self.actions = np.zeros(self.actions_size, dtype=np.float32)
        self.observations = np.zeros(self.observations_size, dtype=np.float32)
        self.first_action_logged = False

        self.node.get_logger().info('Inference controller initialised')

    def load_config(self):
        self.joint_names = [
            'joint_l_yaw', 'joint_l_roll', 'joint_l_pitch', 'joint_l_knee', 'joint_l_ankle',
            'joint_r_yaw', 'joint_r_roll', 'joint_r_pitch', 'joint_r_knee', 'joint_r_ankle',
        ]

        self.init_state = {
            'joint_l_yaw':   0.0,
            'joint_l_roll':  0.0,
            'joint_l_pitch': 0.45,
            'joint_l_knee':  0.9,
            'joint_l_ankle': 0.45,
            'joint_r_yaw':   0.0,
            'joint_r_roll':  0.0,
            'joint_r_pitch': -0.45,
            'joint_r_knee':  -0.9,
            'joint_r_ankle': -0.45,
        }
        self.stand_duration = 1.0

        # Per-joint-type action scale (JointPositionActionCfg.scale in training).
        action_scale_by_type = {
            'yaw':   0.25,
            'roll':  0.15,
            'pitch': 0.4,
            'knee':  0.35,
            'ankle': 0.25,
        }
        action_scale_pos = np.zeros(len(self.joint_names), dtype=np.float32)
        for i, name in enumerate(self.joint_names):
            joint_type = name.split('_')[-1]
            action_scale_pos[i] = action_scale_by_type[joint_type]

        self.control_cfg = {
            'action_scale_pos': action_scale_pos,
        }
        self.rl_cfg = {
            'clip_scales': {
                'clip_observations': 100.0,
                # agent.yaml: clip_actions: null -- training never clipped actions
                # (mjlab RslRlVecEnvWrapper only clamps when clip_actions is not
                # None). None here means "don't clip" in gait_controller.py.
                'clip_actions': None,
            },
        }
        self.obs_scales = {
            'lin_vel': 1.0,
            'ang_vel': 1.0,
            'dof_pos': 1.0,
            'dof_vel': 1.0,
        }
        self.user_cmd_cfg = {
            'lin_vel_x': 1.0,
            'lin_vel_y': 1.0,
            'ang_vel_yaw': 1.0,
        }
        self.actions_size = 10
        self.observations_size = 49
        # Policy control rate = decimation(10) * physics dt(0.002s) = 0.02s -> 50 Hz.
        self.loop_frequency = 50
        self.init_joint_angles = np.zeros(len(self.joint_names), dtype=np.float32)
        for i, name in enumerate(self.joint_names):
            self.init_joint_angles[i] = self.init_state[name]

    def compute_observation(
        self,
        imu_rpy: np.ndarray,   # (3,) 
        imu_gyro: np.ndarray,   # (3,)
        imu_accel: np.ndarray,   # (3,)
        velocity_command: np.ndarray,   # (3,)
        gait_phase: np.ndarray,   # (7,)
        joint_pos:  np.ndarray,   # (10,)
        joint_vel: np.ndarray,   # (10,) 
        last_action: np.ndarray,   # (10,)
    ):
        try:

            obs = np.concatenate([
                imu_rpy,     # 3
                imu_gyro,     # 3
                imu_accel,  # 3
                velocity_command,    # 3
                gait_phase,        # 7
                joint_pos,        # 10
                joint_vel,     # 10
                last_action,     # 10
            ])

            self.observations = obs.astype(np.float32)

        except Exception as e:
            self.node.get_logger().error(f'compute_observation error: {e}')

    def compute_actions(self):
        try:
            clip_value = float(self.rl_cfg['clip_scales']['clip_observations'])
            obs = np.clip(self.observations, -clip_value, clip_value)
            obs = obs.astype(np.float32)

            if self.obs_needs_batch:
                model_input = obs.reshape(1, -1)
            else:
                model_input = obs.flatten()

            output = self.policy_session.run(
                [self.policy_output_name],
                {self.policy_input_name: model_input},
            )

            self.actions = np.asarray(output[0], dtype=np.float32).flatten()

        except Exception as e:
            self.node.get_logger().error(f'compute_actions error: {e}')

