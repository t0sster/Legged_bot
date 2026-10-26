import os
import numpy as np
import onnxruntime as ort
from rclpy.node import Node


class InferenceController:
    """ONNX actor of the Mjlab-Tinker-Upd task (biped-mjrl, tasks/tinker_upd)."""

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

        self.actions = np.zeros(self.actions_size, dtype=np.float32)
        self.observations = np.zeros(self.observations_size, dtype=np.float32)

        self.node.get_logger().info('Inference controller initialised')

    def load_config(self):
        # Values from TinkerRobotCfg (mj_biped/assets/zoo/robots/tinker/mjlab_cfg.py),
        # see also params/env.yaml of the training run.
        self.joint_names = [
            'joint_l_yaw', 'joint_l_roll', 'joint_l_pitch', 'joint_l_knee', 'joint_l_ankle',
            'joint_r_yaw', 'joint_r_roll', 'joint_r_pitch', 'joint_r_knee', 'joint_r_ankle',
        ]
        self.init_joint_angles = np.array(
            [0.0, 0.0, 0.85, 1.5, 0.65,
             0.0, 0.0, -0.85, -1.5, -0.65], dtype=np.float32)
        # JointPositionActionCfg.scale: 0.3 for every joint, use_default_offset=True.
        self.action_scale = 0.8
        self.actions_size = 10
        self.observations_size = 41
        # decimation(10) * physics dt(0.002s) = 0.02s -> 50 Hz.
        self.loop_frequency = 50

    def compute_observation(
        self,
        base_ang_vel: np.ndarray,       # (3,) base_link frame
        projected_gravity: np.ndarray,  # (3,) base_link frame
        velocity_command: np.ndarray,   # (3,) filtered command
        joint_pos: np.ndarray,          # (10,) relative to init_joint_angles
        joint_vel: np.ndarray,          # (10,)
        last_action: np.ndarray,        # (10,)
        gait_phase: np.ndarray,         # (2,)
    ):
        # Actor term order of tinker_upd/env_factory.py.
        self.observations = np.concatenate([
            base_ang_vel,
            projected_gravity,
            velocity_command,
            joint_pos,
            joint_vel,
            last_action,
            gait_phase,
        ]).astype(np.float32)

    def compute_actions(self):
        output = self.policy_session.run(
            [self.policy_output_name],
            {self.policy_input_name: self.observations.reshape(1, -1)},
        )
        self.actions = np.asarray(output[0], dtype=np.float32).flatten()
