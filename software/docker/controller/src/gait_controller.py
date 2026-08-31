import numpy as np
import rclpy
import rclpy.parameter
from rclpy.node import Node

from .factories import InputDeviceFactory
from .inference_controller import InferenceController
from .data_logger import RunLogger
from tinker_msgs.msg import LowState, LowCmd, MotorCmd


class GaitController(Node):

    _GAIT_FREQUENCY_HZ = 1.5
    _GAIT_DUTY_CYCLE = 0.5

    # def __init__(self, device_type: str, model_path: str):
    def __init__(self, model_path: str):
        super().__init__(
            'gait_controller',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time',
                    rclpy.parameter.Parameter.Type.BOOL,
                    True)
            ]
        )

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

        self.first_state_received = False
        self._step_count = 0

        self.logger = RunLogger()

        self.lowstate_subscriber = self.create_subscription(
            LowState, '/low_level_state', self.lowstate_callback, 10)
        self.lowcmd_publisher = self.create_publisher(
            LowCmd, '/low_level_cmd', 10)

        self._control_dt = 1.0 / loop_freq
        self.control_timer = self.create_timer(self._control_dt, self.control_loop)

    def lowstate_callback(self, msg: LowState):
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

    def control_loop(self):
        try:
            if not self.first_state_received:
                return

            # self.commands = np.array(self.device.get_commands(), dtype=np.float32)
            self.commands = np.array([0.2, 0.0, 0.0], dtype=np.float32)

            self.inference_controller.compute_observation(
                imu_rpy=self.rpy,
                imu_gyro=self.ang_vel,
                imu_accel=self.accel,
                velocity_command=self.commands,
                gait_phase=self.gait_phase_obs(),
                joint_pos=self.positions - self.inference_controller.init_joint_angles,
                joint_vel=self.velocities,
                last_action=self.inference_controller.actions,
            )

            self._step_count += 1

            # Policy inference
            self.inference_controller.compute_actions()

            clip_act = self.inference_controller.rl_cfg['clip_scales']['clip_actions']
            if clip_act is not None:
                actions = np.clip(self.inference_controller.actions, -float(clip_act), float(clip_act))
                self.inference_controller.actions = actions
            else:
                actions = self.inference_controller.actions

            q_des = (actions * self.inference_controller.control_cfg['action_scale_pos']
                        + self.inference_controller.init_joint_angles)

            self.logger.log_step(
                step=self._step_count,
                sent_command=q_des,
                motor_positions=self.positions,
                imu_quat=self.imu_quat,
                imu_rpy=self.rpy,
                imu_gyro=self.ang_vel,
                imu_accel=self.accel,
            )

            self.publish_lowcmd_action(q_des)

        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')

    def gait_phase_obs(self):

        time_s = self._step_count * self._control_dt

        phase_left = (time_s * self._GAIT_FREQUENCY_HZ) % 1.0
        phase_right = (phase_left + 0.5) % 1.0

        phases = np.array([phase_left, phase_right], dtype=np.float32)
        phase_angle = 2.0 * np.pi * phases

        sin_part = np.sin(phase_angle)
        cos_part = np.cos(phase_angle)
        
        contact_left = 1.0 if phase_left < self._GAIT_DUTY_CYCLE else 0.0
        contact_right = 1.0 if phase_right < self._GAIT_DUTY_CYCLE else 0.0

        command_tail = np.array([
            contact_left,
            contact_right,
            self._GAIT_FREQUENCY_HZ,
        ], dtype=np.float32)

        return np.concatenate([sin_part, cos_part, command_tail])

    def publish_lowcmd_action(self, action: np.ndarray):
        msg = LowCmd()
        msg.motor_cmd = [MotorCmd() for _ in range(10)]

        for i in range(5):
            msg.motor_cmd[i].position     = float(action[i])
            msg.motor_cmd[i + 5].position = float(action[i + 5])
            msg.motor_cmd[i].kp = 5.0
            msg.motor_cmd[i].kd = 0.35
            msg.motor_cmd[i + 5].kp = 5.0
            msg.motor_cmd[i + 5].kd = 0.35

        self.lowcmd_publisher.publish(msg)

    def shutdown(self):
        # self.device.shutdown()
        self.logger.finalize()
        self.destroy_node()


if __name__ == '__main__':
    rclpy.init()
    node = GaitController(model_path='/workspace/models/params')
        
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.shutdown()
