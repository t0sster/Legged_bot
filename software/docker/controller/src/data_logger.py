import os
import csv
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


JOINT_NAMES = [
    'joint_l_yaw', 'joint_l_roll', 'joint_l_pitch', 'joint_l_knee', 'joint_l_ankle',
    'joint_r_yaw', 'joint_r_roll', 'joint_r_pitch', 'joint_r_knee', 'joint_r_ankle',
]


class RunLogger:

    def __init__(self, base_dir='/workspace/logs'):
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self._start_time = time.time()

        # In-memory history, used only to draw the debug plots at the end.
        self.time_history = []
        self.sent_command_history = []
        self.motor_position_history = []
        self.imu_rpy_history = []
        self.imu_gyro_history = []
        self.imu_accel_history = []

        self._commands_file = open(os.path.join(self.run_dir, 'commands.csv'), 'w', newline='')
        self._positions_file = open(os.path.join(self.run_dir, 'motor_positions.csv'), 'w', newline='')
        self._imu_file = open(os.path.join(self.run_dir, 'imu.csv'), 'w', newline='')

        self._commands_writer = csv.writer(self._commands_file)
        self._positions_writer = csv.writer(self._positions_file)
        self._imu_writer = csv.writer(self._imu_file)

        self._commands_writer.writerow(['time', 'step'] + [f'cmd_{name}' for name in JOINT_NAMES])
        self._positions_writer.writerow(['time', 'step'] + [f'pos_{name}' for name in JOINT_NAMES])
        self._imu_writer.writerow([
            'time', 'step',
            'quat_w', 'quat_x', 'quat_y', 'quat_z',
            'roll', 'pitch', 'yaw',
            'gyro_x', 'gyro_y', 'gyro_z',
            'accel_x', 'accel_y', 'accel_z',
        ])

    def log_step(self, step, sent_command, motor_positions, imu_quat, imu_rpy, imu_gyro, imu_accel):
        t = time.time() - self._start_time

        self._commands_writer.writerow([t, step] + list(sent_command))
        self._commands_file.flush()

        self._positions_writer.writerow([t, step] + list(motor_positions))
        self._positions_file.flush()

        self._imu_writer.writerow(
            [t, step] + list(imu_quat) + list(imu_rpy) + list(imu_gyro) + list(imu_accel))
        self._imu_file.flush()

        self.time_history.append(t)
        self.sent_command_history.append(list(sent_command))
        self.motor_position_history.append(list(motor_positions))
        self.imu_rpy_history.append(list(imu_rpy))
        self.imu_gyro_history.append(list(imu_gyro))
        self.imu_accel_history.append(list(imu_accel))

    def finalize(self):
        self._commands_file.close()
        self._positions_file.close()
        self._imu_file.close()

        if not self.time_history:
            return

        self._plot_joint_tracking()
        self._plot_imu_orientation()
        self._plot_imu_rates()

    def _plot_joint_tracking(self):
        fig, axes = plt.subplots(5, 2, figsize=(12, 14), sharex=True)
        for joint_index, joint_name in enumerate(JOINT_NAMES):
            ax = axes[joint_index % 5, joint_index // 5]
            commanded = [row[joint_index] for row in self.sent_command_history]
            actual = [row[joint_index] for row in self.motor_position_history]
            ax.plot(self.time_history, commanded, label='commanded')
            ax.plot(self.time_history, actual, label='actual')
            ax.set_title(joint_name)
            ax.set_ylabel('rad')
        axes[0, 0].legend()
        for ax in axes[-1]:
            ax.set_xlabel('time, s')
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, 'joint_tracking.png'))
        plt.close(fig)

    def _plot_imu_orientation(self):
        roll = [row[0] for row in self.imu_rpy_history]
        pitch = [row[1] for row in self.imu_rpy_history]
        yaw = [row[2] for row in self.imu_rpy_history]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.time_history, roll, label='roll')
        ax.plot(self.time_history, pitch, label='pitch')
        ax.plot(self.time_history, yaw, label='yaw')
        ax.set_xlabel('time, s')
        ax.set_ylabel('rad')
        ax.set_title('IMU orientation (rpy)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, 'imu_orientation.png'))
        plt.close(fig)

    def _plot_imu_rates(self):
        fig, (ax_gyro, ax_accel) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for axis_index, axis_name in enumerate(('x', 'y', 'z')):
            gyro_axis = [row[axis_index] for row in self.imu_gyro_history]
            accel_axis = [row[axis_index] for row in self.imu_accel_history]
            ax_gyro.plot(self.time_history, gyro_axis, label=axis_name)
            ax_accel.plot(self.time_history, accel_axis, label=axis_name)
        ax_gyro.set_title('IMU gyroscope')
        ax_gyro.set_ylabel('rad/s')
        ax_gyro.legend()
        ax_accel.set_title('IMU accelerometer')
        ax_accel.set_ylabel('m/s^2')
        ax_accel.set_xlabel('time, s')
        ax_accel.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, 'imu_rates.png'))
        plt.close(fig)
