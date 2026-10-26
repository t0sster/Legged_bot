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

# kind -> CSV columns (without time/step)
COLUMNS = {
    'motor_positions': [f'pos_{name}' for name in JOINT_NAMES],
    'motor_velocities': [f'vel_{name}' for name in JOINT_NAMES],
    'motor_torques': [f'torque_{name}' for name in JOINT_NAMES],
    'imu': ['quat_w', 'quat_x', 'quat_y', 'quat_z',
            'roll', 'pitch', 'yaw',
            'gyro_x', 'gyro_y', 'gyro_z',
            'accel_x', 'accel_y', 'accel_z'],
}

# How each source is drawn when several are overlaid.
LINE_STYLES = {'real': '-', 'sim': '--'}


def state_from_msg(msg):
    """LowState -> {kind: values} for RunLogger.log_step."""
    imu = msg.imu_state
    return {
        'motor_positions': [m.position for m in msg.motor_state],
        'motor_velocities': [m.velocity for m in msg.motor_state],
        'motor_torques': [m.torque for m in msg.motor_state],
        'imu': list(imu.quaternion) + list(imu.rpy) + list(imu.gyroscope) + list(imu.accelerometer),
    }


class RunLogger:
    """Writes one run to logs/<prefix>_<timestamp>/.

    sources: the first one is the state the controller runs on (files without a
    prefix), the others are logged alongside it at the same steps (files prefixed
    with '<source>_'), e.g. ['real', 'sim'] for a real run mirrored in the simulator.
    """

    def __init__(self, prefix, sources, base_dir='/workspace/logs'):
        run_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = os.path.join(base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self._start_time = time.time()
        self.sources = sources

        # In-memory history, used only to draw the debug plots at the end.
        self.time_history = []
        self.sent_command_history = []
        self.history = {source: {kind: [] for kind in COLUMNS} for source in sources}

        self._files = []
        self._commands_writer = self._open_csv(
            'commands.csv', [f'cmd_{name}' for name in JOINT_NAMES])
        self._writers = {
            source: {
                kind: self._open_csv(
                    f'{kind}.csv' if source == sources[0] else f'{source}_{kind}.csv', columns)
                for kind, columns in COLUMNS.items()
            }
            for source in sources
        }

    def _open_csv(self, filename, columns):
        file = open(os.path.join(self.run_dir, filename), 'w', newline='')
        self._files.append(file)
        writer = csv.writer(file)
        writer.writerow(['time', 'step'] + columns)
        return writer

    def log_step(self, step, sent_command, states):
        """states: {source: state_from_msg(...)} for every source."""
        t = time.time() - self._start_time

        self._commands_writer.writerow([t, step] + list(sent_command))
        self.time_history.append(t)
        self.sent_command_history.append(list(sent_command))

        for source in self.sources:
            for kind in COLUMNS:
                row = list(states[source][kind])
                self._writers[source][kind].writerow([t, step] + row)
                self.history[source][kind].append(row)

        for file in self._files:
            file.flush()

    def finalize(self):
        for file in self._files:
            file.close()

        if not self.time_history:
            return

        self._plot_joints('motor_positions', 'rad', 'joint_tracking.png', with_command=True)
        self._plot_joints('motor_velocities', 'rad/s', 'motor_velocities.png')
        self._plot_joints('motor_torques', 'N*m', 'motor_torques.png')
        self._plot_imu(['roll', 'pitch', 'yaw'], 'rad', 'IMU orientation (rpy)',
                       'imu_orientation.png')
        self._plot_imu(['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z'],
                       None, 'IMU gyroscope (rad/s) and accelerometer (m/s^2)', 'imu_rates.png')

    def _plot_joints(self, kind, unit, filename, with_command=False):
        fig, axes = plt.subplots(5, 2, figsize=(12, 14), sharex=True)
        for joint_index, joint_name in enumerate(JOINT_NAMES):
            ax = axes[joint_index % 5, joint_index // 5]
            if with_command:
                ax.plot(self.time_history,
                        [row[joint_index] for row in self.sent_command_history],
                        color='black', linewidth=0.8, label='commanded')
            for source in self.sources:
                ax.plot(self.time_history,
                        [row[joint_index] for row in self.history[source][kind]],
                        LINE_STYLES[source], label=source)
            ax.set_title(joint_name)
            ax.set_ylabel(unit)
        axes[0, 0].legend()
        for ax in axes[-1]:
            ax.set_xlabel('time, s')
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, filename))
        plt.close(fig)

    def _plot_imu(self, columns, unit, title, filename):
        fig, axes = plt.subplots(len(columns), 1, figsize=(10, 2.2 * len(columns)), sharex=True)
        for ax, column in zip(axes, columns):
            index = COLUMNS['imu'].index(column)
            for source in self.sources:
                ax.plot(self.time_history,
                        [row[index] for row in self.history[source]['imu']],
                        LINE_STYLES[source], label=source)
            ax.set_ylabel(column if unit is None else f'{column}, {unit}')
        axes[0].set_title(title)
        axes[0].legend()
        axes[-1].set_xlabel('time, s')
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, filename))
        plt.close(fig)
