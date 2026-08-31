import mujoco
import mujoco.viewer
import os
import time
import rclpy
import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from tinker_msgs.msg import LowState, LowCmd, MotorCmd
from rosgraph_msgs.msg import Clock # Для передачи шага моделирования в инференс

class MujocoSim(Node):
    def __init__(self, xml_path):
        super().__init__("mujoco_sim")

        self.rpy = np.zeros(3)
        self.imu_quat = np.zeros(4)
        self.ang_vel = np.zeros(3)
        self.accel = np.zeros(3)
        self.commands = np.zeros(3)
        self.positions = np.zeros(10)
        self.velocities = np.zeros(10)

        self.actions = np.zeros(10)
        self.init_ctrl = np.array([0.0, 0.0, 0.45, 0.9, 0.45, 0.0, 0.0, -0.45, -0.9, -0.45])
        self.ctrl = self.init_ctrl.copy()
        # Base freejoint height that puts the feet on the floor for init_ctrl's
        # crouched pose (feet sit ~0.322m below the base at qpos[2]=0, floor at z=-0.40).
        self.init_base_height = -0.078
        self._last_time = 0.0

        self.IS_ACTIONS = False

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        # self._apply_training_solver_settings(self.model)
        self.data = mujoco.MjData(self.model)

        self.ctrl_range = self.model.actuator_ctrlrange.copy()

        self.cmd_subscriber = self.create_subscription(
            LowCmd,
            '/low_level_cmd',
            self.cmd_callback,
            10
        )

        self.state_publisher = self.create_publisher(
            LowState,
            '/low_level_state',
            10
        )

        # В инференесе не создаётся листенер, он автоматически подписывается на симуляционное время
        self.clock_publisher = self.create_publisher(Clock, '/clock', 10)


    def cmd_callback(self, msg: LowCmd):
        cmd = [msg.motor_cmd[i].position for i in range(10)]
        for i in range(5):
            self.actions[i]     = cmd[i]
            self.actions[i + 5] = cmd[i + 5]
        self.IS_ACTIONS = True


    def publish_state(self):
        msg = LowState()
        sim_time = float(self.data.time)
        msg.timestamp_state.sec = int(sim_time)
        msg.timestamp_state.nanosec = int((sim_time % 1.0) * 1e9)
        msg.imu_state.gyroscope = self.ang_vel
        msg.imu_state.accelerometer = self.accel
        msg.imu_state.rpy = self.rpy
        msg.imu_state.quaternion = self.imu_quat

        for i in range(10):
            msg.motor_state[i].position = float(self.positions[i])
            msg.motor_state[i].velocity = float(self.velocities[i])

        self.state_publisher.publish(msg)


    def apply_init_pose(self):
        self.data.qpos[2] = self.init_base_height
        self.data.qpos[7:17] = self.init_ctrl.copy()
        self.data.ctrl[:] = self.init_ctrl.copy()
        self.ctrl = self.init_ctrl.copy()
        self.actions = self.init_ctrl.copy()
        self.IS_ACTIONS = False
        mujoco.mj_forward(self.model, self.data)


    def control_loop(self):
        try:
            # The viewer's own Reset/Backspace resets data directly (outside this
            # loop), which snaps sim time back to 0 and drops qpos/ctrl to the
            # model's all-zero defaults. Catch that here and re-apply the crouched
            # init pose instead of leaving the robot in its flat zero pose.
            if self.data.time < self._last_time:
                self.apply_init_pose()

            if not self.IS_ACTIONS:
                # Manual mode: viewer sliders drive data.ctrl directly — don't overwrite.
                # Sync self.ctrl so there's no jump when policy takes over.
                self.ctrl = self.data.ctrl.copy()
            else:
                self.ctrl = np.clip(self.actions, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
                self.data.ctrl[:] = self.ctrl

            if self.data.qpos[2] < -0.35:
                mujoco.mj_resetData(self.model, self.data)
                self.apply_init_pose()
            mujoco.mj_step(self.model, self.data)
            self._last_time = self.data.time

            # Observations
            self.imu_quat = self.data.sensor('orientation').data.copy()  # [w, x, y, z]
            w, x, y, z = self.imu_quat
            self.ang_vel = self.data.sensor('angular-velocity').data.copy()
            self.accel   = self.data.sensor('linear-acceleration').data.copy()

            # scipy wants the quaternion as [x, y, z, w], the IMU sensor gives [w, x, y, z].
            self.rpy = R.from_quat([x, y, z, w]).as_euler('xyz', degrees=False).astype(np.float32)
       
            self.positions = self.data.qpos[7:17]
            self.velocities = self.data.qvel[6:16]
            
            self.publish_state()
            self.publish_clock()

        except Exception as e:
            print(f'sim2sim control loop error: {e}')


    def publish_clock(self):
        sim_time = float(self.data.time)
        msg = Clock()
        msg.clock.sec = int(sim_time)
        msg.clock.nanosec = int((sim_time % 1.0) * 1e9)
        self.clock_publisher.publish(msg)

    def shutdown(self):
        self.destroy_node()


if __name__ == "__main__":
    rclpy.init()
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "xml", "world.xml")

        # model = mujoco.MjModel.from_xml_path(xml_path)
        # data = mujoco.MjData(model)

        mujoco_sim = MujocoSim(xml_path)
        mujoco_sim.apply_init_pose()

        executor = MultiThreadedExecutor()
        executor.add_node(mujoco_sim)

        sim_dt = float(mujoco_sim.model.opt.timestep)

        print('executor started')
        # with mujoco.viewer.launch(mujoco_sim.model, mujoco_sim.data) as viewer:
        viewer = mujoco.viewer.launch_passive(mujoco_sim.model, mujoco_sim.data)

        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 135

        print('viewer started')

        last_print_time = time.time()

        # sim_dt = float(mujoco_sim.model.opt.timestep)

        while viewer.is_running():
            step_start = time.perf_counter()
            mujoco_sim.control_loop()
            executor.spin_once(timeout_sec=0)
            viewer.sync()
            
            elapsed = time.perf_counter() - step_start
            if elapsed < sim_dt:
                time.sleep(sim_dt - elapsed)
                
            # time.sleep(0.01)
            # current_time = time.time()

            # if current_time - last_print_time >= 3.0:
            #     print("Positions:", mujoco_sim.positions[2])
            #     print("Velocities:", mujoco_sim.velocities[2])
            #     print("Ctrl:", mujoco_sim.data.ctrl[2])
            #     last_print_time = current_time

    except Exception as e:
        print(f"Simulator script error: {e}")

    finally:
        if 'mojoco_sim' in locals():
            mujoco_sim.shutdown()
        rclpy.shutdown()
