#!/usr/bin/env python3

import argparse
import rclpy
from rclpy.executors import MultiThreadedExecutor
import src
from src.gait_controller import GaitController

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--device",
                    type=str,
                    default="keyboard",
                    choices=['keyboard', 'gamepad', 'fixed_forward'],
                    help="set input device: 'keyboard'/'gamepad'/'fixed_forward' (default is keyboard)")
    ap.add_argument("-p", "--path",
                    type=str,
                    default=None,
                    help="define inference model path")
    ap.add_argument("-g", "--gait",
                    type=str,
                    default="legacy",
                    choices=["legacy", "bd_lip", "lip_play"],
                    help="select gait adapter: legacy or bd_lip")
    args = ap.parse_args()

    model_path = args.path
    if model_path is None:
        model_path = "./src/model/BipeD" if args.gait in {"bd_lip", "lip_play"} else "./src/model/tinker"

    rclpy.init()

    try:
        controller = GaitController(
            # adapter_type = args.object,
            device_type = args.device,
            model_path = model_path,
            gait_mode = args.gait
        )

        executor = MultiThreadedExecutor()
        executor.add_node(controller)

        executor.spin()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')

    finally:
        if 'controller' in locals():
            controller.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
