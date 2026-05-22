#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE_NAME="sim2sim:mujoco"

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Image ${IMAGE_NAME} not found. Building..."
  docker build -t "${IMAGE_NAME}" "${ROOT_DIR}"
fi

if command -v xhost >/dev/null 2>&1; then
  xhost +local:docker >/dev/null 2>&1 || true
fi

docker run -it --network host --ipc host \
  -e DISPLAY="${DISPLAY:-}" \
  -e CYCLONEDDS_URI=file:///tmp/cyclone.xml \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
  -v /tmp/cyclone.xml:/tmp/cyclone.xml \
  -v "${ROOT_DIR}/sim_mujoco.py:/workspace/sim_mujoco.py" \
  -v "${ROOT_DIR}/xml:/workspace/xml" \
  -v "${ROOT_DIR}/meshes:/workspace/meshes" \
  -v "${ROOT_DIR}/tinker_msgs:/workspace/tinker_msgs" \
  --device /dev/dri --group-add video \
  "${IMAGE_NAME}" "$@"
