#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE_NAME="gait-controller:jazzy"

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Image ${IMAGE_NAME} not found. Building..."
  docker build -t "${IMAGE_NAME}" "${ROOT_DIR}"
fi

if command -v xhost >/dev/null 2>&1; then
  xhost +local:docker >/dev/null 2>&1 || true
fi

docker run -it --net host --ipc host --gpus all \
  -e DISPLAY="${DISPLAY:-}" \
  -e CYCLONEDDS_URI=file:///tmp/cyclone.xml \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp/cyclone.xml:/tmp/cyclone.xml \
  -v "${ROOT_DIR}/src:/workspace/src" \
  -v "${ROOT_DIR}/inference_controller_setup.py:/workspace/inference_controller_setup.py" \
  "${IMAGE_NAME}" "$@"
