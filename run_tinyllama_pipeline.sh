#!/usr/bin/env bash
#
# run_tinyllama_pipeline.sh  --  launch distributed TinyLlama on Jetsons
#
# Usage (run on each Jetson in separate terminals):
#   ./run_tinyllama_pipeline.sh worker 192.168.3.15 29500    # on Jetson-B
#   ./run_tinyllama_pipeline.sh master 192.168.3.15 29500    # on Jetson-A
#
# Positional args:
#   role        master | worker
#   MASTER_ADDR IP of rank-0 (master)
#   MASTER_PORT TCP port (e.g., 29500)

set -e

ROLE=${1:-master}
MASTER_ADDR=${2:-192.168.3.15}
MASTER_PORT=${3:-29500}

# Adjustable paths & settings -------------------------------------------------
NIC="wlP1p1s0"                                 # your Ethernet iface inside container
IMAGE="dustynv/pytorch:2.6-r36.4.0-cu128-24.04"
SCRIPT="jetson_tinyllama_pipeline.py"      # python script we wrote earlier
MOUNT_SRC="$HOME/llm_demo"                 # folder with the script
HF_CACHE="$HOME/hf_cache"                  # shared HF cache directory
K_LAYERS=6                                 # #layers on rank-0
PROMPT="TinyLlama on Jetson is"
# -----------------------------------------------------------------------------


if [[ ! -f "$MOUNT_SRC/$SCRIPT" ]]; then
    echo "❌  $SCRIPT not found in $MOUNT_SRC"; exit 1
fi

RANK=0
if [[ "$ROLE" == "worker" ]]; then RANK=1; fi

docker run --runtime nvidia --rm -it \
  --name tinyllama_$ROLE \
  --network host --ipc host --shm-size 8g \
  -e GLOO_SOCKET_IFNAME=${NIC} \
  -e NCCL_P2P_DISABLE=1 \
  -v "${MOUNT_SRC}":/workspace/demo \
  -v "${HF_CACHE}":/root/.cache/huggingface \
  "${IMAGE}" \
  bash -c "
    pip install -q transformers==4.42.0 sentencepiece accelerate \
                 --extra-index-url https://pypi.jetson-ai-lab.dev/root/pypi ;
    python /workspace/demo/${SCRIPT} \
       --rank ${RANK} --world_size 2 \
       --k ${K_LAYERS} \
       --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
       --prompt \"${PROMPT}\" "

