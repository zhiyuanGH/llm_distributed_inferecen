#!/usr/bin/env bash
#
# run_qwen3_pipeline.sh  --  launch distributed Qwen3-4B on Jetsons
#
# Usage (run on each Jetson in separate terminals):
#   ./run_qwen3_pipeline.sh worker 192.168.3.19 29500    # on Jetson-B (192.168.3.13)
#   ./run_qwen3_pipeline.sh master 192.168.3.19 29500    # on Jetson-A (192.168.3.19)
#
# Positional args:
#   role        master | worker
#   MASTER_ADDR IP of rank-0 (master)
#   MASTER_PORT TCP port (e.g., 29500)

set -e

ROLE=${1:-master}
MASTER_ADDR=${2:-192.168.3.19}
MASTER_PORT=${3:-29500}

# Adjustable paths & settings -------------------------------------------------
NIC="wlP1p1s0"                                 # your Ethernet iface inside container
IMAGE="dustynv/pytorch:2.6-r36.4.0-cu128-24.04"
SCRIPT="jetson_tinyllama_pipeline.py"      # python script we wrote earlier
MOUNT_SRC="$(pwd)"                 # folder with the script
HF_CACHE="$HOME/.cache/huggingface"        # shared HF cache directory
K_LAYERS=16                                # #layers on rank-0 (16 out of 32 total for Qwen3-4B)
PROMPT="Qwen3 on Jetson is"
MODEL_NAME="Qwen/Qwen3-4B"
# -----------------------------------------------------------------------------


if [[ ! -f "$MOUNT_SRC/$SCRIPT" ]]; then
    echo "❌  $SCRIPT not found in $MOUNT_SRC"; exit 1
fi

RANK=0
if [[ "$ROLE" == "worker" ]]; then RANK=1; fi

echo "🚀 Starting $ROLE (rank $RANK) with model $MODEL_NAME"
echo "   Master: $MASTER_ADDR:$MASTER_PORT"
echo "   Layers split: $K_LAYERS on rank-0, $((32-K_LAYERS)) on rank-1"

docker run --runtime nvidia --rm -it \
  --name qwen3_$ROLE \
  --network host --ipc host --shm-size 8g \
  -e GLOO_SOCKET_IFNAME=${NIC} \
  -e NCCL_P2P_DISABLE=1 \
  -v "${MOUNT_SRC}":/workspace/demo \
  -v "${HF_CACHE}":/root/.cache/huggingface \
  "${IMAGE}" \
  bash -c "
    pip install -q transformers==4.53.1 sentencepiece accelerate \
                 --extra-index-url https://pypi.jetson-ai-lab.dev/root/pypi ;
    python /workspace/demo/${SCRIPT} \
       --rank ${RANK} --world_size 2 \
       --k ${K_LAYERS} \
       --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
       --model_name ${MODEL_NAME} \
       --prompt \"${PROMPT}\" "

