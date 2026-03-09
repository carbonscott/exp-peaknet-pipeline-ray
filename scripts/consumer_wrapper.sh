#!/bin/bash
#===============================================================================
# Consumer Wrapper — Runs on Ada (GPU) node
#
# Starts Ray head, peaknet-pipeline, and CXI writer in the correct order.
# Called by run-pipeline.sbatch via srun --het-group=1.
#===============================================================================
set -euo pipefail

#===============================================================================
# Parse arguments
#===============================================================================
RUN_DIR=""
GPU_IDS="0,1,2,3"
NUM_ACTORS=4
COMPILE_MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-dir)      RUN_DIR="$2";      shift 2 ;;
        --gpu-ids)      GPU_IDS="$2";      shift 2 ;;
        --num-actors)   NUM_ACTORS="$2";   shift 2 ;;
        --compile-mode) COMPILE_MODE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${RUN_DIR}" ]; then
    echo "ERROR: --run-dir is required"
    exit 1
fi

PROJECT_DIR="/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml"

echo "$(date): Consumer wrapper starting on $(hostname)"
echo "  RUN_DIR:      ${RUN_DIR}"
echo "  GPU_IDS:      ${GPU_IDS}"
echo "  NUM_ACTORS:   ${NUM_ACTORS}"
echo "  COMPILE_MODE: ${COMPILE_MODE:-standard}"

#===============================================================================
# Cleanup handler — propagate signals to children and stop Ray
#===============================================================================
PEAKNET_PID=""
WRITER_PID=""

cleanup() {
    echo ""
    echo "$(date): Consumer cleanup starting..."

    if [ -n "${WRITER_PID}" ] && kill -0 "${WRITER_PID}" 2>/dev/null; then
        echo "$(date): Stopping CXI writer (PID ${WRITER_PID})..."
        kill -TERM "${WRITER_PID}" 2>/dev/null
        wait "${WRITER_PID}" 2>/dev/null || true
    fi

    if [ -n "${PEAKNET_PID}" ] && kill -0 "${PEAKNET_PID}" 2>/dev/null; then
        echo "$(date): Stopping peaknet-pipeline (PID ${PEAKNET_PID})..."
        kill -TERM "${PEAKNET_PID}" 2>/dev/null
        wait "${PEAKNET_PID}" 2>/dev/null || true
    fi

    echo "$(date): Stopping Ray..."
    ray stop 2>/dev/null || true

    echo "$(date): Consumer cleanup complete."
}
trap cleanup TERM INT EXIT

#===============================================================================
# Step 1: Activate conda environment
#===============================================================================
echo "$(date): Activating conda environment..."
set +eu
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate /sdf/scratch/users/c/cwang31/miniconda2/pytorch-2.6
set -eu

#===============================================================================
# Step 2: Start Ray head node
#===============================================================================
echo "$(date): Starting Ray head node..."
CUDA_VISIBLE_DEVICES="${GPU_IDS}" ray start --head

# Wait for Ray to be ready
echo "$(date): Waiting for Ray to be ready..."
RETRIES=0
MAX_RETRIES=30
while ! ray status &>/dev/null; do
    RETRIES=$((RETRIES + 1))
    if [ ${RETRIES} -ge ${MAX_RETRIES} ]; then
        echo "ERROR: Ray failed to start after ${MAX_RETRIES} attempts"
        exit 1
    fi
    sleep 1
done
echo "$(date): Ray is ready (took ${RETRIES}s)"

#===============================================================================
# Step 3: Launch peaknet-pipeline in background
#===============================================================================
echo "$(date): Launching peaknet-pipeline..."

COMPILE_FLAG=""
if [ -n "${COMPILE_MODE}" ]; then
    COMPILE_FLAG="--compile-mode ${COMPILE_MODE}"
fi

cd "${PROJECT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" peaknet-pipeline \
    --config "${RUN_DIR}/peaknet.yaml" \
    --max-actors "${NUM_ACTORS}" \
    --verbose \
    ${COMPILE_FLAG} &
PEAKNET_PID=$!

echo "$(date): peaknet-pipeline launched (PID ${PEAKNET_PID})"

#===============================================================================
# Step 4: Wait for pipeline to initialize before starting CXI writer
#===============================================================================
echo "$(date): Waiting 10s for pipeline to initialize..."
sleep 10

# Verify peaknet-pipeline is still running
if ! kill -0 "${PEAKNET_PID}" 2>/dev/null; then
    echo "ERROR: peaknet-pipeline exited prematurely"
    wait "${PEAKNET_PID}"
    exit 1
fi

#===============================================================================
# Step 5: Launch CXI writer in background
#===============================================================================
echo "$(date): Launching CXI writer..."

cxi-writer \
    --config "${RUN_DIR}/cxi_writer.yaml" \
    --batches-per-file 10 &
WRITER_PID=$!

echo "$(date): CXI writer launched (PID ${WRITER_PID})"

#===============================================================================
# Wait for all background processes
#===============================================================================
echo "$(date): Consumer pipeline running. Waiting for processes..."
wait "${PEAKNET_PID}" "${WRITER_PID}"
echo "$(date): Consumer processes finished."
