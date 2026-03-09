#!/bin/bash
#===============================================================================
# Heterogeneous SLURM Job: PeakNet Inference Pipeline (Demo)
#
# Milano (CPU) producers --> Ada (GPU) consumer in Apptainer container
#
# Prerequisites:
#   1. Build container: apptainer build --fakeroot consumer.sif consumer.def
#   2. Install lclstreamer: cd /path/to/lclstreamer && pixi install
#
# Usage:
#   cd demo/
#   mkdir -p runs    # required: SLURM needs output dir before job starts
#   sbatch run-demo.sbatch
#   NUM_PRODUCERS=16 sbatch run-demo.sbatch
#   COMPILE_MODE=reduce-overhead sbatch run-demo.sbatch
#===============================================================================

# === Job-wide settings ===
#SBATCH --job-name=peaknet-demo
#SBATCH --account=lcls:prjdat21
#SBATCH --time=24:00:00
#SBATCH --output=runs/peaknet-%j.out
#SBATCH --error=runs/peaknet-%j.err

# === Component 0: Milano (CPU) — producers ===
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --exclusive

#SBATCH hetjob

# === Component 1: Ada (GPU) — consumer ===
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --gpus=10
#SBATCH --exclusive

#===============================================================================
# Parameters (override via environment variables)
#===============================================================================
NUM_PRODUCERS=${NUM_PRODUCERS:-8}
NUM_ACTORS=${NUM_ACTORS:-4}
GPU_IDS=${GPU_IDS:-"0,1,2,3"}
COMPILE_MODE=${COMPILE_MODE:-""}
EXPERIMENT=${EXPERIMENT:-"mfxl1038923"}
RUN=${RUN:-"r0278"}

# Path to lclstreamer installation (must have pixi installed)
LCLSTREAMER_DIR=${LCLSTREAMER_DIR:-"/sdf/data/lcls/ds/prj/prjcwang31/results/software/lclstreamer"}

#===============================================================================
# Derived paths
#===============================================================================
# SLURM copies the script to a spool dir, so BASH_SOURCE won't resolve correctly.
# Use SLURM_SUBMIT_DIR (the directory where sbatch was invoked) instead.
DEMO_DIR="${SLURM_SUBMIT_DIR}"
TIMESTAMP=$(date +%Y-%m%d-%H%M)
RUN_DIR="${DEMO_DIR}/runs/${TIMESTAMP}-${SLURM_JOB_ID}"
SIF="${DEMO_DIR}/consumer.sif"

CONSUMER_NODE=${SLURM_JOB_NODELIST_HET_GROUP_1}
PRODUCER_NODE=${SLURM_JOB_NODELIST_HET_GROUP_0}

if [ "${COMPILE_MODE}" = "reduce-overhead" ]; then
    COMPILED_TAG="yes"
else
    COMPILED_TAG="no"
fi

#===============================================================================
# Validate prerequisites
#===============================================================================
if [ ! -f "${SIF}" ]; then
    echo "ERROR: Container not found: ${SIF}"
    echo "Build it first: apptainer build --fakeroot consumer.sif consumer.def"
    exit 1
fi

#===============================================================================
# Create run directory
#===============================================================================
mkdir -p "${RUN_DIR}"

echo "============================================"
echo "PeakNet Pipeline Demo"
echo "============================================"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Start Time:      $(date)"
echo "Run Dir:         ${RUN_DIR}"
echo ""
echo "Experiment:      ${EXPERIMENT}"
echo "Run:             ${RUN}"
echo ""
echo "=== Component 0: Milano (CPU) ==="
echo "Node:            ${PRODUCER_NODE}"
echo "Num Producers:   ${NUM_PRODUCERS}"
echo ""
echo "=== Component 1: Ada (GPU) ==="
echo "Node:            ${CONSUMER_NODE}"
echo "Container:       ${SIF}"
echo "Num Actors:      ${NUM_ACTORS}"
echo "GPU IDs:         ${GPU_IDS}"
echo "Compile Mode:    ${COMPILE_MODE:-standard}"
echo "============================================"

#===============================================================================
# Copy and patch config files
#===============================================================================
echo "$(date): Copying and patching config files..."

cp "${DEMO_DIR}/configs/peaknet.yaml"      "${RUN_DIR}/peaknet.yaml"
cp "${DEMO_DIR}/configs/cxi_writer.yaml"   "${RUN_DIR}/cxi_writer.yaml"
cp "${DEMO_DIR}/configs/lclstreamer.yaml"  "${RUN_DIR}/lclstreamer.yaml"

# Replace CONSUMER_NODE placeholder with actual node name
sed -i "s/CONSUMER_NODE/${CONSUMER_NODE}/g" "${RUN_DIR}/peaknet.yaml"
sed -i "s/CONSUMER_NODE/${CONSUMER_NODE}/g" "${RUN_DIR}/lclstreamer.yaml"
sed -i "s/CONSUMER_NODE/${CONSUMER_NODE}/g" "${RUN_DIR}/cxi_writer.yaml"

# Set CXI output directory for this run
sed -i "s|output_dir: \"./cxi_output\"|output_dir: \"${RUN_DIR}/cxi_output\"|" "${RUN_DIR}/cxi_writer.yaml"

echo "$(date): Patched configs with consumer node: ${CONSUMER_NODE}"

#===============================================================================
# Write run-info.txt
#===============================================================================
cat > "${RUN_DIR}/run-info.txt" << EOF
Run Info
========
Job ID:          ${SLURM_JOB_ID}
Timestamp:       ${TIMESTAMP}
Start Time:      $(date)

Experiment:      ${EXPERIMENT}
Run:             ${RUN}

Nodes:
  Consumer (Ada):    ${CONSUMER_NODE}
  Producer (Milano): ${PRODUCER_NODE}

Parameters:
  NUM_PRODUCERS:   ${NUM_PRODUCERS}
  NUM_ACTORS:      ${NUM_ACTORS}
  GPU_IDS:         ${GPU_IDS}
  COMPILE_MODE:    ${COMPILE_MODE:-standard}

Container:       ${SIF}
EOF

#===============================================================================
# Cleanup handler
#===============================================================================
CONSUMER_PID=""
PRODUCER_PID=""

cleanup() {
    echo ""
    echo "$(date): Received termination signal, cleaning up..."

    if [ -n "${PRODUCER_PID}" ] && kill -0 "${PRODUCER_PID}" 2>/dev/null; then
        echo "$(date): Stopping producer (PID ${PRODUCER_PID})..."
        kill -TERM "${PRODUCER_PID}" 2>/dev/null
        wait "${PRODUCER_PID}" 2>/dev/null
    fi

    if [ -n "${CONSUMER_PID}" ] && kill -0 "${CONSUMER_PID}" 2>/dev/null; then
        echo "$(date): Stopping consumer (PID ${CONSUMER_PID})..."
        kill -TERM "${CONSUMER_PID}" 2>/dev/null
        wait "${CONSUMER_PID}" 2>/dev/null
    fi

    echo "End Time: $(date)" >> "${RUN_DIR}/run-info.txt"
    echo "$(date): Cleanup complete."
    exit 0
}
trap cleanup TERM INT

#===============================================================================
# Launch consumer on Ada (het-group 1) inside Apptainer
#===============================================================================
echo "$(date): Launching consumer on ${CONSUMER_NODE} (containerized)..."

# Run everything inside a SINGLE apptainer exec to avoid squashfuse mount teardown
# between invocations (Ray daemon processes lose their filesystem if the mount dies).
srun --het-group=1 --ntasks=1 --overlap \
    env APPTAINERENV_RAY_TMPDIR=/tmp APPTAINERENV_TMPDIR=/tmp \
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    apptainer exec --nv --bind /sdf "${SIF}" \
    bash -c '
set -euo pipefail

RUN_DIR="'"${RUN_DIR}"'"
NUM_ACTORS="'"${NUM_ACTORS}"'"
COMPILE_MODE="'"${COMPILE_MODE}"'"

echo "$(date): Consumer starting on $(hostname)"

# --- Cleanup handler ---
PEAKNET_PID=""
WRITER_PID=""

cleanup_consumer() {
    echo ""
    echo "$(date): Consumer cleanup starting..."
    if [ -n "${WRITER_PID}" ] && kill -0 "${WRITER_PID}" 2>/dev/null; then
        kill -TERM "${WRITER_PID}" 2>/dev/null
        wait "${WRITER_PID}" 2>/dev/null || true
    fi
    if [ -n "${PEAKNET_PID}" ] && kill -0 "${PEAKNET_PID}" 2>/dev/null; then
        kill -TERM "${PEAKNET_PID}" 2>/dev/null
        wait "${PEAKNET_PID}" 2>/dev/null || true
    fi
    echo "$(date): Stopping Ray..."
    ray stop 2>/dev/null || true
    echo "$(date): Consumer cleanup complete."
}
trap cleanup_consumer TERM INT EXIT

# --- Start Ray head node ---
echo "$(date): Starting Ray head node..."
ray start --head

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

# --- Launch peaknet-pipeline ---
echo "$(date): Launching peaknet-pipeline..."

COMPILE_FLAG=""
if [ -n "${COMPILE_MODE}" ]; then
    COMPILE_FLAG="--compile-mode ${COMPILE_MODE}"
fi

peaknet-pipeline \
    --config "${RUN_DIR}/peaknet.yaml" \
    --max-actors "${NUM_ACTORS}" \
    --verbose \
    ${COMPILE_FLAG} &
PEAKNET_PID=$!

echo "$(date): peaknet-pipeline launched (PID ${PEAKNET_PID})"

# --- Wait for pipeline to initialize ---
echo "$(date): Waiting 10s for pipeline to initialize..."
sleep 10

if ! kill -0 "${PEAKNET_PID}" 2>/dev/null; then
    echo "ERROR: peaknet-pipeline exited prematurely"
    wait "${PEAKNET_PID}"
    exit 1
fi

# --- Launch cxi-writer ---
echo "$(date): Launching cxi-writer..."

cxi-writer \
    --config "${RUN_DIR}/cxi_writer.yaml" \
    --batches-per-file 10 &
WRITER_PID=$!

echo "$(date): cxi-writer launched (PID ${WRITER_PID})"

# --- Wait for both ---
echo "$(date): Consumer pipeline running. Waiting for processes..."
wait "${PEAKNET_PID}" "${WRITER_PID}"
echo "$(date): Consumer processes finished."
' &
CONSUMER_PID=$!

echo "$(date): Consumer launched (PID ${CONSUMER_PID})"

#===============================================================================
# Wait for consumer to be ready
#===============================================================================
echo "$(date): Waiting 30s for consumer to initialize..."
sleep 30

#===============================================================================
# Launch producer on Milano (het-group 0) — native, no container
# NOTE: No srun needed — sbatch script already runs on het-group 0 (Milano).
# Calling mpirun inside srun can cause MPI launcher conflicts.
#===============================================================================
echo "$(date): Launching producers on ${PRODUCER_NODE} (${NUM_PRODUCERS} ranks)..."

(
set -uo pipefail

RETRY_INTERVAL=5
MAX_RETRIES=60

echo "$(date): Producer wrapper starting on $(hostname)"
echo "  RUN_DIR:         ${RUN_DIR}"
echo "  NUM_PRODUCERS:   ${NUM_PRODUCERS}"
echo "  LCLSTREAMER_DIR: ${LCLSTREAMER_DIR}"

cleanup() {
    echo "$(date): Producer cleanup — terminating children..."
    kill -- -$$ 2>/dev/null || true
}
trap cleanup TERM INT

cd "${LCLSTREAMER_DIR}"

ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "$(date): Launching lclstreamer (attempt ${ATTEMPT}/${MAX_RETRIES})..."

    pixi run --environment psana1 \
        mpirun -n "${NUM_PRODUCERS}" \
        lclstreamer --config "${RUN_DIR}/lclstreamer.yaml"
    EXIT_CODE=$?

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "$(date): Producers finished successfully."
        break
    fi

    if [ ${ATTEMPT} -ge ${MAX_RETRIES} ]; then
        echo "ERROR: lclstreamer failed after ${MAX_RETRIES} attempts."
        exit 1
    fi

    echo "$(date): lclstreamer exited with code ${EXIT_CODE}, retrying in ${RETRY_INTERVAL}s..."
    sleep ${RETRY_INTERVAL}
done

echo "$(date): Producer wrapper finished."
) &
PRODUCER_PID=$!

echo "$(date): Producers launched (PID ${PRODUCER_PID})"

#===============================================================================
# Wait for both to finish
#===============================================================================
echo "$(date): Pipeline running. Waiting for completion..."
wait "${CONSUMER_PID}" "${PRODUCER_PID}"

echo ""
echo "$(date): Pipeline finished."
echo "End Time: $(date)" >> "${RUN_DIR}/run-info.txt"
echo "============================================"
echo "CXI output: ${RUN_DIR}/cxi_output/"
echo "============================================"
