#!/bin/bash
#===============================================================================
# Heterogeneous SLURM Job: PeakNet Inference Pipeline
#
# Milano (CPU) producers --> Ada (GPU) consumer pipeline
# Consumer logic is inlined (Ray, peaknet-pipeline, cxi-writer).
#
# Usage:
#   sbatch run-pipeline.sbatch                                    # defaults (mfxl1038923 r0278)
#   NUM_PRODUCERS=32 sbatch run-pipeline.sbatch                   # 32 producers
#   EXPERIMENT=mfxl1047723 RUN=r0266 sbatch run-pipeline.sbatch   # different experiment
#   NUM_PRODUCERS=16 COMPILE_MODE=reduce-overhead sbatch run-pipeline.sbatch
#===============================================================================

# === Job-wide settings ===
#SBATCH --job-name=peaknet-pipeline
#SBATCH --account=lcls:prjdat21
#SBATCH --time=24:00:00
#SBATCH --output=logs/peaknet-%j.out
#SBATCH --error=logs/peaknet-%j.err

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
# Parameters (set via environment variables before sbatch, or use defaults)
#===============================================================================
NUM_PRODUCERS=${NUM_PRODUCERS:-8}
NUM_ACTORS=${NUM_ACTORS:-4}
GPU_IDS=${GPU_IDS:-"0,1,2,3"}
COMPILE_MODE=${COMPILE_MODE:-""}          # "" = standard, "reduce-overhead" = compiled
EXPERIMENT=${EXPERIMENT:-"mfxl1038923"}   # Experiment ID
RUN=${RUN:-"r0278"}                       # Run number (with r prefix)

#===============================================================================
# Derived paths and variables
#===============================================================================
PROJECT_DIR="/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml"
TIMESTAMP=$(date +%Y-%m%d-%H%M)
RUN_DIR="${PROJECT_DIR}/run-config/${TIMESTAMP}"

CONSUMER_NODE=${SLURM_JOB_NODELIST_HET_GROUP_1}
PRODUCER_NODE=${SLURM_JOB_NODELIST_HET_GROUP_0}

if [ "${COMPILE_MODE}" = "reduce-overhead" ]; then
    COMPILED_TAG="yes"
else
    COMPILED_TAG="no"
fi

#===============================================================================
# Create run directory and logs directory
#===============================================================================
mkdir -p "${RUN_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

echo "============================================"
echo "PeakNet Pipeline — Het Job Started"
echo "============================================"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Start Time:      $(date)"
echo "Timestamp:       ${TIMESTAMP}"
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
echo "Num Actors:      ${NUM_ACTORS}"
echo "GPU IDs:         ${GPU_IDS}"
echo "Compile Mode:    ${COMPILE_MODE:-standard}"
echo "============================================"

#===============================================================================
# Copy and patch config files
#===============================================================================
echo "$(date): Copying and patching config files..."

PEAKNET_CONFIG="${PROJECT_DIR}/peaknet-${EXPERIMENT}-${RUN}.yaml"
CXI_WRITER_CONFIG="${PROJECT_DIR}/cxi_writer-${EXPERIMENT}-${RUN}.yaml"
LCLSTREAMER_CONFIG="${PROJECT_DIR}/lclstreamer-${EXPERIMENT}-${RUN}.yaml"

# Verify config files exist
for cfg in "${PEAKNET_CONFIG}" "${CXI_WRITER_CONFIG}" "${LCLSTREAMER_CONFIG}"; do
    if [ ! -f "${cfg}" ]; then
        echo "ERROR: Config file not found: ${cfg}"
        exit 1
    fi
done

cp "${PEAKNET_CONFIG}" "${RUN_DIR}/peaknet.yaml"
cp "${CXI_WRITER_CONFIG}" "${RUN_DIR}/cxi_writer.yaml"
cp "${LCLSTREAMER_CONFIG}" "${RUN_DIR}/lclstreamer.yaml"

# Patch hostnames: replace sdfadaNNN with the actual consumer node
sed -i "s/sdfada[0-9]\{3\}/${CONSUMER_NODE}/g" "${RUN_DIR}/peaknet.yaml"
sed -i "s/sdfada[0-9]\{3\}/${CONSUMER_NODE}/g" "${RUN_DIR}/lclstreamer.yaml"
sed -i "s/sdfada[0-9]\{3\}/${CONSUMER_NODE}/g" "${RUN_DIR}/cxi_writer.yaml"

echo "$(date): Patched configs with consumer node: ${CONSUMER_NODE}"

#===============================================================================
# Write run-info.txt for reproducibility
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
  COMPILED_TAG:    ${COMPILED_TAG}

Config Files:
  peaknet.yaml:    ${RUN_DIR}/peaknet.yaml
  cxi_writer.yaml: ${RUN_DIR}/cxi_writer.yaml
  lclstreamer.yaml:${RUN_DIR}/lclstreamer.yaml
EOF

echo "$(date): Wrote ${RUN_DIR}/run-info.txt"

#===============================================================================
# Cleanup handler
#===============================================================================
CONSUMER_PID=""
PRODUCER_PID=""

cleanup() {
    echo ""
    echo "$(date): Received termination signal, cleaning up..."

    # Kill producer first (stop sending data)
    if [ -n "${PRODUCER_PID}" ] && kill -0 "${PRODUCER_PID}" 2>/dev/null; then
        echo "$(date): Stopping producer (PID ${PRODUCER_PID})..."
        kill -TERM "${PRODUCER_PID}" 2>/dev/null
        wait "${PRODUCER_PID}" 2>/dev/null
    fi

    # Then kill consumer
    if [ -n "${CONSUMER_PID}" ] && kill -0 "${CONSUMER_PID}" 2>/dev/null; then
        echo "$(date): Stopping consumer (PID ${CONSUMER_PID})..."
        kill -TERM "${CONSUMER_PID}" 2>/dev/null
        wait "${CONSUMER_PID}" 2>/dev/null
    fi

    # Record end time
    echo "End Time: $(date)" >> "${RUN_DIR}/run-info.txt"
    echo "$(date): Cleanup complete."
    exit 0
}
trap cleanup TERM INT

#===============================================================================
# Launch consumer on Ada (het-group 1)
#===============================================================================
echo "$(date): Launching consumer on ${CONSUMER_NODE}..."

srun --het-group=1 --ntasks=1 --overlap bash -c '
set -euo pipefail

PROJECT_DIR="/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml"
RUN_DIR="'"${RUN_DIR}"'"
GPU_IDS="'"${GPU_IDS}"'"
NUM_ACTORS="'"${NUM_ACTORS}"'"
COMPILE_MODE="'"${COMPILE_MODE}"'"

echo "$(date): Consumer starting on $(hostname)"
echo "  RUN_DIR:      ${RUN_DIR}"
echo "  GPU_IDS:      ${GPU_IDS}"
echo "  NUM_ACTORS:   ${NUM_ACTORS}"
echo "  COMPILE_MODE: ${COMPILE_MODE:-standard}"

# --- Cleanup handler ---
PEAKNET_PID=""
WRITER_PID=""

cleanup_consumer() {
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
trap cleanup_consumer TERM INT EXIT

# --- Step 1: Activate conda environment ---
echo "$(date): Activating conda environment..."
set +eu
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate /sdf/scratch/users/c/cwang31/miniconda2/pytorch-2.6
set -eu

# --- Step 2: Start Ray head node ---
echo "$(date): Starting Ray head node..."
CUDA_VISIBLE_DEVICES="${GPU_IDS}" ray start --head

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

# --- Step 3: Launch peaknet-pipeline ---
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

# --- Step 4: Wait for pipeline to initialize ---
echo "$(date): Waiting 10s for pipeline to initialize..."
sleep 10

if ! kill -0 "${PEAKNET_PID}" 2>/dev/null; then
    echo "ERROR: peaknet-pipeline exited prematurely"
    wait "${PEAKNET_PID}"
    exit 1
fi

# --- Step 5: Launch cxi-writer ---
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
# Launch producer on Milano (het-group 0)
#===============================================================================
echo "$(date): Launching producers on ${PRODUCER_NODE} (${NUM_PRODUCERS} ranks)..."

"${PROJECT_DIR}/scripts/producer_wrapper.sh" \
    --run-dir "${RUN_DIR}" \
    --num-producers "${NUM_PRODUCERS}" &
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
