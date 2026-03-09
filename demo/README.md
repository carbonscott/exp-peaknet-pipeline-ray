# PeakNet Streaming Pipeline Demo

End-to-end real-time crystallography inference: LCLS detector data streams from
psana through ZMQ to PeakNet-673M on GPU, with results written to CXI files.

## Architecture

```
Milano (CPU) — native                Ada (GPU) — Apptainer container
+------------------+                 +------------------------------------------+
| lclstreamer      |  ZMQ PUSH/PULL  | socket_producer.py                       |
| N MPI ranks      | ──────────────> | (2 sockets, ports 12321-12322)           |
| psana1 -> image  |                 |       |                                  |
| pad -> 1696x1696 |                 |       v                                  |
| serialize (npz)  |                 | Q1 (Ray ShardedQueue, 8 shards)          |
+------------------+                 |       |                                  |
                                     |       v                                  |
  pixi + psana1                      | PeakNet-673M (4x L40S GPUs)             |
  (no container)                     | batch=8, bfloat16, 3-way buffered       |
                                     |       |                                  |
                                     |       v                                  |
                                     | Q2 -> CXI Writer (peaks + HDF5 output)  |
                                     +------------------------------------------+
                                       consumer.sif (apptainer exec --nv)
```

**Data flow:** psana1 event -> assembled image (1667x1668) -> pad to 1696x1696
-> ZMQ -> Ray queue -> GPU inference -> peak finding (argmax) -> CXI file

## Prerequisites

- S3DF account with `lcls:prjdat21` allocation
- Access to experiment data (default: mfxl1038923 run 278)
- PeakNet-673M weights at `peaknet-673m.bin` (in the parent directory by default)

## One-Time Setup

### 1. Build the consumer container

```bash
cd demo/
apptainer build --fakeroot consumer.sif consumer.def
```

This builds a container with PyTorch, Ray, PeakNet, and the pipeline packages.
Takes ~10-15 minutes. The resulting `consumer.sif` is ~6 GB.

### 2. Install lclstreamer (producer side)

lclstreamer uses pixi with psana1. If not already installed:

```bash
cd /path/to/lclstreamer
pixi install
```

The default path is set in `run-demo.sbatch` via `LCLSTREAMER_DIR`. Override it
if your installation is elsewhere:

```bash
LCLSTREAMER_DIR=/my/path/to/lclstreamer sbatch run-demo.sbatch
```

### 3. Model weights

The pipeline config points to `peaknet-673m.bin` in the parent project directory.
This path is bind-mounted into the container via `--bind /sdf`. No action needed
if the weights are already at:

```
/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml/peaknet-673m.bin
```

## Running the Demo

```bash
cd demo/
mkdir -p runs        # SLURM needs this dir for job output logs
sbatch run-demo.sbatch
```

This submits a heterogeneous SLURM job:
- **Component 0 (Milano):** N lclstreamer producers (CPU, native with pixi)
- **Component 1 (Ada):** peaknet-pipeline + CXI writer (GPU, inside Apptainer)

### Monitor the job

```bash
# Job status
squeue -j <JOBID>

# Output log (producer + orchestration)
tail -f runs/peaknet-<JOBID>.out

# Error log (consumer pipeline + GPU actors)
tail -f runs/peaknet-<JOBID>.err

# Batch processing progress
grep "Processed batch" runs/peaknet-<JOBID>.err
```

### Cancel

```bash
scancel <JOBID>
```

## Customizing

Override defaults with environment variables:

```bash
# Number of producers (default: 8)
NUM_PRODUCERS=16 sbatch run-demo.sbatch

# Enable torch.compile for faster inference
COMPILE_MODE=reduce-overhead sbatch run-demo.sbatch

# Change GPU count / actor count
NUM_ACTORS=2 GPU_IDS="0,1" sbatch run-demo.sbatch

# Different experiment/run
EXPERIMENT=mfxl1047723 RUN=r0266 sbatch run-demo.sbatch

# Custom lclstreamer path
LCLSTREAMER_DIR=/my/lclstreamer sbatch run-demo.sbatch
```

## Config Consistency Rules

These settings **must match** across configs:

| Setting | lclstreamer.yaml | peaknet.yaml | cxi_writer.yaml |
|---------|-----------------|--------------|-----------------|
| Batch size | `batch_size: 8` | `batch_size: 8` | - |
| Image dims | `target_height/width: 1696` | `image_size: 1696`, `data.shape: [1,1696,1696]` | - |
| Queue name | - | `output_queue: peaknet_q2` | `name: peaknet_q2` |
| Queue shards | - | `queue_num_shards: 8` | `num_shards: 8` |
| Ray namespace | - | `namespace: peaknet-pipeline` | `namespace: peaknet-pipeline` |

## Viewing Results

CXI output is written to `runs/<timestamp>-<jobid>/cxi_output/`.

Launch the interactive visualizer (from the parent project directory):

```bash
marimo edit inspect_writer_marimo.py
```

## Troubleshooting

### Container build fails
- Ensure `--fakeroot` is used (no root required)
- Check network access to `nvcr.io` and `github.com`
- On S3DF, Apptainer v1.4.1 is available by default

### Job stuck in "PD" (Pending)
- Check reason: `squeue -j <JOBID> --format="%.18i %.9P %.2t %R"`
- Het jobs need both Milano AND Ada nodes simultaneously

### GPU OOM
- Reduce `batch_size` in both lclstreamer and peaknet configs (try 4 or 2)
- Or reduce `pipeline_concurrency` from 3 to 2 in peaknet.yaml

### Producers fail to connect
- Consumer must be running first (ZMQ PULL binds on Ada)
- The producer retries automatically for up to 5 minutes
- Check Ada node reachability: `ssh <ada-node> hostname`
- Known issue: sdfada007 has network problems and GPU ECC errors

### No "Processed batch" messages
- Check GPU utilization: `ssh <ada-node> nvidia-smi`
- 0% utilization with high memory = likely OOM
- Check for CUDA errors: `grep -i "cuda\|oom\|error" runs/peaknet-<JOBID>.err`

## What's in the Container vs. Native

| Component | Where | Why |
|-----------|-------|-----|
| PeakNet model + inference | Container (Ada GPU) | Needs PyTorch + CUDA, isolated from system Python |
| Ray cluster | Container (Ada GPU) | Co-located with inference actors |
| CXI writer | Container (Ada GPU) | Reads from Ray Q2, needs cxi-pipeline-ray |
| lclstreamer + psana1 | Native (Milano CPU) | psana1 requires pixi + LCLS conda channels, not easily containerized |
| Model weights | Bind-mounted `/sdf` | Shared filesystem, no need to bake into image |

## Directory Layout

```
demo/
├── README.md              # This file
├── consumer.def           # Apptainer definition for GPU consumer
├── consumer.sif           # Built container (not checked in, ~6 GB)
├── configs/
│   ├── lclstreamer.yaml   # Producer config (CONSUMER_NODE placeholder)
│   ├── peaknet.yaml       # Pipeline config (CONSUMER_NODE placeholder)
│   └── cxi_writer.yaml    # CXI output config
├── run-demo.sbatch        # Het-job launch script
└── runs/                  # Created at runtime
    └── <timestamp>-<jobid>/
        ├── peaknet.yaml       # Patched configs for this run
        ├── cxi_writer.yaml
        ├── lclstreamer.yaml
        ├── run-info.txt
        └── cxi_output/        # CXI result files
```
