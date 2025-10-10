# Consumer Node Agent Configuration

## Node Information
- **Node:** sdfada001
- **Working directory:** `/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml`
- **Environment activation:** `conda.torch`
- **peaknet-pipeline config file:** `peaknet-socket-profile-673m.yaml`

## Launch Command Templates

### Single GPU Configuration

**Non-compiled:**
```bash
CUDA_VISIBLE_DEVICES=1 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors 1 --verbose
```

**Compiled:**
```bash
CUDA_VISIBLE_DEVICES=1 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors 1 --verbose --compile-mode reduce-overhead
```

### Multiple GPU Configuration

**Non-compiled:**
```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors <NUM_CONSUMERS> --verbose
```

**Compiled:**
```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors <NUM_CONSUMERS> --verbose --compile-mode reduce-overhead
```

## GPU Selection Guidelines

### Best Practices

1. **Match `--max-actors` to the number of GPUs specified:** The `<NUM_CONSUMERS>` value should equal the number of GPU IDs listed in `CUDA_VISIBLE_DEVICES`.

2. **Prefer contiguous GPU IDs when all devices are healthy:** Use consecutive GPU IDs (e.g., `0,1` or `0,1,2`) for optimal performance and simplicity.

3. **Use non-contiguous GPU IDs only when necessary:** Skip problematic devices (e.g., use `0,2` when Device 1 has errors).

### Example Configurations

#### Two GPU Setup (Preferred)
```bash
CUDA_VISIBLE_DEVICES=0,1 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors 2 --verbose
```
Use this when GPUs 0 and 1 are both functioning properly.

#### Two GPU Setup (Alternative)
```bash
CUDA_VISIBLE_DEVICES=0,2 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors 2 --verbose
```
Use this when Device 1 has errors or is unavailable.

#### Three GPU Setup (Preferred)
```bash
CUDA_VISIBLE_DEVICES=0,1,2 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors 3 --verbose
```
Use this when GPUs 0, 1, and 2 are all functioning properly.

#### Three GPU Setup (Alternative)
```bash
CUDA_VISIBLE_DEVICES=0,2,3 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors 3 --verbose
```
Use this when Device 1 has errors or is unavailable.

## Configuration Parameters

- **`<GPU_IDS>`**: Comma-separated list of GPU device IDs to use (e.g., `0,1` or `0,2,3`)
- **`<NUM_CONSUMERS>`**: Number of consumer actors, should match the count of GPUs specified in `CUDA_VISIBLE_DEVICES`
- **`--compile-mode reduce-overhead`**: Optional compilation mode for optimized performance

## Notes

- The actual values for `<GPU_IDS>` and `<NUM_CONSUMERS>` should be specified when invoking the experiment based on:
  - Available GPU resources
  - Desired producer-consumer ratio
  - GPU health status
- Always verify GPU availability and health before launching with multiple devices
- Non-contiguous GPU selection (e.g., skipping Device 1) is acceptable but should only be used when specific devices are experiencing issues
