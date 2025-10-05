## Consumer Node Agent Configuration

- **Node:** sdfada013
- **Working directory:** `/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml`
- **Environment activation:** `conda.torch`
- **peaknet-pipeline config file:** `peaknet-socket-profile-673m.yaml`
- **Launch command template (non-compiled):**
  ```
  CUDA_VISIBLE_DEVICES=1 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors <NUM_CONSUMERS> --verbose
  ```
- **Launch command template (compiled):**
  ```
  CUDA_VISIBLE_DEVICES=1 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors <NUM_CONSUMERS> --verbose --compile-mode reduce-overhead
  ```
  **Note:** `<NUM_CONSUMERS>` is configurable. The actual value should be specified when invoking the experiment based on the desired producer-consumer ratio.
