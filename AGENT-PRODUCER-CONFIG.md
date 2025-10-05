## Producer Node Agent Configuration

- **Node:** sdfmilan167
- **Working directory:** `/sdf/data/lcls/ds/prj/prjcwang31/results/software/lclstreamer`
- **Environment activation:** `conda.start` (run after SSH)
- **lclstreamer config file:** `examples/lclstreamer-random-to-sdfada-numpy.yaml`
- **Producer launch template:**
  ```
  pixi run --environment psana1 mpirun -n <NUM_PRODUCERS> lclstreamer --config examples/lclstreamer-random-to-sdfada-numpy.yaml
  ```
  **Note:** `<NUM_PRODUCERS>` is configurable (e.g., 4, 8, 16, 32, 64). The actual value should be specified when invoking the experiment based on the desired producer-consumer ratio.
