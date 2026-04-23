## Producer Node Agent Configuration

- **Node:** sdfmilan203
- **Working directory:** `/sdf/data/lcls/ds/prj/prjcwang31/results/software/lclstreamer`
- **Git branch:** `feature/numpy-serializer`
- **Environment:** pixi (no conda needed)
- **lclstreamer config file:** `examples/lclstreamer-psana1-to-sdfada-numpy-v3.yaml`
- **Producer launch template:**
  ```
  pixi run --environment psana1 mpirun -n <NUM_PRODUCERS> lclstreamer --config examples/lclstreamer-psana1-to-sdfada-numpy-v3.yaml
  ```
  **Note:** `<NUM_PRODUCERS>` is configurable (e.g., 4, 8, 16, 32, 64). The actual value should be specified when invoking the experiment based on the desired producer-consumer ratio.
