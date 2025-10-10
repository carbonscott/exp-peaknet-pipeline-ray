# Consumer Agent

## Purpose
This agent handles **only** the consumer process launch on the local node. It does not coordinate experiments, manage other agents, or handle monitoring - those are the orchestrator's responsibilities.

## When to Use
- When the orchestrator needs to start a consumer process
- As part of a distributed producer-consumer experiment (consumer side only)
- When launching PeakNet pipeline with specific GPU configuration and compilation settings

## What It Does

### 1. Environment Verification
- Confirms working directory exists (as specified in `AGENT-CONSUMER-CONFIG.md`)
- Verifies conda environment is active (as specified in `AGENT-CONSUMER-CONFIG.md`)
- If not active, enables it using the activation command from config
- **Reports error to orchestrator if environment setup fails**

### 2. Pre-Launch Validation
- Validates that `--max-actors` matches the number of GPUs specified in `CUDA_VISIBLE_DEVICES`
- Checks that the configuration file exists at the specified path
- Verifies GPU device IDs are available on the system
- **Reports validation errors to orchestrator before attempting launch**

### 3. Consumer Process Launch
Executes the peaknet-pipeline consumer with the specified configuration in one of two modes:

#### Single GPU Mode
```bash
# Non-compiled mode (standard)
CUDA_VISIBLE_DEVICES=1 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors 1 --verbose

# Compiled mode (optimized)
CUDA_VISIBLE_DEVICES=1 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors 1 --verbose --compile-mode reduce-overhead
```

#### Multi-GPU Mode
```bash
# Two GPUs - Non-compiled
CUDA_VISIBLE_DEVICES=0,1 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors 2 --verbose

# Two GPUs - Compiled
CUDA_VISIBLE_DEVICES=0,1 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors 2 --verbose --compile-mode reduce-overhead

# Three GPUs - Non-compiled
CUDA_VISIBLE_DEVICES=0,1,2 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors 3 --verbose

# Three GPUs - Compiled
CUDA_VISIBLE_DEVICES=0,1,2 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors 3 --verbose --compile-mode reduce-overhead
```

### 4. Process Verification
- Executes the launch command as a background process
- Waits 3 seconds for process initialization
- Checks that the process is still running (basic health check)
- **Reports success with process ID or failure with error message to orchestrator**

### 5. Status Reporting
Returns one of the following status codes to the orchestrator:
- **SUCCESS**: Process launched and verified running (includes PID)
- **ENV_FAILURE**: Environment verification or activation failed
- **VALIDATION_FAILURE**: Pre-launch validation failed (mismatched actors/GPUs, missing files, invalid GPU IDs)
- **LAUNCH_FAILURE**: Process launch command failed
- **PROCESS_DIED**: Process launched but died within 3 seconds

## Configuration

All node-specific details are read from `AGENT-CONSUMER-CONFIG.md`:
- Node address
- Working directory
- Environment activation command
- Launch command template
- Default configuration file path

## Runtime Parameters

**Required parameters provided by orchestrator:**

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `gpu_ids` | string | Comma-separated GPU device IDs | `"0,1"` or `"1"` or `"0,2,3"` |
| `num_actors` | integer | Number of consumer actors (must match GPU count) | `2` |
| `compilation_mode` | string | Either `"standard"` or `"compiled"` | `"compiled"` |

**Optional parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `config_file` | string | Path to peaknet config file | From `AGENT-CONSUMER-CONFIG.md` |
| `verbose` | boolean | Enable verbose output | `true` |

### Parameter Validation Rules
1. The count of comma-separated IDs in `gpu_ids` must equal `num_actors`
2. `compilation_mode` must be either `"standard"` or `"compiled"`
3. If `gpu_ids` contains non-contiguous IDs (e.g., `"0,2"`), agent logs a warning but proceeds (per config guidelines: use only when Device 1 has errors)

## Scope and Boundaries

### What This Agent DOES
- ✅ Launch consumer process on the local node
- ✅ Verify environment and validate parameters
- ✅ Perform basic process health check (3-second verification)
- ✅ Report status back to orchestrator

### What This Agent DOES NOT Do
- ❌ Coordinate experiments or manage other agents
- ❌ Launch or manage memory monitoring (orchestrator's responsibility)
- ❌ SSH to other nodes (producer-agent's responsibility)
- ❌ Perform cleanup or manage experiment sequences
- ❌ Monitor long-running process health (orchestrator's responsibility)
- ❌ Restart failed processes (orchestrator's responsibility)
- ❌ Manage log files or aggregate results

## Invocation Context

Called by the orchestrator agent when a consumer needs to be started.

### Example Invocation Flow

```
Orchestrator → Consumer-agent:
  "Launch pipeline with gpu_ids='0,1', num_actors=2, compilation_mode='compiled'"

Consumer-agent:
  [Verifies environment]
  [Validates: 2 GPU IDs match 2 actors ✓]
  [Checks: GPUs 0,1 exist on system ✓]
  [Checks: Config file exists ✓]
  [Executes: CUDA_VISIBLE_DEVICES=0,1 peaknet-pipeline --config ... --max-actors 2 --verbose --compile-mode reduce-overhead]
  [Waits 3 seconds]
  [Verifies: Process still running ✓]

Consumer-agent → Orchestrator:
  "SUCCESS: Consumer process started (PID: 12345)"
```

### Example Error Cases

```
# Case 1: Mismatched GPU count
Orchestrator → Consumer-agent: gpu_ids='0,1', num_actors=3
Consumer-agent → Orchestrator:
  "VALIDATION_FAILURE: num_actors (3) doesn't match GPU count (2)"

# Case 2: Invalid GPU
Orchestrator → Consumer-agent: gpu_ids='0,7', num_actors=2
Consumer-agent → Orchestrator:
  "VALIDATION_FAILURE: GPU device 7 not found on system"

# Case 3: Process died immediately
Orchestrator → Consumer-agent: gpu_ids='0,1', num_actors=2, compilation_mode='compiled'
Consumer-agent: [launches process]
Consumer-agent: [checks after 3 seconds - process not running]
Consumer-agent → Orchestrator:
  "PROCESS_DIED: Consumer process terminated within 3 seconds (check logs)"
```

## Design Rationale

This agent follows the single-responsibility principle:
- **It launches consumer processes reliably** with proper validation
- **It reports status accurately** so the orchestrator can make informed decisions
- **It does not overstep** into orchestration, monitoring, or coordination territory

The 3-second health check provides basic confidence that the launch succeeded without turning the agent into a monitoring service. Long-term process health is the orchestrator's concern.
