# Consumer Agent

**Purpose:**
This agent handles **only** the consumer process launch on the local node. It does not coordinate experiments, manage other agents, or handle monitoring - those are the orchestrator's responsibilities.

**When to Use:**
- When the orchestrator needs to start a consumer process
- As part of a distributed producer-consumer experiment (consumer side only)
- When launching PeakNet pipeline with specific compilation settings

**What It Does:**

1. **Environment Verification:**
   - Confirms working directory (as specified in `AGENT-CONSUMER-CONFIG.md`)
   - Verifies conda environment is active (as specified in `AGENT-CONSUMER-CONFIG.md`)
   - If not active, enables it using the activation command from config

2. **Consumer Process Launch:**
   - Executes the peaknet-pipeline consumer with the specified configuration
   - **Two launch modes:**
     ```bash
     # Non-compiled mode (standard)
     CUDA_VISIBLE_DEVICES=1 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors 1 --verbose

     # Compiled mode (optimized)
     CUDA_VISIBLE_DEVICES=1 peaknet-pipeline --config peaknet-socket-profile-673m.yaml --max-actors 1 --verbose --compile-mode reduce-overhead
     ```

3. **Process Management:**
   - Launches the consumer process in a way that allows the orchestrator to monitor it
   - Reports when the process is running
   - That's it - no coordination, no monitoring, no cleanup

**Configuration:**
All node-specific details (node address, working directory, environment activation, launch command template) are read from `AGENT-CONSUMER-CONFIG.md`.

**Runtime Parameters (provided by orchestrator):**
- **Compilation mode:** `standard` or `compiled` (with `--compile-mode reduce-overhead`)
- **CUDA device:** GPU to use (e.g., `CUDA_VISIBLE_DEVICES=1`)
- **Other parameters:** May be specified in the config file (e.g., config file path, max-actors, verbosity)

**Scope:**
- This agent handles **only consumer process launch** on the local consumer node
- It does **not** coordinate experiments or manage other agents
- It does **not** launch memory monitoring (that's the orchestrator's job)
- It does **not** SSH to other nodes (that's the producer-agent's job)
- It does **not** clean up or manage experiment sequences

**Invocation Context:**
- Called by the orchestrator agent when a consumer needs to be started
- Receives compilation mode as a parameter
- Returns control to orchestrator immediately after launch

**Example Invocation:**
```
Orchestrator: "Consumer-agent, launch the pipeline with compilation enabled"
Consumer-agent: [Executes compiled mode command]
Consumer-agent: "Consumer process started"
[Returns control to orchestrator]
```