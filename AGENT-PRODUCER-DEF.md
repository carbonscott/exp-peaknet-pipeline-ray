# Producer Agent

**Purpose:**
This agent handles **only** producer process launch on a remote node. It is invoked via SSH by the orchestrator to set up and launch streaming data producers with configurable parallelism.

**When to Use:**
- When the orchestrator needs to launch producers on a remote node
- As part of distributed producer-consumer experiments (producer side only)
- When conducting producer-consumer ratio scaling studies

**What It Does:**

1. **Remote Connection:**
   - SSHs to the producer node (as specified in `AGENT-PRODUCER-CONFIG.md`)

2. **Environment Setup:**
   - Navigates to working directory (as specified in `AGENT-PRODUCER-CONFIG.md`)
   - Activates conda environment using activation command from config

3. **Producer Launch:**
   - Executes the producer launch command template from `AGENT-PRODUCER-CONFIG.md` with **orchestrator-specified parallelism**
   - The number of producers (e.g., 8, 16, 32, 64, 96) is provided at runtime by the orchestrator and substituted into the template

4. **Process Management:**
   - Ensures producers are running
   - Reports success to orchestrator
   - That's it - no monitoring, no coordination, no cleanup

**Configuration:**
All node-specific details (node address, working directory, environment activation command, launch command template) are read from `AGENT-PRODUCER-CONFIG.md`.

The launch template in the config file uses `<NUM_PRODUCERS>` as a placeholder that gets substituted with the actual value at runtime.

**Runtime Parameters (provided by orchestrator):**
- **Number of producers:** Variable (8, 16, 32, 64, 96, etc.) - passed as the value to substitute `<NUM_PRODUCERS>` in the launch template

**Scope:**
- This agent handles **only producer launch** on the remote producer node
- It does **not** coordinate experiments or manage other agents
- It does **not** monitor the consumer or memory usage
- It does **not** handle cleanup (orchestrator's job)
- It operates entirely on the remote producer node

**Critical Timing Note:**
- The orchestrator is responsible for timing when to invoke this agent
- For **compiled mode** experiments, the orchestrator waits until the consumer has processed ~100 batches (compilation complete) before invoking the producer-agent
- For **non-compiled mode** experiments, the orchestrator can invoke this agent immediately
- This agent does not need to know about compilation - it just launches producers when told to

**Invocation Context:**
- Called by the orchestrator via SSH when producers need to be started
- Receives the desired number of producers as a parameter
- Returns control to orchestrator after launch confirmation

**Example Invocation:**
```
Orchestrator: "Producer-agent, launch 32 producers"
Producer-agent: [Reads AGENT-PRODUCER-CONFIG.md]
Producer-agent: [SSHs to configured node, activates environment]
Producer-agent: [Executes launch template with 32 substituted for <NUM_PRODUCERS>]
Producer-agent: "32 producers launched successfully"
[Returns control to orchestrator]
```