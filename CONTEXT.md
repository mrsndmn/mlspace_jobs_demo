# mlspace_jobs_demo

Demonstration repository for launching ML workloads as MLS (ML Space / Cloud.ru) jobs,
including single-GPU, multi-GPU, and multinode regimes.

## Language

**MLS job**:
A batch workload submitted to the ML Space scheduler via a YAML config or the `mls` API.
_Avoid_: task, run, container

**Multinode regime**:
An MLS job spread across more than one worker (node), coordinated under MPI/PMIX, where a
distributed framework must discover peers from the MPI environment.
_Avoid_: multi-worker, cluster mode, distributed (ambiguous)

**Launch (a vLLM model)**:
Starting `vllm serve` to expose an OpenAI-compatible inference API for the model.
_Avoid_: deploy, run inference, generate

**Ray cluster bootstrap**:
Within a multinode job, electing one worker as the Ray head and joining the others as Ray
workers, so a single vLLM process can span all nodes' GPUs.
_Avoid_: ray init, cluster setup

**Self-test**:
The in-job verification the head node runs against its own vLLM endpoint (health poll plus
one `/v1/chat/completions` request) to prove the model serves correctly across nodes; its
success is the job's pass/fail signal.
_Avoid_: smoke test, healthcheck, probe (when used loosely)

**DONE sentinel**:
A marker file written to the shared NFS by the head after a passing self-test, signalling the
Ray worker nodes to stop and let the job exit cleanly.
_Avoid_: flag file, lock, signal
