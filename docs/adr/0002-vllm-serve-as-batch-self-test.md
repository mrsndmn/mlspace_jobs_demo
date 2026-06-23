# Use `vllm serve` as a self-testing batch job, not a persistent endpoint

The goal is to demonstrate launching SmolLM2-135M with `vllm serve` (OpenAI-compatible API) in a
multinode regime. But MLS `binary` jobs are batch workloads with **no external ingress**, so a
long-lived server would be unreachable and would keep the job (and 2 GPUs) occupied indefinitely.
We instead start the server on the head, wait for `/health`, log `ray status` (confirming 2 nodes),
send one `/v1/chat/completions` self-test request, log the generated text, then shut the server
down so the job **exits green**.

This is surprising (`vllm serve` normally implies a persistent endpoint), a real trade-off (we give
up a reusable endpoint for a clean, self-verifying batch run), and meaningfully shapes the
entrypoint. Cross-node shutdown is coordinated with a DONE sentinel file on the shared NFS that all
nodes see: the head writes it after the self-test, the Ray worker polls for it and exits, preventing
the job from hanging.
