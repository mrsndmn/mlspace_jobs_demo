"""Head-node orchestrator for the multinode vLLM serve demo.

Runs ONLY on the Ray head (rank 0). Sequence:
  1. wait until the Ray cluster has the expected number of nodes/GPUs,
  2. launch `vllm serve` spanning the cluster (pipeline parallel across nodes),
  3. wait for /health, log `ray status`,
  4. send one /v1/completions self-test request and log the generated text,
  5. tear the server down and write the DONE sentinel (always, via finally).

The self-test uses /v1/completions (not /v1/chat/completions) because
HuggingFaceTB/SmolLM2-135M is a base model with no chat template.
See docs/adr/0002-vllm-serve-as-batch-self-test.md
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


def log(msg):
    print(f"[serve_vllm_multinode] {msg}", flush=True)


def wait_for_ray_cluster(expected_nodes, expected_gpus, timeout=600):
    """Block until the Ray cluster is big enough for vLLM placement."""
    import ray

    ray.init(address="auto")
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            resources = ray.cluster_resources()
            gpus = int(resources.get("GPU", 0))
            alive = [n for n in ray.nodes() if n.get("Alive")]
            log(f"ray cluster: alive_nodes={len(alive)} GPUs={gpus} "
                f"(need nodes>={expected_nodes}, GPUs>={expected_gpus})")
            if len(alive) >= expected_nodes and gpus >= expected_gpus:
                log("ray cluster is ready")
                return True
            time.sleep(5)
        return False
    finally:
        # Disconnect this driver; the cluster keeps running for vLLM.
        ray.shutdown()


def log_ray_status(ray_bin):
    try:
        out = subprocess.run([ray_bin, "status"], capture_output=True, text=True, timeout=60)
        log("`ray status`:\n" + out.stdout + (out.stderr or ""))
    except Exception as e:  # noqa: BLE001 - best-effort logging
        log(f"could not run `ray status`: {e}")


def wait_for_health(port, timeout=900):
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    log("vLLM /health is OK")
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(5)
    return False


def run_self_test(port, served_name):
    url = f"http://localhost:{port}/v1/completions"
    payload = json.dumps({
        "model": served_name,
        "prompt": "The capital of France is",
        "max_tokens": 32,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        status = resp.status
        body = json.loads(resp.read())
    text = body["choices"][0]["text"]
    return status, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--served-name", default="SmolLM2-135M")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--expected-nodes", type=int, default=2)
    parser.add_argument("--expected-gpus", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--done-sentinel", required=True)
    args = parser.parse_args()

    env_bin = os.path.dirname(sys.executable)
    vllm_bin = os.path.join(env_bin, "vllm")
    ray_bin = os.path.join(env_bin, "ray")

    rc = 1
    server = None
    try:
        # 1. Cluster must be fully formed before vLLM places the model.
        if not wait_for_ray_cluster(args.expected_nodes, args.expected_gpus):
            log("ERROR: Ray cluster did not reach the expected size in time")
            return 1
        log_ray_status(ray_bin)

        # 2. Launch the server spanning the cluster (Ray backend, PP across nodes).
        cmd = [
            vllm_bin, "serve", args.model,
            "--port", str(args.port),
            "--served-model-name", args.served_name,
            "--pipeline-parallel-size", str(args.pipeline_parallel_size),
            "--tensor-parallel-size", str(args.tensor_parallel_size),
            "--distributed-executor-backend", "ray",
            "--dtype", args.dtype,
            "--enforce-eager",
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
        ]
        log("launching: " + " ".join(cmd))
        server = subprocess.Popen(cmd)

        # 3. Wait until it is serving.
        if not wait_for_health(args.port):
            log("ERROR: vLLM server did not become healthy in time")
            return 1
        log_ray_status(ray_bin)

        # 4. Self-test: one completion request must return HTTP 200 + non-empty text.
        status, text = run_self_test(args.port, args.served_name)
        log(f"self-test HTTP status: {status}")
        log(f"self-test generated text: {text!r}")
        if status == 200 and text and text.strip():
            log("SELF-TEST PASSED")
            rc = 0
        else:
            log("SELF-TEST FAILED (empty text or non-200)")
            rc = 1
        return rc
    except Exception as e:  # noqa: BLE001 - surface any failure as a failed job
        log(f"ERROR during orchestration: {e!r}")
        return 1
    finally:
        # 5. Always stop the server and signal workers to exit.
        if server is not None:
            log("terminating vLLM server")
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
        try:
            with open(args.done_sentinel, "w") as f:
                f.write("done\n")
            log(f"wrote DONE sentinel: {args.done_sentinel}")
        except Exception as e:  # noqa: BLE001
            log(f"could not write DONE sentinel: {e!r}")


if __name__ == "__main__":
    sys.exit(main())
