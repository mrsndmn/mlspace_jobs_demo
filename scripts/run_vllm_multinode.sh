#! /bin/bash

# Multinode vLLM launcher (runs on EVERY node under MPI/PMIX).
#
# Mechanism:
#   * MPI launches one process per node (resource.processes=1 in the yaml).
#   * We reuse the master-host discovery from scripts/run_training_multinode.sh
#     to find the head node, but coordination is done by RAY, not by DDP env vars.
#   * rank 0  -> Ray head + vllm serve + self-test (scripts/serve_vllm_multinode.py)
#   * rank !0 -> Ray worker, then wait for the DONE sentinel and exit.
# See docs/adr/0001-ray-on-mpi-bootstrap-for-vllm-multinode.md

set -ex

ENV_PREFIX=/workspace-SR004.nfs2/d.tarasov/envs/vllm_cu126/bin
WORKDIR=/workspace-SR004.nfs2/d.tarasov/mlspace_jobs_demo

MODEL="HuggingFaceTB/SmolLM2-135M"
SERVED_NAME="SmolLM2-135M"
PORT=8000
PP_SIZE=2          # pipeline parallel: одна стадия на ноду
TP_SIZE=1          # tensor parallel внутри ноды (1 GPU на ноду)
RAY_PORT=6379

# --- Discover the head (master) node from the MPI/PMIX environment ---
# Same perl trick as scripts/run_training_multinode.sh: hostname looks like
# <prefix>-<role>-<idx>; the head is <prefix>-mpimaster-0.<domain>
MASTER_HOST_PREFIX=$(perl -E "my \$x = '$PMIX_HOSTNAME'; \$x =~ s/-\w+-\d+$//; print \$x ")
MASTER_HOST=$(perl -E "my \$x = '$PMIX_HOSTNAME'; \$x =~ s/-\w+-\d+$/-mpimaster-0/; print \$x ")
MASTER_HOST_FULL="$MASTER_HOST.$MASTER_HOST_PREFIX"

RANK=${OMPI_COMM_WORLD_RANK:-0}
WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
RAY_ADDRESS="$MASTER_HOST_FULL:$RAY_PORT"

echo "RANK=$RANK WORLD_SIZE=$WORLD_SIZE PMIX_HOSTNAME=$PMIX_HOSTNAME HEAD=$MASTER_HOST_FULL"

# --- Shared-NFS sentinel used to tell workers the job is finished ---
SENTINEL_DIR="$WORKDIR/run/vllm_multinode"
DONE_SENTINEL="$SENTINEL_DIR/DONE"
mkdir -p "$SENTINEL_DIR"

cd "$WORKDIR"
export PATH="$ENV_PREFIX:$PATH"

RAY_BIN="$ENV_PREFIX/ray"
PY_BIN="$ENV_PREFIX/python"

if [ "$RANK" = "0" ]; then
    # ============================ HEAD NODE ============================
    # Clear any stale sentinel BEFORE starting Ray. Workers can only join
    # after the head is up, so they will never observe a stale DONE.
    rm -f "$DONE_SENTINEL"

    "$RAY_BIN" stop || true
    "$RAY_BIN" start --head --port "$RAY_PORT" --num-gpus 1 --include-dashboard=false

    # Run serve + self-test. Always drop the sentinel afterwards so the
    # worker can exit even if the orchestrator crashes.
    set +e
    "$PY_BIN" scripts/serve_vllm_multinode.py \
        --model "$MODEL" \
        --served-name "$SERVED_NAME" \
        --port "$PORT" \
        --pipeline-parallel-size "$PP_SIZE" \
        --tensor-parallel-size "$TP_SIZE" \
        --expected-nodes "$WORLD_SIZE" \
        --expected-gpus "$WORLD_SIZE" \
        --done-sentinel "$DONE_SENTINEL"
    RC=$?
    set -e

    touch "$DONE_SENTINEL"
    "$RAY_BIN" stop || true
    echo "HEAD finished with rc=$RC"
    exit $RC
else
    # =========================== WORKER NODE ==========================
    # Retry the join until the head's Ray is reachable.
    until "$RAY_BIN" start --address "$RAY_ADDRESS" --num-gpus 1; do
        echo "head $RAY_ADDRESS not ready yet, retrying Ray join in 5s..."
        "$RAY_BIN" stop || true
        sleep 5
    done

    echo "WORKER joined Ray at $RAY_ADDRESS, waiting for DONE sentinel..."
    while [ ! -f "$DONE_SENTINEL" ]; do
        sleep 5
    done

    echo "WORKER saw DONE sentinel, stopping Ray and exiting."
    "$RAY_BIN" stop || true
    exit 0
fi
