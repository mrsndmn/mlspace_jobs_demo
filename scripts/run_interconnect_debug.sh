#! /bin/bash

# Multinode interconnect diagnostic (runs on EVERY node under MPI/PMIX).
#
# Part 1 (every node): print a hardware/network inventory -- does this node have
#   an InfiniBand/RoCE HCA, link rate, active state, GPU<->NIC topology.
# Part 2 (collective): run a torch.distributed (NCCL) benchmark BETWEEN the nodes
#   and report achieved channel throughput. NCCL_DEBUG=INFO reveals whether NCCL
#   transports over IB (NET/IB) or falls back to TCP sockets (NET/Socket).
#
# Note: we DON'T use `set -e` -- some inspection tools may be absent on the image,
# and we still want every other diagnostic to run.

set +e

ENV_PREFIX=/workspace-SR004.nfs2/d.tarasov/envs/vllm_cu126/bin
WORKDIR=/workspace-SR004.nfs2/d.tarasov/mlspace_jobs_demo

# `python` is a real ELF binary; the `ray`/`vllm` console scripts are not portable
# across mount paths (see commit history), so we only ever call python directly.
PY_BIN="$ENV_PREFIX/python"

# --- Discover the master (rank-0) node from the MPI/PMIX environment ---
MASTER_HOST_PREFIX=$(perl -E "my \$x = '$PMIX_HOSTNAME'; \$x =~ s/-\w+-\d+$//; print \$x ")
MASTER_HOST=$(perl -E "my \$x = '$PMIX_HOSTNAME'; \$x =~ s/-\w+-\d+$/-mpimaster-0/; print \$x ")
MASTER_HOST_FULL="$MASTER_HOST.$MASTER_HOST_PREFIX"

# torch.distributed reads these (init_method='env://')
export MASTER_ADDR=$MASTER_HOST_FULL
export MASTER_PORT=12345
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
export RANK=${OMPI_COMM_WORLD_RANK:-0}

# Make NCCL tell us which transport it picks (NET/IB vs NET/Socket). Keep the
# subsystems minimal -- with one rank per GPU the INIT+NET logs are already large,
# and GRAPH/ENV would flood the NFS tee on a 16-rank job.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

HOST=$(hostname)

# NCCL_DEBUG=INFO is very verbose, and `mls job logs` only returns a tail window,
# which can hide the early inventory. Also persist the FULL per-node output to a
# file on the shared NFS so nothing is lost to log truncation.
LOGDIR="$WORKDIR/run/interconnect_debug"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/rank${RANK}_${HOST}.txt"
exec > >(tee -a "$LOGFILE") 2>&1
echo "(full output of this rank is also saved to: $LOGFILE)"

# With one process per GPU (e.g. 8 ranks/node), only the node-local rank 0 prints
# the hardware inventory -- otherwise it repeats once per GPU.
LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
if [ "$LOCAL_RANK" = "0" ]; then
    echo "########################################################################"
    echo "## NODE INVENTORY | rank=$RANK/$WORLD_SIZE | host=$HOST | master=$MASTER_ADDR"
    echo "########################################################################"

    echo "==== [1] network interfaces (ip -br) ===="
    ip -br link 2>/dev/null
    ip -br addr 2>/dev/null

    echo "==== [2] PCI InfiniBand/Mellanox devices (lspci) ===="
    lspci 2>/dev/null | grep -iE 'mellanox|infiniband|connectx' || echo "(none found / lspci unavailable)"

    echo "==== [3] /sys/class/infiniband (authoritative IB/RoCE presence) ===="
    if [ -d /sys/class/infiniband ] && [ -n "$(ls -A /sys/class/infiniband 2>/dev/null)" ]; then
        for dev in /sys/class/infiniband/*; do
            echo "HCA: $(basename "$dev")  fw=$(cat "$dev/fw_ver" 2>/dev/null)  node_guid=$(cat "$dev/node_guid" 2>/dev/null)"
            for port in "$dev"/ports/*; do
                echo "  port $(basename "$port"): state=$(cat "$port/state" 2>/dev/null) | phys=$(cat "$port/phys_state" 2>/dev/null) | rate=$(cat "$port/rate" 2>/dev/null) | link_layer=$(cat "$port/link_layer" 2>/dev/null)"
            done
        done
    else
        echo "(no /sys/class/infiniband entries -> NO InfiniBand/RoCE HCA visible on this node)"
    fi

    echo "==== [4] ibstat / ibv_devinfo (if rdma-core/perftest installed) ===="
    if command -v ibstat >/dev/null 2>&1; then ibstat; else echo "(ibstat not installed)"; fi
    if command -v ibv_devinfo >/dev/null 2>&1; then ibv_devinfo 2>/dev/null; else echo "(ibv_devinfo not installed)"; fi

    echo "==== [5] UCX transports (HPCX) ===="
    if command -v ucx_info >/dev/null 2>&1; then
        ucx_info -v 2>/dev/null | head -3
        ucx_info -d 2>/dev/null | grep -iE 'Transport|Device:|bandwidth|latency' | head -60
    else
        echo "(ucx_info not available)"
    fi

    echo "==== [6] GPU <-> NIC topology (nvidia-smi topo -m) ===="
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi topo -m 2>/dev/null; else echo "(nvidia-smi unavailable)"; fi
    echo
fi
echo "########################################################################"
echo "## NCCL BANDWIDTH BENCHMARK (inter-node) | starting from rank=$RANK"
echo "########################################################################"
cd "$WORKDIR"
"$PY_BIN" scripts/interconnect_debug.py
RC=$?
echo "rank=$RANK interconnect_debug.py exited rc=$RC"
exit $RC
