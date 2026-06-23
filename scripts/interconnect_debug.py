#!/usr/bin/env python
"""Interconnect bandwidth benchmark over torch.distributed (NCCL).

Launched via MPI, one rank per GPU (see run_interconnect_debug.sh). Runs the
canonical all-reduce bandwidth sweep over ALL ranks -- nothing else, no
subgroups. Interpretation depends on the job shape:

  * 2 nodes x G GPUs  -> the all-reduce ring crosses the node boundary, so busbw
                         is bounded by the INTER-node link (InfiniBand). This is
                         the "interconnect between the two nodes" number.
  * 1 node  x G GPUs  -> stays on-node, so busbw reflects intra-node NVLink.

Run both shapes to contrast NVLink vs IB. busbw = algbw * 2*(n-1)/n is the
standard NCCL "bus bandwidth" (comparable to nccl-tests all_reduce_perf).

Deliberately NO dist.new_group / p2p subgroups: creating a subgroup NCCL comm
goes through ncclCommSplit (collective over the whole world) and reliably
deadlocked this 2x8 setup. The world all-reduce needs no subgroup and is robust.
Whether NCCL uses IB or TCP is printed by NCCL itself (NCCL_DEBUG=INFO): grep
the logs for 'NET/IB' vs 'NET/Socket'.
"""
import os

import torch
import torch.distributed as dist


def gbps(nbytes, seconds):
    return nbytes / seconds / 1e9


def fmt_size(nbytes):
    v = float(nbytes)
    for u in ["B", "KB", "MB", "GB"]:
        if v < 1024 or u == "GB":
            return f"{v:.0f}{u}"
        v /= 1024


def iters_for(nbytes):
    if nbytes <= 4 * 1024**2:
        return 50, 10
    if nbytes <= 64 * 1024**2:
        return 30, 5
    return 15, 3


def time_op(op, iters, warmup, dev):
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        op()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0 / iters


def main():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    ngpu = torch.cuda.device_count()
    gpus_per_node = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", ngpu)) or 1
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank % gpus_per_node))
    nodes = max(world // gpus_per_node, 1)
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    if world < 2:
        print(f"[interconnect] WORLD_SIZE={world} (<2): nothing to measure. "
              f"Submit with workers>=2.")
        return 0

    dist.init_process_group(backend="nccl", init_method="env://", rank=rank,
                            world_size=world)

    scope = "INTER-node (IB)" if nodes >= 2 else "intra-node (NVLink)"
    if rank == 0:
        nccl = ".".join(map(str, torch.cuda.nccl.version()))
        print(f"[interconnect] torch={torch.__version__} cuda={torch.version.cuda} "
              f"nccl={nccl}")
        print(f"[interconnect] world_size={world} nodes={nodes} "
              f"gpus_per_node={gpus_per_node} device={torch.cuda.get_device_name(local_rank)}")
        print(f"[interconnect] all-reduce scope: {scope}")
        print("[interconnect] NOTE: grep NCCL INFO for 'NET/IB' (InfiniBand) vs "
              "'NET/Socket' (TCP).")

    sizes = [1 * 1024**2, 4 * 1024**2, 16 * 1024**2, 64 * 1024**2,
             256 * 1024**2, 512 * 1024**2, 1024**3]

    rows = []
    for nbytes in sizes:
        x = torch.ones(nbytes // 4, dtype=torch.float32, device=dev)
        iters, warmup = iters_for(nbytes)
        t = time_op(lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM), iters, warmup, dev)
        algbw = gbps(nbytes, t)
        busbw = algbw * 2 * (world - 1) / world
        rows.append((nbytes, t, algbw, busbw))
        del x
        torch.cuda.empty_cache()

    dist.barrier()

    if rank == 0:
        print(f"\n========== ALL-REDUCE over {world} ranks -- {scope} ==========")
        print(f"{'size':>8} | {'time(ms)':>9} | {'algbw(GB/s)':>11} | {'busbw(GB/s)':>11}")
        for nbytes, t, algbw, busbw in rows:
            print(f"{fmt_size(nbytes):>8} | {t*1e3:9.3f} | {algbw:11.2f} | {busbw:11.2f}")
        peak = max(r[3] for r in rows)
        print(f"\n[interconnect] SUMMARY ({scope})")
        print(f"  peak all-reduce busbw : {peak:.2f} GB/s (~{peak*8:.0f} Gb/s)")
        if nodes >= 2:
            print("  -> this is the effective cross-node bandwidth NCCL achieves over "
                  "the InfiniBand fabric (all NICs combined).")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
