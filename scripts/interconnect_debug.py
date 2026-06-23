#!/usr/bin/env python
"""Interconnect benchmark over torch.distributed (NCCL), any #GPUs/node.

Launched via MPI, one rank per GPU (see run_interconnect_debug.sh). With
gpus_per_node=G and N nodes the world has N*G ranks. We measure:

  * all-reduce over ALL ranks         -> aggregate bw; with N>1 this is bounded by
                                         the INTER-node link (the ring crosses nodes)
  * all-reduce within each node (G)    -> intra-node NVLink bw at scale
  * p2p rank0<->rank1 (same node)      -> intra-node NVLink point-to-point (if G>=2)

All results in GB/s. Whether NCCL uses InfiniBand or TCP is printed by NCCL
itself (NCCL_DEBUG=INFO): grep the logs for 'NET/IB' vs 'NET/Socket'.

Design notes (learned the hard way):
  * Subgroup NCCL comms are created via ncclCommSplit, which is COLLECTIVE over the
    whole world. We pass device_id to init_process_group so new_group() builds them
    EAGERLY (all ranks present) instead of lazily on first use (which deadlocks).
  * We only ever build SAME-NODE subgroups. Cross-node 2-rank subgroups proved
    fragile here, so inter-node bandwidth is read from the world all-reduce (which
    is inter-node-bound) rather than from per-pair p2p.
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
    """Mean seconds/op via CUDA events. EVERY rank must call this (barrier inside)."""
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


def bench_allreduce(sizes, dev, rank, group, gsize, report_rank):
    """all-reduce within `group` (None = world). All ranks call; the timed op uses
    each rank's own `group`. Rows returned only on report_rank."""
    rows = []
    for nbytes in sizes:
        x = torch.ones(nbytes // 4, dtype=torch.float32, device=dev)
        iters, warmup = iters_for(nbytes)
        t = time_op(lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group),
                    iters, warmup, dev)
        algbw = gbps(nbytes, t)
        busbw = algbw * 2 * (gsize - 1) / gsize
        if rank == report_rank:
            rows.append((nbytes, t, algbw, busbw))
        del x
        torch.cuda.empty_cache()
    return rows


def bench_pair(lo, hi, group, sizes, rank, dev):
    """p2p between global ranks lo<hi using a pre-created 2-rank `group`
    (group-local 0=lo, 1=hi). ALL ranks call this; non-members no-op so the
    per-op barriers stay aligned. Rows returned only on rank lo."""
    rows = []
    part = rank in (lo, hi)
    peer_local = (1 if rank == lo else 0) if part else None
    for nbytes in sizes:
        iters, warmup = iters_for(nbytes)
        if part:
            n = nbytes // 4
            sbuf = torch.ones(n, dtype=torch.float32, device=dev)
            rbuf = torch.empty(n, dtype=torch.float32, device=dev)

            def uni():
                if rank == lo:
                    dist.send(sbuf, group=group, group_dst=1)
                else:
                    dist.recv(rbuf, group=group, group_src=0)

            def bidir():
                ops = [dist.P2POp(dist.isend, sbuf, group=group, group_peer=peer_local),
                       dist.P2POp(dist.irecv, rbuf, group=group, group_peer=peer_local)]
                for w in dist.batch_isend_irecv(ops):
                    w.wait()
        else:
            def uni():
                pass

            def bidir():
                pass

        t_uni = time_op(uni, iters, warmup, dev)
        t_bi = time_op(bidir, iters, warmup, dev)
        if rank == lo:
            rows.append((nbytes, gbps(nbytes, t_uni), gbps(nbytes, t_bi)))
        if part:
            del sbuf, rbuf
            torch.cuda.empty_cache()
    return rows


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

    # device_id -> EAGER init so new_group() builds same-node subgroup comms
    # collectively at creation (avoids the lazy ncclCommSplit deadlock).
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank,
                            world_size=world, device_id=dev)

    if rank == 0:
        nccl = ".".join(map(str, torch.cuda.nccl.version()))
        print(f"[interconnect] torch={torch.__version__} cuda={torch.version.cuda} "
              f"nccl={nccl}")
        print(f"[interconnect] world_size={world} nodes={nodes} "
              f"gpus_per_node={gpus_per_node} device={torch.cuda.get_device_name(local_rank)}")
        print("[interconnect] NOTE: grep NCCL INFO for 'NET/IB' (InfiniBand) vs "
              "'NET/Socket' (TCP); inter-node bw = world all-reduce busbw below.")

    # Same-node subgroups only (built eagerly, all ranks call new_group in order).
    node_groups = [dist.new_group(ranks=list(range(n * gpus_per_node,
                                                    (n + 1) * gpus_per_node)))
                   for n in range(nodes)]
    my_node_group = node_groups[rank // gpus_per_node]
    intra_group = dist.new_group(ranks=[0, 1]) if gpus_per_node >= 2 else None

    sizes = [1 * 1024**2, 4 * 1024**2, 16 * 1024**2, 64 * 1024**2,
             256 * 1024**2, 1024**3]

    world_rows = bench_allreduce(sizes, dev, rank, None, world, report_rank=0)
    node_rows = bench_allreduce(sizes, dev, rank, my_node_group, gpus_per_node,
                                report_rank=0) if nodes >= 2 else []
    intra_rows = bench_pair(0, 1, intra_group, sizes, rank, dev) if intra_group is not None else []

    dist.barrier()

    if rank == 0:
        def artab(title, rows):
            print(f"\n========== {title} ==========")
            print(f"{'size':>8} | {'time(ms)':>9} | {'algbw(GB/s)':>11} | {'busbw(GB/s)':>11}")
            for nbytes, t, algbw, busbw in rows:
                print(f"{fmt_size(nbytes):>8} | {t*1e3:9.3f} | {algbw:11.2f} | {busbw:11.2f}")

        artab(f"ALL-REDUCE over ALL {world} ranks (INTER-node bound)", world_rows)
        if node_rows:
            artab(f"ALL-REDUCE within one node ({gpus_per_node} GPUs, NVLink)", node_rows)
        if intra_rows:
            print(f"\n========== INTRA-node p2p rank0<->rank1 (NVLink) ==========")
            print(f"{'size':>8} | {'uni(GB/s)':>10} | {'bidir/dir':>10}")
            for nbytes, uni, bi in intra_rows:
                print(f"{fmt_size(nbytes):>8} | {uni:10.2f} | {bi:10.2f}")

        print("\n[interconnect] SUMMARY")
        w_peak = max(r[3] for r in world_rows)
        print(f"  world all-reduce busbw (inter-node) : {w_peak:.2f} GB/s (~{w_peak*8:.0f} Gb/s)")
        if node_rows:
            n_peak = max(r[3] for r in node_rows)
            print(f"  per-node all-reduce busbw (NVLink)  : {n_peak:.2f} GB/s")
        if intra_rows:
            print(f"  NVLink p2p peak (uni)               : {max(r[1] for r in intra_rows):.2f} GB/s")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
