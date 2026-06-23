#!/usr/bin/env python
"""Interconnect benchmark over torch.distributed (NCCL), any #GPUs/node.

Launched via MPI, one rank per GPU (see run_interconnect_debug.sh). With
gpus_per_node=G and N nodes the world has N*G ranks. We measure:

  * all-reduce over ALL ranks            -> aggregate collective bw (NVLink+IB mix)
  * p2p rank0<->rank1   (same node)      -> intra-node NVLink bw   (if G >= 2)
  * p2p rank0<->rankG   (other node)     -> single inter-node IB link bw (if N >= 2)
  * inter-node bisection (G parallel     -> total cross-node bw with all NICs
    node0->node1 flows)                     (if N >= 2)

All results in GB/s. Whether NCCL uses InfiniBand or TCP is printed by NCCL
itself (NCCL_DEBUG=INFO): grep the logs for 'NET/IB' vs 'NET/Socket'.

IMPORTANT: every pairwise communicator is created up front with dist.new_group()
which ALL ranks call collectively. Lazy/implicit P2P comm creation goes through
ncclCommSplit (collective over the whole world), so creating it on-demand from
only the 2 pair members deadlocks the other ranks.
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


def bench_bisection(groups, sizes, rank, gpus_per_node, nodes, dev):
    """All node0 GPUs send to their node1 counterpart at once (group i = {i, i+G}).
    Aggregate = (G * msg) / slowest-flow-time. Rows on rank 0."""
    rows = []
    node = rank // gpus_per_node
    part = nodes >= 2 and node in (0, 1)
    group = groups[rank % gpus_per_node] if part else None
    is_sender = part and node == 0
    for nbytes in sizes:
        iters, warmup = iters_for(nbytes)
        if part:
            n = nbytes // 4
            sbuf = torch.ones(n, dtype=torch.float32, device=dev)
            rbuf = torch.empty(n, dtype=torch.float32, device=dev)

            def op():
                if is_sender:
                    dist.send(sbuf, group=group, group_dst=1)
                else:
                    dist.recv(rbuf, group=group, group_src=0)
        else:
            def op():
                pass

        t = time_op(op, iters, warmup, dev)
        tt = torch.tensor([t], device=dev)
        dist.all_reduce(tt, op=dist.ReduceOp.MAX)  # slowest flow bounds completion
        if rank == 0:
            rows.append((nbytes, gbps(nbytes * gpus_per_node, tt.item())))
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

    # Pass device_id to enable EAGER initialization. This is essential: it makes
    # dist.new_group() create each subgroup's NCCL comm collectively at creation
    # time (all ranks present for the ncclCommSplit). Without it, the subgroup comm
    # is built lazily on first use by only its 2 members, and the split -- being
    # collective over the whole world -- deadlocks the other ranks.
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank,
                            world_size=world, device_id=dev)

    if rank == 0:
        nccl = ".".join(map(str, torch.cuda.nccl.version()))
        print(f"[interconnect] torch={torch.__version__} cuda={torch.version.cuda} "
              f"nccl={nccl}")
        print(f"[interconnect] world_size={world} nodes={nodes} "
              f"gpus_per_node={gpus_per_node} device={torch.cuda.get_device_name(local_rank)}")
        print("[interconnect] NOTE: grep NCCL INFO for 'NET/IB' (InfiniBand) vs "
              "'NET/Socket' (TCP); NVLink shows as intra-node channels.")

    # Pre-create every pairwise group collectively (ALL ranks call new_group, in
    # the same order). This avoids the ncclCommSplit deadlock from lazy P2P comms.
    intra_group = dist.new_group(ranks=[0, 1]) if gpus_per_node >= 2 else None
    inter_group = dist.new_group(ranks=[0, gpus_per_node]) if nodes >= 2 else None
    bisec_groups = []
    if nodes >= 2:
        for r in range(gpus_per_node):
            bisec_groups.append(dist.new_group(ranks=[r, r + gpus_per_node]))

    sizes = [1 * 1024**2, 4 * 1024**2, 16 * 1024**2, 64 * 1024**2,
             256 * 1024**2, 1024**3]

    # 1) all-reduce over all ranks (NVLink + IB combined)
    ar_rows = []
    for nbytes in sizes:
        x = torch.ones(nbytes // 4, dtype=torch.float32, device=dev)
        iters, warmup = iters_for(nbytes)
        t = time_op(lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM), iters, warmup, dev)
        algbw = gbps(nbytes, t)
        busbw = algbw * 2 * (world - 1) / world
        ar_rows.append((nbytes, t, algbw, busbw))
        del x
        torch.cuda.empty_cache()

    intra_rows = bench_pair(0, 1, intra_group, sizes, rank, dev) if intra_group is not None else []
    inter_rows = bench_pair(0, gpus_per_node, inter_group, sizes, rank, dev) if inter_group is not None else []
    bisec_rows = bench_bisection(bisec_groups, sizes, rank, gpus_per_node, nodes, dev) if bisec_groups else []

    dist.barrier()

    if rank == 0:
        print(f"\n========== ALL-REDUCE over {world} ranks (NVLink+IB) ==========")
        print(f"{'size':>8} | {'time(ms)':>9} | {'algbw(GB/s)':>11} | {'busbw(GB/s)':>11}")
        for nbytes, t, algbw, busbw in ar_rows:
            print(f"{fmt_size(nbytes):>8} | {t*1e3:9.3f} | {algbw:11.2f} | {busbw:11.2f}")

        def ptab(title, rows):
            print(f"\n========== {title} ==========")
            print(f"{'size':>8} | {'uni(GB/s)':>10} | {'bidir/dir':>10}")
            for nbytes, uni, bi in rows:
                print(f"{fmt_size(nbytes):>8} | {uni:10.2f} | {bi:10.2f}")

        if intra_rows:
            ptab("INTRA-node p2p rank0<->rank1 (NVLink)", intra_rows)
        if inter_rows:
            ptab(f"INTER-node p2p rank0<->rank{gpus_per_node} (single IB link)", inter_rows)
        if bisec_rows:
            print(f"\n===== INTER-node BISECTION: {gpus_per_node} parallel node0->node1 "
                  f"flows (total cross-node) =====")
            print(f"{'size':>8} | {'aggregate(GB/s)':>16}")
            for nbytes, agg in bisec_rows:
                print(f"{fmt_size(nbytes):>8} | {agg:16.2f}")

        print("\n[interconnect] SUMMARY")
        print(f"  all-reduce peak busbw : {max(r[3] for r in ar_rows):.2f} GB/s")
        if intra_rows:
            print(f"  NVLink p2p peak (uni) : {max(r[1] for r in intra_rows):.2f} GB/s")
        if inter_rows:
            peak_link = max(r[1] for r in inter_rows)
            print(f"  IB single-link (uni)  : {peak_link:.2f} GB/s (~{peak_link*8:.0f} Gb/s)")
        if bisec_rows:
            peak_bi = max(r[1] for r in bisec_rows)
            print(f"  IB cross-node total   : {peak_bi:.2f} GB/s "
                  f"(~{peak_bi*8:.0f} Gb/s over {gpus_per_node} NIC paths)")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
