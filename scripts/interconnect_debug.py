#!/usr/bin/env python
"""Inter-node interconnect benchmark over torch.distributed (NCCL).

Launched via MPI on >=2 nodes (one rank per node, see run_interconnect_debug.sh).
Reports:
  * torch / cuda / NCCL versions and the GPU model,
  * all-reduce algbw + busbw across the ranks (the headline inter-node number),
  * point-to-point send/recv throughput between rank 0 and rank 1
    (unidirectional and bidirectional / full-duplex), in GB/s.

Whether NCCL transports over InfiniBand or falls back to TCP sockets is printed
by NCCL itself (NCCL_DEBUG=INFO): grep the logs for 'NET/IB' vs 'NET/Socket'.
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


def time_op(op, iters, warmup):
    """Median-free mean time per op using CUDA events. Barriers all ranks first."""
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
    return start.elapsed_time(end) / 1000.0 / iters  # seconds per iter


def main():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    ngpu = torch.cuda.device_count()
    local = rank % max(ngpu, 1)
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    if world < 2:
        print(f"[interconnect] WORLD_SIZE={world} (<2): nothing to measure between "
              f"nodes. Submit with workers>=2.")
        return 0

    dist.init_process_group(backend="nccl", init_method="env://", rank=rank,
                            world_size=world)

    if rank == 0:
        nccl = ".".join(map(str, torch.cuda.nccl.version()))
        print(f"[interconnect] torch={torch.__version__} cuda={torch.version.cuda} "
              f"nccl={nccl} world_size={world} gpus_per_node={ngpu}")
        print(f"[interconnect] device={torch.cuda.get_device_name(local)}")
        print("[interconnect] NOTE: grep the NCCL INFO lines for 'NET/IB' "
              "(InfiniBand) vs 'NET/Socket' (TCP fallback).")

    sizes = [1 * 1024**2, 4 * 1024**2, 16 * 1024**2, 64 * 1024**2,
             256 * 1024**2, 1024**3]

    # ---------------- all-reduce (all ranks) ----------------
    ar_rows = []
    for nbytes in sizes:
        x = torch.ones(nbytes // 4, dtype=torch.float32, device=dev)
        iters, warmup = iters_for(nbytes)
        t = time_op(lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM), iters, warmup)
        algbw = gbps(nbytes, t)
        busbw = algbw * 2 * (world - 1) / world
        ar_rows.append((nbytes, t, algbw, busbw))
        del x
        torch.cuda.empty_cache()

    # ---------------- p2p rank0<->rank1 (only when exactly 2 ranks) ----------
    # Guarded to world==2 so the per-op barriers inside time_op always involve
    # every rank (avoids a barrier mismatch / hang when world>2).
    p2p_rows = []
    if world == 2:
        peer = 1 - rank
        for nbytes in sizes:
            sbuf = torch.ones(nbytes // 4, dtype=torch.float32, device=dev)
            rbuf = torch.empty(nbytes // 4, dtype=torch.float32, device=dev)
            iters, warmup = iters_for(nbytes)

            def uni():
                if rank == 0:
                    dist.send(sbuf, dst=1)
                else:
                    dist.recv(rbuf, src=0)

            def bidir():
                ops = [dist.P2POp(dist.isend, sbuf, peer),
                       dist.P2POp(dist.irecv, rbuf, peer)]
                for w in dist.batch_isend_irecv(ops):
                    w.wait()

            t_uni = time_op(uni, iters, warmup)
            t_bi = time_op(bidir, iters, warmup)
            p2p_rows.append((nbytes, gbps(nbytes, t_uni), gbps(nbytes, t_bi)))
            del sbuf, rbuf
            torch.cuda.empty_cache()

    dist.barrier()

    if rank == 0:
        print("\n================= ALL-REDUCE (across all ranks) =================")
        print(f"{'size':>8} | {'time(ms)':>9} | {'algbw(GB/s)':>11} | {'busbw(GB/s)':>11}")
        for nbytes, t, algbw, busbw in ar_rows:
            print(f"{fmt_size(nbytes):>8} | {t*1e3:9.3f} | {algbw:11.2f} | {busbw:11.2f}")

        if p2p_rows:
            print("\n=========== P2P rank0<->rank1 (per-direction GB/s) ===========")
            print(f"{'size':>8} | {'send/recv':>10} | {'bidir/dir':>10}")
            for nbytes, uni, bi in p2p_rows:
                print(f"{fmt_size(nbytes):>8} | {uni:10.2f} | {bi:10.2f}")

        peak = max(r[3] for r in ar_rows)
        print(f"\n[interconnect] PEAK all-reduce busbw: {peak:.2f} GB/s "
              f"(~{peak*8:.0f} Gb/s). Compare with the IB port 'rate' above "
              f"(e.g. 200 Gb/s HDR) to see how close to line-rate NCCL gets.")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
