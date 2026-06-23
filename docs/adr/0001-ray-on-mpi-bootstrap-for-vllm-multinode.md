# Bootstrap a Ray cluster from the MPI environment for vLLM multinode

MLS multinode jobs launch under MPI/PMIX, and the existing DDP scripts coordinate workers
through `MASTER_ADDR`/`RANK`/`WORLD_SIZE` env vars (`scripts/run_training_multinode.sh`).
vLLM multinode does **not** use that path — it needs a single Ray cluster spanning the nodes.
We therefore reuse only the MPI **host discovery** (master host from `$PMIX_HOSTNAME`) and the
**rank** (`OMPI_COMM_WORLD_RANK`) to elect a head, then `ray start --head` on rank 0 and
`ray start --address=HEAD:6379 --block` on the others, and run `vllm serve` once on the head with
`--distributed-executor-backend ray --pipeline-parallel-size 2 --tensor-parallel-size 1`.

This is hard to reverse (the whole launcher is built around it), surprising (a reader seeing the
MPI launch would expect torchrun/DDP, not Ray), and a real trade-off: Ray is vLLM's supported
multinode backend, so we accept running one process per node (`processes_per_worker: 1`) instead
of MPI's default one-process-per-GPU. We chose pipeline parallelism across nodes (TP within a node
only) because cross-node tensor parallelism is communication-bound; with 1 GPU/node that means
PP=2, TP=1.
