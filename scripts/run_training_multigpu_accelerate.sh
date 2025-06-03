set -ex

ENV_PREFIX=/workspace-SR004.nfs2/d.tarasov/envs/jobs_demo/bin
WORKDIR=/workspace-SR004.nfs2/d.tarasov/mlspace_jobs_demo

# Добавим в PATH префикс окружения
PATH=$ENV_PREFIX:$PATH

cd $WORKDIR
accelerate launch --config_file $WORKDIR/accelerate/accelerator_config_2gpu.yaml scripts/train_demo_multigpu_accelerate.py $@
