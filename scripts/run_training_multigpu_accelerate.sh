set -ex

ENV_PREFIX=/workspace-SR004.nfs2/d.tarasov/envs/jobs_demo/bin
WORKDIR=/workspace-SR004.nfs2/d.tarasov/mlspace_jobs_demo

cd $WORKDIR
$ENV_PREFIX/python $ENV_PREFIX/accelerate launch --config_file $WORKDIR/accelerate/accelerator_config_2gpu.yaml scripts/train_demo_multigpu_accelerate.py $@
