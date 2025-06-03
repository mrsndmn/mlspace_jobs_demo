#! /bin/bash

set -ex

ENV_PREFIX=/workspace-SR004.nfs2/d.tarasov/envs/jobs_demo/bin
WORKDIR=/workspace-SR004.nfs2/d.tarasov/mlspace_jobs_demo

# Добавим в PATH префикс окружения
PATH=$ENV_PREFIX:$PATH

# Джобы запускаются под управлением MPI
# Поэтому нам нужно получить имя мастер-ноды
# Чтобы сохранить его в переменные окружения,
# с которыми работает DDP
MASTER_HOST_PREFIX=$(perl -E "my \$x = '$PMIX_HOSTNAME'; \$x =~ s/-\w+-\d+$//; print \$x ")
MASTER_HOST=$(perl -E "my \$x = '$PMIX_HOSTNAME'; \$x =~ s/-\w+-\d+$/-mpimaster-0/; print \$x ")

MASTER_HOST_FULL="$MASTER_HOST.$MASTER_HOST_PREFIX"
echo "MASTER_HOST_FULL $MASTER_HOST_FULL"

# Сохраняем в переменные окружения,
# с которыми работает DDP
export MASTER_ADDR=$MASTER_HOST_FULL
export MASTER_PORT=12345
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export RANK=$OMPI_COMM_WORLD_RANK

# Запускаем обучение
cd $WORKDIR
python scripts/train_demo_multinode_torchddp.py $@
