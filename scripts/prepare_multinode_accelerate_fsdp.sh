set -x
set -e

ENV_PREFIX=/workspace-SR004.nfs2/d.tarasov/envs/tokens_pruning/bin
WORKDIR=/workspace-SR004.nfs2/d.tarasov/sentence_attention


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

JOBS_TMP_PROC="/workspace-SR004.nfs2/d.tarasov/jobs_tmp_proc/$MASTER_HOST"
echo "JOBS_TMP_PROC $JOBS_TMP_PROC"
mkdir -p $JOBS_TMP_PROC

# Сохраняем в переменные окружения,
# с которыми работает DDP
export MASTER_ADDR=$MASTER_HOST_FULL
export MASTER_PORT=12345
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export RANK=$OMPI_COMM_WORLD_RANK

NUM_GPUS=$(nvidia-smi -L | nvidia-smi -L | grep -c "GPU")

echo "NUM_GPUS $NUM_GPUS"
echo "WORLD_SIZE $WORLD_SIZE"
echo "RANK $RANK"
echo "MASTER_ADDR $MASTER_ADDR"

cd $WORKDIR

# CURRENT_CONFIG_FILE=$JOBS_TMP_PROC/accelerate_${RANK}.yaml

# echo "Copying multinode-template.yaml to $CURRENT_CONFIG_FILE"
# ls -l ./configs/accelerate/multinode-template.yaml

# echo "Replacing TODO with $RANK in $CURRENT_CONFIG_FILE"
# sed -i "s/machine_rank: TODO/machine_rank: $RANK/g" $CURRENT_CONFIG_FILE
# sed -i "s/num_processes: TODO/num_processes: $NUM_GPUS/g" $CURRENT_CONFIG_FILE
# sed -i "s/num_machines: TODO/num_machines: $WORLD_SIZE/g" $CURRENT_CONFIG_FILE
# sed -i "s/main_process_ip: TODO/main_process_ip: $MASTER_ADDR/g" $CURRENT_CONFIG_FILE
# sed -i "s/main_process_port: TODO/main_process_port: $MASTER_PORT/g" $CURRENT_CONFIG_FILE

${ENV_PREFIX}/python ${ENV_PREFIX}/accelerate launch \
    --config_file ./configs/accelerate/multinode-template-fsdp.yaml \
    --machine_rank $RANK \
    --num_processes $(($NUM_GPUS * $WORLD_SIZE)) \
    --num_machines $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    $@
