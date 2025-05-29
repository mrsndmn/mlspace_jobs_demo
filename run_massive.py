
import os

from mls.manager.job.utils import training_job_api_from_profile

if __name__ == "__main__":

    client, extra_options = training_job_api_from_profile('default')

    workdir = os.getcwd()

    author_name = 'd.tarasov'

    for learning_rate in [ 0.1, 0.01 ]:
        for batch_size in [ 16, 32 ]:
            result = client.run_job(
                payload={
                    'script': f"bash {workdir}/scripts/run_training_multigpu_accelerate.sh --learning_rate {learning_rate} --per_device_train_batch_size {batch_size} --output_dir ./train_demo_lr_{learning_rate}_bs_{batch_size}",
                    'job_desc': f'Massive Jobs Demo {learning_rate} {batch_size} #{author_name} #rnd #multimodal',
                    'env_variables': {
                        'WANDB_API_KEY': os.environ.get('WANDB_API_KEY'),
                    },
                    'instance_type': 'a100.2gpu',
                    'region': extra_options['region'],
                    'type': 'binary_exp',
                    'shm_size_class': 'medium',
                    'base_image': 'cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36',
                    'n_workers': 1,              # Количество воркеров.
                    'processes_per_worker': 1,   # Количество процессов на воркер. Для accelerate нужно запускать 1 процесс на воркер. Для torchrun лучше не заполнять этот параметр. По умолчанию запускается по количеству GPU на одном воркере - это подходит для torchrun.
                }
            )

            print(learning_rate, batch_size, result)

