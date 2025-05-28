import os

from mls.manager.job.utils import training_job_api_from_profile

if __name__ == "__main__":

    client = training_job_api_from_profile('default')

    for learning_rate in [ 0.1, 0.01 ]:
        for batch_size in [ 16, 32 ]:
            result = client.run_job(
                payload={
                    'script': f"bash samples/run_training.sh --learning_rate {learning_rate} --batch_size {batch_size} --output_dir ./train_demo_lr_{learning_rate}_bs_{batch_size}",
                    'job_desc': 'Massive Jobs Demo #author_name #rnd #multimodal',
                    'env_variables': {
                        'WANDB_API_KEY': os.environ.get('WANDB_API_KEY'),
                    },
                    'instance_type': 'a100.1gpu',
                    'region': 'SR004',
                    'type': 'binary_exp',
                    'shm_size_class': 'medium',
                    'base_image': 'cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36',
                    'n_workers': 1,              # Не трогайте эти параметры! Они ничего не ускорят, а только скорее могут сломать Multi-GPU обучение. Изменяйте их только если вы точно понимаете, что делаете
                    'processes_per_worker': 1,   # Не трогайте эти параметры! Они ничего не ускорят, а только скорее могут сломать Multi-GPU обучение. Изменяйте их только если вы точно понимаете, что делаете
                }
            )

            print(learning_rate, batch_size, result)