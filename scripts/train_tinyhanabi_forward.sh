#!/bin/sh
env="TinyHanabi"
hanabi="TinyHanabi"
algo="mappo"
exp="check"
seed_max=1
ulimit -n 22222

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

# for seed in `seq ${seed_max}`;
# do
#     echo "seed is ${seed}:"
#     CUDA_VISIBLE_DEVICES=0 python train/train_tinyhanabi_forward.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#     --hanabi_name ${hanabi} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 \
#     --num_mini_batch 1 --num_env_steps 100000 --ppo_epoch 15 \
#     --gain 0.01 --lr 7e-4 --critic_lr 1e-3 --hidden_size 512 --layer_N 2 --entropy_coef 0.015 \
#     --log_interval 100 --save_interval 100 \
#     echo "training is done!"
# done
d
# num_env_steps 100000 to 1000
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_tinyhanabi_forward.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --hanabi_name ${hanabi} --seed ${seed} --n_training_threads 1 --n_rollout_threads 128 \
    --num_mini_batch 1 --num_env_steps 100000 --ppo_epoch 10 \
    --gain 0.01 --lr 5e-4 --critic_lr 5e-4 --hidden_size 32 --layer_N 2 --entropy_coef 0.015 \
    --log_interval 100 --save_interval 100 \
    echo "training is done!"
done