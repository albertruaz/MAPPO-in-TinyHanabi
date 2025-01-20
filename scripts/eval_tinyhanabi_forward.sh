#!/bin/sh

# 환경 설정
env="TinyHanabi"
hanabi="TinyHanabi-Example"
num_agents=2
algo="mappo"
exp="eval_test"
seed_max=10
ulimit -n 22222  # 파일 디스크립터 수 제한 해제

base_dir="./results/TinyHanabi/TinyHanabi/mappo/check1/wandb"

# wandb 폴더에서 run- 으로 시작하는 디렉토리들을 찾아서 배열에 저장 (각 경로에 /files 추가)
readarray -t model_dirs < <(ls -1 "$base_dir" | grep "^run-" | sed 's|$|/files|' | head -n $seed_max)

echo "Environment: ${env}, Algorithm: ${algo}, Experiment: ${exp}, Max seed: ${seed_max}"
echo "Found following model directories:"
printf '%s\n' "${model_dirs[@]}"

# 시드 반복 실행
for seed in `seq 1 ${seed_max}`; do
    echo "Current seed: ${seed}"
    
    # 배열 인덱스는 0부터 시작하므로 seed-1
    model_dir="${base_dir}/${model_dirs[$((seed-1))]}"
    echo "Using model directory: ${model_dir}"

    # 평가 실행
    CUDA_VISIBLE_DEVICES=0 python eval/eval_tinyhanabi.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --hanabi_name ${hanabi} \
    --num_agents ${num_agents} \
    --seed ${seed} \
    --n_training_threads 128 \
    --n_rollout_threads 1 \
    --n_eval_rollout_threads 100 \
    --episode_length 100 \
    --num_env_steps 1000000 \
    --ppo_epoch 15 \
    --gain 0.01 \
    --lr 7e-4 \
    --critic_lr 1e-3 \
    --hidden_size 32 \
    --layer_N 2 \
    --use_eval \
    --use_recurrent_policy \
    --entropy_coef 0.015 \
    --model_dir "${model_dir}"
    
    echo "Evaluation for seed ${seed} is done!"
done

    #512 