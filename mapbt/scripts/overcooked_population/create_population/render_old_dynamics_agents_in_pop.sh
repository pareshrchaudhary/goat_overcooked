#!/bin/bash
env="Overcooked"

exp=$1
layout=$2
model_dir=$3

num_agents=2
algo="mappo"
seed_max=1

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python render/render_overcooked_sp.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --episode_length 400 \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
    --use_render --render_episodes 1 --eval_stochastic \
    --old_dynamics \
    --wandb_name "WANDB_NAME" --user_name "USER_NAME" --use_wandb \
    --model_dir ${model_dir}
done
