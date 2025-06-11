#!/bin/bash
env="Overcooked"

# Notes:
# Pay attention to: --old_dynamics
layout=asymmetric_advantages

num_agents=2
algo="bc"
exp="bc_render"
seed_max=1

path=./overcooekd_population
export POLICY_POOL=${path}

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python render/render_overcooked_bc.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
     --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --episode_length 400 \
     --use_recurrent_policy \
     --old_dynamics \
     --use_render --render_episodes 1 --eval_stochastic \
     --wandb_name "WANDB_NAME" --user_name "USER_NAME" --use_wandb \
     --model_dir "model_dir"
done
