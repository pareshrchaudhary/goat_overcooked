#!/bin/bash
env="Overcooked"

# Notes:
# Some layout_name and human_layout_name are different, lile:
#   layout=counter_circuit_o_1order
#   human_layout=random3
# Pay attention to: --old_dynamics; --human_data_refresh
layout=asymmetric_advantages
human_layout=asymmetric_advantages

num_agents=2
algo="bc"
exp="bc"
seed_max=1

path=./overcooekd_population
export POLICY_POOL=${path}

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python train/train_overcooked_bc.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
     --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --reward_shaping_horizon 0 \
     --ppo_epoch 15 \
     --save_interval 25 --log_inerval 10 --use_recurrent_policy \
     --old_dynamics \
     --human_data_refresh \
     --bc_num_epochs 100 --bc_batch_size 128 --lr 1e-2 \
     --use_eval --eval_stochastic --eval_interval 25 --eval_episodes 5 \
     --human_layout_name ${human_layout} \
     --wandb_name "WANDB_NAME" --user_name "USER_NAME" --use_wandb
done