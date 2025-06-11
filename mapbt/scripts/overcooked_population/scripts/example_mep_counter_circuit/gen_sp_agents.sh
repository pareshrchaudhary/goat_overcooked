#!/bin/bash
env="Overcooked"

# asymmetric_advantages coordination_ring counter_circuit_o_1order=random3, cramped_room forced_coordination=random0
layout=$2 # asymmetric_advantages diverse_counter_circuit_6x5

num_agents=2
algo="mappo"
reward_command="(step, potting_onion, both, accumulate, 5), (step, potting_tomato, both, accumulate, 5)"
exp="${0##*/}"
seed_max=10

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=$1 python overcooked_population/self_play_reward_shaping.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
     --seed ${seed} --n_training_threads 1 --n_rollout_threads 100 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --reward_shaping_horizon 100000000 \
     --ppo_epoch 15 \
     --random_start_prob 1.0 \
     --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --save_interval 25 --log_inerval 10 --use_recurrent_policy \
     --trainer_reward_shaping_command "${reward_command}" \
     --wandb_name "WANDB_NAME" --user_name "USER_NAME" --use_wandb
done
