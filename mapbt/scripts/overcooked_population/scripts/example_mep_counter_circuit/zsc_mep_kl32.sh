#!/bin/bash
env="Overcooked"

# asymmetric_advantages coordination_ring counter_circuit_o_1order=random3, cramped_room forced_coordination=random0
layout=counter_circuit_o_1order # asymmetric_advantages diverse_counter_circuit_6x5
pop=${layout}_mep

num_agents=2
algo="adaptive"
exp="${0##*/}"

path=./overcooked_population
population_yaml_path=${path}/pop_data/${pop}/zsc_config.yml
vae_model_dir=${path}/pop_data/${pop}/vae_models/best_logp_kl_32

export POLICY_POOL=${path}


echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in $2
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=$1 python overcooked_population/train_coordinator_vs_vae.py \
        --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
        --seed ${seed} --n_training_threads 1 --num_mini_batch 1 --episode_length 400 --num_env_steps 150000000 \
        --ppo_epoch 15 --reward_shaping_horizon 100000000 \
        --n_rollout_threads 200 --train_env_batch 1 \
        --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" \
        --stage 2 --save_interval 20 --log_interval 10 \
        --use_eval --n_eval_rollout_threads 24 --eval_interval 25 --eval_episodes 24 --eval_stochastic --use_evaluation_agents --eval_on_old_dynamics \
        --save_interval 25 \
        --population_yaml_path ${population_yaml_path} \
        --population_size 32 --adaptive_agent_name coordinator --use_agent_policy_id \
        --vae_hidden_size 256 --vae_encoder_input partner_obs \
        --vae_model_dir "${vae_model_dir}" \
        --vae_z_change_prob 0 \
        --wandb_name "WANDB_NAME" --user_name "USER_NAME" --use_wandb # --wandb_name $(cat ~/.config/wandb/overcooked_org) --user_name $(cat ~/.config/wandb/overcooked_id)
done
