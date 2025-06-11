#!/bin/bash
env="Overcooked"

# asymmetric_advantages coordination_ring counter_circuit_o_1order=random3, cramped_room forced_coordination=random0, diverse_counter_circuit_6x5
layout=diverse_counter_circuit_6x5 
pop=${layout}_mep

num_agents=2
algo="adaptive"
exp="render_regret" 

run_path=./results/Overcooked/diverse_counter_circuit_6x5/adaptive/vae_sp_vs_minimax_regret_kl_46_mean_clip_max_0.2_min_0.2/seed4
path=./overcooked_population
population_yaml_path=${path}/pop_data/${pop}/zsc_config.yml
vae_model_dir=${path}/pop_data/${pop}/vae_models/best_logp_kl_46 # 32, 46, 68, 100 

export POLICY_POOL=${path}

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in $2
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=$1 python overcooked_population/render_vae_sp.py \
        --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
        --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --episode_length 400 \
        --ppo_epoch 15 --reward_shaping_horizon 0 \
        --n_rollout_threads 1 --train_env_batch 1 \
        --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" \
        --population_yaml_path ${population_yaml_path} \
        --population_size 32 --adaptive_agent_name coordinator --use_agent_policy_id \
        --vae_hidden_size 256 --vae_encoder_input partner_obs \
        --vae_model_dir "${vae_model_dir}" \
        --vae_z_change_prob 0 \
        --adversary_learning_rate 0.0005 \
        --use_std_scaling --scale_log_std 4.0 --scale_std_offset 1.0 \
        --use_render --eval_stochastic --old_dynamics  \
        --render_episodes 1 \
        --run_path "${run_path}" \
        --wandb_name "social-rl" --user_name "USER_NAME" --use_wandb
done