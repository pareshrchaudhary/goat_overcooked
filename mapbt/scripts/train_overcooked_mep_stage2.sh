
#!/bin/bash
env="Overcooked"

# unident_s, random1, random3, distant_tomato, many_orders
layout=asymmetric_advantages


num_agents=2
algo="mep"
exp="mep"
stage="S2"
seed=1
path=./overcooked_population

export POLICY_POOL=${path}

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, seed is ${seed}, stage is ${stage}"
CUDA_VISIBLE_DEVICES=2 python train/train_overcooked_adaptive.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}-${stage}" --layout_name ${layout} --num_agents ${num_agents} \
--seed 1 --n_training_threads 1 --num_mini_batch 1 --episode_length 400 --num_env_steps 100000000 \
--ppo_epoch 15 --reward_shaping_horizon 100000000 \
--n_rollout_threads 200 --train_env_batch 1 \
--stage 2 --save_interval 20 --log_interval 10 \
--population_yaml_path ${path}/config/${layout}/mep/s2/train.yml \
--population_size 36 --adaptive_agent_name mep_adaptive --use_agent_policy_id \
--old_dynamics \
--wandb_name "WANDB_NAME" --user_name "USER_NAME" --use_wandb
