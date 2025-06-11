#!/bin/bash
env="Overcooked"

# asymmetric_advantages coordination_ring counter_circuit_o_1order=random3, cramped_room forced_coordination=random0
layout=$2 # asymmetric_advantages diverse_counter_circuit_6x5
pop=${layout}_comedi

num_agents=2
algo="population"
agent0_policy_name="comedi_oracle"
agent1_policy_name="proxy"
exp="eval-${agent0_policy_name}-${agent1_policy_name}"

path=./overcooked_population
population_yaml_path=${path}/pop_data/${pop}/comedi_oracle_vs_proxy.yml

export POLICY_POOL=${path}

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
CUDA_VISIBLE_DEVICES=$1 python eval/eval_overcooked.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} \
--user_name "user_name" --num_agents ${num_agents} --seed 1 --episode_length 400 --n_eval_rollout_threads 3 --eval_episodes 3 --eval_stochastic \
--wandb_name "wandb_name" --use_wandb \
--population_yaml_path ${population_yaml_path} \
--agent0_policy_name ${agent0_policy_name} \
--agent1_policy_name ${agent1_policy_name} \
--old_dynamics 
