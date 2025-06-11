#!/bin/bash
env="Overcooked"

# asymmetric_advantages coordination_ring counter_circuit_o_1order=random3, cramped_room forced_coordination=random0
layout=counter_circuit_o_1order # asymmetric_advantages diverse_counter_circuit_6x5

num_agents=2
algo="population"
pop="${layout}_fcp" 
exp="${0##*/}"
seed_max=1

path=./overcooked_population
population_yaml_path=${path}/pop_data/${pop}/zsc_config.yml
vae_model_dir=${path}/pop_data/${pop}/vae_models/finetune_decoder_epoch_1

export POLICY_POOL=${path}

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python overcooked_population/render_vae.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name "${layout}_bc1" --num_agents ${num_agents} \
        --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --episode_length 400 --reward_shaping_horizon 0 --render_episodes 1 \
        --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --log_inerval 1 \
        --population_yaml_path ${population_yaml_path} \
        --log_interval 1 --save_interval 20 --use_eval --eval_interval 10 \
        --use_render --eval_stochastic --old_dynamics \
        --hidden_size 256 --weight_decay 1e-4 --vae_chunk_length 100 \
        --vae_ego_agents 0 1 --vae_encoder_input partner_obs \
        --vae_model_dir ${vae_model_dir} --dataset_file "../../../overcooked_dataset/${layout}_human/dataset.hdf5" --use_wandb
done
