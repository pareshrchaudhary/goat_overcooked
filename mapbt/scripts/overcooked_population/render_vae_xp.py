#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import torch
import yaml
import numpy as np
from argparse import Namespace
from pathlib import Path
from datetime import datetime

from mapbt.config import get_config

from mapbt.envs.overcooked.Overcooked_Env import Overcooked
from mapbt.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, ChooseSubprocVecEnv, ChooseDummyVecEnv
from mapbt.envs.wrappers.env_policy import PartialPolicyEnv

def make_train_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                env = Overcooked(all_args, run_dir)
                env = PartialPolicyEnv(all_args, env)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(_all_args, run_dir):
    from copy import deepcopy
    all_args = deepcopy(_all_args)
    if all_args.eval_on_old_dynamics:
        all_args.old_dynamics = True
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                env = Overcooked(all_args, run_dir)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ChooseSubprocVecEnv([get_env_fn(0)])
    else:
        return ChooseDummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument("--old_dynamics", default=False, action='store_true', help="old_dynamics in mdp")
    parser.add_argument("--eval_on_old_dynamics", default=False, action='store_true', help="old_dynamics in mdp for evaluation environment; useful for BC policy and human player")
    parser.add_argument("--layout_name", type=str, default='cramped_room', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
    parser.add_argument('--num_agents', type=int, default=1, help="number of players")
    parser.add_argument("--initial_reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_horizon", type=int, default=2.5e6, help="Shaping factor of potential dense reward.")
    parser.add_argument("--use_phi", default=False, action='store_true', help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.")
    parser.add_argument("--use_hsp", default=False, action='store_true')   
    parser.add_argument("--random_index", default=False, action='store_true')
    parser.add_argument("--use_agent_policy_id", default=False, action='store_true', help="Add policy id into share obs, default False")
    parser.add_argument("--predict_other_shaped_info", default=False, action='store_true', help="Predict other agent's shaped info within a short horizon, default False")
    parser.add_argument("--predict_shaped_info_horizon", default=50, type=int, help="Horizon for shaped info target, default 50")
    parser.add_argument("--predict_shaped_info_event_count", default=10, type=int, help="Event count for shaped info target, default 10")
    parser.add_argument("--shaped_info_coef", default=0.5, type=float)
    parser.add_argument("--policy_group_normalization", default=False, action="store_true")
    parser.add_argument("--use_advantage_prioritized_sampling", default=False, action='store_true')
    parser.add_argument("--uniform_preference", default=False, action='store_true')
    parser.add_argument("--uniform_sampling_repeat", default=0, type=int)
    parser.add_argument("--use_task_v_out", default=False, action='store_true')
    parser.add_argument("--random_start_prob", default=0., type=float, help="Probability to use a random start state, default 0.")
    parser.add_argument("--project_name", type=str, default="overcooked_adversary", help="Project name for wandb.")
    
    # population
    parser.add_argument("--population_yaml_path", type=str, help="Path to yaml file that stores the population info.")

    # mep
    parser.add_argument("--stage", type=int, default=1 ,help="Stages of MEP training. 1 for Maximum-Entropy PBT. 2 for FCP-like training.")
    parser.add_argument("--mep_use_prioritized_sampling", default=False, action='store_true', help="Use prioritized sampling in MEP stage 2.")
    parser.add_argument("--mep_prioritized_alpha", type=float, default=3.0, help="Alpha used in softing prioritized sampling probability.")
    parser.add_argument("--mep_entropy_alpha", type=float, default=0.01, help="Weight for population entropy reward. MEP uses 0.01 in general except 0.04 for Forced Coordination")
    # population
    parser.add_argument("--population_size", type=int, default=5, help="Population size involved in training.")
    parser.add_argument("--adaptive_agent_name", type=str, required=True, help="Name of training policy at Stage 2.")
    
    # train and eval batching
    parser.add_argument("--train_env_batch", type=int, default=1, help="Number of parallel threads a policy holds")
    parser.add_argument("--eval_env_batch", type=int, default=1, help="Number of parallel threads a policy holds")

    # eval with fixed policy
    parser.add_argument("--eval_policy", default="", type=str)
    parser.add_argument("--use_detailed_rew_shaping", default=False, action="store_true")
    parser.add_argument("--use_evaluation_agents", default=False, action='store_true', help="Evaluate with some given agents.")

    # vae arguments
    parser.add_argument("--z_dim", default=16, type=int, help="representation dimension of the partner")
    parser.add_argument("--vae_hidden_size", type=int, default=-1, help="number of hidden nodes in VAE model")
    parser.add_argument("--vae_chunk_length", default=None, type=int, help="chunk length of the observation used to predict the representation")
    parser.add_argument("--vae_encoder_input", default="ego_obs", choices=["ego_obs", "partner_obs", "ego_share_obs"])
    parser.add_argument("--vae_model_dir", type=str, default=None, help="by default None. set the path to pretrained vae model.")
    
    # z generator to train the coordinator
    parser.add_argument("--vae_z_generator", type=str, default="adversarial", choices=["normal_gussian", "human_biased", "human_biased_std", "adversarial"], help="the type of the z generator")
    parser.add_argument("--vae_z_change_prob", type=float, default=0, help="probability that the partner change strategy during the episode at each step")
    parser.add_argument("--dataset_file", default="dataset.hdf5", type=str, help="path to the file that stores the data")

    # adversary network parameters
    parser.add_argument("--hidden_dim", default=128, type=int, help="number of hidden nodes in the adversary model")
    parser.add_argument("--use_mean_clipping", default=False, action='store_true', help="use mean scaling")
    parser.add_argument("--clip_mean_min", default=-0.5, type=float, help="clipping parameter for mean min")
    parser.add_argument("--clip_mean_max", default=0.5, type=float, help="clipping parameter for mean max")
    parser.add_argument("--use_std_scaling", default=False, action='store_true', help="use std scaling")
    parser.add_argument("--scale_log_std", default=4.0, type=float, help="scaling parameter for log_std")
    parser.add_argument("--scale_std_offset", default=1.0, type=float, help="offset for std")

    # adversary loss parameters
    parser.add_argument("--clip_epsilon", default=0.2, type=float, help="PPO Clip")
    parser.add_argument("--n_epochs", default=15, type=int, help="number of epochs for ppo")
    parser.add_argument("--target_kl", default=0.02, type=float, help="target kl for ppo")
    parser.add_argument("--use_kl", default=False, action='store_true', help="use kl constraint in the training")
    parser.add_argument("--kl_coeff", default=0.001, type=float, help="beta for kl constraint")
    parser.add_argument("--use_entropy_bonus", default=False, action='store_true', help="use entropy bonus in the training")
    parser.add_argument("--entropy_coeff", default=0.01, type=float, help="alpha for entropy bonus")

    # advesary optimizer parameters
    parser.add_argument("--adversary_learning_rate" , default=0.001, type=float, help="learning rate of the z generator")
    parser.add_argument("--adversary_weight_decay", default=0.0001, type=float, help="weight decay of the z generator")
    parser.add_argument("--use_grad_clip", default=False, action='store_true', help="use gradient clip in the training")
    parser.add_argument("--grad_clip_norm", default=0.5, type=float, help="norm for gradient clip")

    # lr scheduler parameters 
    parser.add_argument("--use_lr_scheduler", default=False, action='store_true', help="use lr scheduler")
    parser.add_argument("--start_factor", default=1.0, type=float, help="start factor of the lr scheduler")
    parser.add_argument("--end_factor", default=0.9, type=float, help="end factor of the lr scheduler")
    
    # render
    parser.add_argument("--run_path", default="", type=str, help="the path of the run")

    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    assert all_args.algorithm_name == "adaptive"

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(all_args.run_path) if all_args.run_path else Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.layout_name / all_args.algorithm_name / all_args.experiment_name

    if all_args.use_wandb:
        assert False, "to render! no wandb here"
    else:
        if not run_dir.exists():
            raise ValueError(f"Run directory {run_dir} does not exist")

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    torch.use_deterministic_algorithms(True)

    # env init
    envs = make_train_env(all_args, run_dir)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from mapbt.scripts.overcooked_population.adversarial.render_xp import OvercookedRunner as Runner
    else:
        raise NotImplementedError

    runner = Runner(config)
    
    # load population
    # print("population_yaml_path: ",all_args.population_yaml_path)

    # override policy config
    population_config = yaml.load(open(all_args.population_yaml_path), yaml.Loader)
    override_policy_config = {}
    agent_name = all_args.adaptive_agent_name
    override_policy_config[agent_name] = (Namespace(use_agent_policy_id=all_args.use_agent_policy_id, 
                                                    predict_other_shaped_info=all_args.predict_other_shaped_info,
                                                    predict_shaped_info_horizon=all_args.predict_shaped_info_horizon,
                                                    predict_shaped_info_event_count=all_args.predict_shaped_info_event_count,
                                                    shaped_info_coef=all_args.shaped_info_coef,
                                                    policy_group_normalization=all_args.policy_group_normalization,
                                                    num_v_out=all_args.num_v_out,
                                                    use_task_v_out=all_args.use_task_v_out,
                                                    use_policy_vhead=all_args.use_policy_vhead), 
                                                    *runner.policy_config[1:])
    for policy_name in population_config:
        if policy_name != agent_name:
            override_policy_config[policy_name] = (None, None, runner.policy_config[2], None) # only override share_obs_space

    runner.policy.load_population(all_args.population_yaml_path, evaluation=False, override_policy_config=override_policy_config)
    runner.trainer.init_population()

    print("runner.policy.population: ", runner.policy.policy_pool)
    runner.render_episodes()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])
