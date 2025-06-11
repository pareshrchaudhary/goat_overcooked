#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path

import torch

from mapbt.config import get_config

from mapbt.envs.overcooked.Overcooked_Env import Overcooked
from mapbt.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

def make_train_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                env = Overcooked(all_args, run_dir)
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


def make_eval_env(all_args, run_dir):
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
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument("--old_dynamics", default=False, action='store_true', help="old_dynamics in mdp")
    parser.add_argument("--layout_name", type=str, default='cramped_room', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
    parser.add_argument('--num_agents', type=int, default=1, help="number of players")
    parser.add_argument("--initial_reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_horizon", type=int, default=2.5e6, help="Shaping factor of potential dense reward.")
    parser.add_argument("--use_phi", default=False, action='store_true', help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.")
    parser.add_argument("--use_hsp", default=False, action='store_true') 
    parser.add_argument("--random_index", default=False, action='store_true') 
    parser.add_argument("--random_start_prob", default=0., type=float, help="Probability to use a random start state, default 0.")
    parser.add_argument("--use_detailed_rew_shaping", default=False, action="store_true")
    # population
    parser.add_argument("--population_yaml_path", type=str, help="Path to yaml file that stores the population info.")
    
    # only used for overcooked evaluation! do not use here
    parser.add_argument("--agent0_policy_name", default="not_applicable", type=str, help="policy name of agent 0")
    parser.add_argument("--agent1_policy_name", default="not_applicable", type=str, help="policy name of agent 1")

    parser.add_argument("--z_dim", default=16, type=int, help="representation dimension of the partner")
    parser.add_argument("--vae_epoch", default=1000, type=int, help="number of total training epoch")
    parser.add_argument("--vae_lr", default=1e-3, type=float, help="vae learning rate")
    parser.add_argument("--vae_batch_size", default=64, type=int, help="vae minibatch of chunks")
    parser.add_argument("--evaluation_batch_size", default=1024, type=int, help="using larger batch size for evaluation")    
    parser.add_argument("--vae_init_kl_penalty", default=None, type=float, help="vae coefficient for the KL divergence term")
    parser.add_argument("--vae_kl_penalty", default=0.1, type=float, help="vae coefficient for the KL divergence term")
    parser.add_argument("--vae_chunk_length", default=None, type=int, help="chunk length of the observation used to predict the representation")
    parser.add_argument("--vae_ego_agents", default=[0], nargs="+", type=int, help="all agents that are considered as the ego agent")
    parser.add_argument("--vae_encoder_input", default="ego_obs", choices=["ego_obs", "partner_obs", "ego_share_obs"])

    parser.add_argument("--dataset_file", default="dataset.hdf5", type=str, help="path to the file that stores the data")
    parser.add_argument("--dataset_validation_ratio", default=0.1, type=float, help="number of samples splited for the validation dataset")
    parser.add_argument("--vae_model_dir", default=None, type=str, help="vae model path")
    parser.add_argument("--vae_model_name", default=None, type=str, help="vae model name")
    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    assert all_args.algorithm_name == "population"

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.layout_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        assert False, "to render! no wandb here"
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args, run_dir)
    eval_envs = make_eval_env(all_args, run_dir) if all_args.use_eval else None
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
        from mapbt.scripts.overcooked_population.vae_constructor.shared_runner import OvercookedRunner as Runner
    else:
        raise NotImplementedError

    runner = Runner(config)

    # load population
    runner.policy.load_population(all_args.population_yaml_path, evaluation=True)

    runner._render_data_support_vae()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])
