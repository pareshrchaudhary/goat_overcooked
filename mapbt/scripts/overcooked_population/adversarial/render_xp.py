import itertools
import os
import time
import warnings
from collections import defaultdict
from copy import deepcopy
import json
import pickle

import numpy as np
import torch
import wandb
import gc
import os
import glob

from mapbt.runner.shared.overcooked_runner import OvercookedRunner as Runner
from ..vae_constructor.vae_model import VAEModel
from .vae_agent import VAEAgent
from .regret_z_generator import get_z_generator

class OvercookedRunner(Runner):
    def adversary_vs_coordinator(self, reset_map_ea2t_ea2z_fn = None):   
        total_num_steps = 0
        env_infos = defaultdict(list)
        self.env_info = dict()

        for episode in range(self.all_args.render_episodes): 
            print(f"Episode {episode + 1}/{self.all_args.render_episodes}")
            self.total_num_steps = total_num_steps

            # reset env agents
            if reset_map_ea2t_ea2z_fn is not None:
                map_ea2t, map_ea2z, adversary_z, adversary_z_old, adversary_z_log_prob, z_current, z_prior = reset_map_ea2t_ea2z_fn(episode)
                self.trainer.reset(map_ea2t, self.n_rollout_threads, self.num_agents, n_repeats=None, load_unused_to_cpu=True)

            ##### MINIMAX #####
            obs, share_obs, available_actions = self.envs.reset()
            if self.use_centralized_V:
                share_obs = share_obs
            else:
                share_obs = obs

            self.trainer.init_first_step(share_obs, obs)
            self.vae_agent.init_first_step(share_obs, obs)

            episode_reward = torch.zeros(self.n_rollout_threads, device=self.device)
            for step in range(self.episode_length):
                actions = self.trainer.step(step)
                vae_actions = self.vae_agent.step(step, map_ea2z[step])
                for (e, a) in vae_actions:
                    actions[e][a] = vae_actions[(e, a)]
                    
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                total_num_steps += self.n_rollout_threads
                # self.envs.anneal_reward_shaping_factor(self.trainer.reward_shaping_steps())

                self.trainer.insert_data(share_obs, obs, rewards, dones, infos=infos)
                self.vae_agent.insert_data(share_obs, obs)
                episode_reward += torch.tensor([r.mean() for r in rewards], device=self.device)
            ##### MINIMAX #####
            print("episode reward: ", episode_reward)

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            episode_env_infos = defaultdict(list)
            if self.env_name == "Overcooked":
                for e, info in enumerate(infos):
                    agent0_trainer = self.trainer.map_ea2t.get((e, 0), "vae")
                    agent1_trainer = self.trainer.map_ea2t.get((e, 1), "vae")
                    for log_name in [f"{agent0_trainer}-{agent1_trainer}"]:
                        episode_env_infos[f'{log_name}-ep_sparse_r'].append(info['episode']['ep_sparse_r'])

                env_infos.update(episode_env_infos)
            self.env_info.update(env_infos)
            
            import sys
            sys.stdout.flush()

    @torch.no_grad()
    def render_episodes(self):
        print("Rendering episodes...")
        print("run_path", self.all_args.run_path)

        # Load VAE model
        self.vae_model = VAEModel(*self.policy_config, device=self.device)
        self.vae_agent = VAEAgent(self.all_args, self.vae_model)

        # Checkpoint Paths
        files_path = os.path.join(self.run_dir, "files")
        adversary_paths = glob.glob(os.path.join(files_path, "adversary/adversary_*.pt"))
        actor_paths = glob.glob(os.path.join(files_path, "coordinator/actor_periodic_*.pt"))
        critic_paths = glob.glob(os.path.join(files_path, "coordinator/critic_periodic_*.pt"))
        
        adversary_steps = {int(f.split('_')[-1].split('.')[0]): f for f in adversary_paths}
        actor_steps = {int(f.split('periodic_')[-1].split('.')[0]): f for f in actor_paths}
        critic_steps = {int(f.split('periodic_')[-1].split('.')[0]): f for f in critic_paths}

        matched_steps = sorted(set(adversary_steps.keys()) & set(actor_steps.keys()) & set(critic_steps.keys()))
        
        assert self.all_args.population_size == len(self.trainer.population)
        self.population_size = self.all_args.population_size
        self.population = sorted(self.trainer.population.keys()) # Note index and trainer name would not match when there are >= 10 agents

        # Stage 2: train an agent against population with prioritized sampling
        agent_name = self.trainer.agent_name
        assert (self.n_eval_rollout_threads % self.all_args.eval_env_batch == 0 and self.all_args.eval_episodes % self.all_args.eval_env_batch == 0)
        assert self.n_rollout_threads % self.all_args.train_env_batch == 0
        self.all_args.eval_episodes = self.all_args.eval_episodes * self.n_eval_rollout_threads // self.all_args.eval_env_batch
        self.eval_idx = 0
        all_agent_pairs = list(itertools.product(self.population, [agent_name])) + list(itertools.product([agent_name], self.population))
        
        print("matched_steps", matched_steps)
        for step in matched_steps:
            self.z_gen = get_z_generator(self.all_args, self.vae_model, device=self.device, run_dir=files_path)
            self.z_gen.restore(steps=step)
            self.policy.policy_pool['coordinator'].policy.actor.load_state_dict(torch.load(os.path.join(self.run_dir, "files/coordinator", "actor_periodic_{}.pt".format(step)), weights_only=True))
            # print("Model parameters:", list(self.policy.policy_pool['coordinator'].policy.actor.parameters())[0])
            
            print(f"Processing step {step} - Total progress {matched_steps.index(step) + 1}/{len(matched_steps)}")

            def vae_reset_map_ea2t_ea2z_fn(episode):
                map_ea2t = {(e, e % 2): agent_name for e in range(self.n_rollout_threads)}
                map_ea2z = {}
                step_map_ea2z = {}
                adversary_z, adversary_z_old, adversary_log_prob, z_current, z_prior = self.z_gen.get_z()

                for e in range(self.n_rollout_threads):
                    for a in range(self.num_agents):
                        if (e, a) not in map_ea2t:
                            step_map_ea2z[(e, a)] = adversary_z[e].cpu().numpy()
                
                map_ea2z = {t: deepcopy(step_map_ea2z) for t in range(self.episode_length + 1)}
                return map_ea2t, map_ea2z, adversary_z, adversary_z_old, adversary_log_prob, z_current, z_prior
            
            self.adversary_vs_coordinator(reset_map_ea2t_ea2z_fn=vae_reset_map_ea2t_ea2z_fn)
