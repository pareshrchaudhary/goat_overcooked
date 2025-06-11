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

from mapbt.runner.shared.overcooked_runner import OvercookedRunner as Runner
from ..vae_constructor.vae_model import VAEModel
from .vae_agent import VAEAgent
from .z_generator import get_z_generator
from .viz import z_distribution, print_env_stats, print_training_stats

class OvercookedRunner(Runner):
    
    def vae_vs_vae(self):
        self.vae_model = VAEModel(*self.policy_config, device=self.device)
        self.vae_agent = VAEAgent(self.all_args, self.vae_model)
        self.z_gen = get_z_generator(self.all_args, self.vae_model, device=self.device, run_dir=self.run_dir)

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0
        env_info = defaultdict(list)
        episode_avg = []
        best_reward = -float('inf')
        best_z = None
        env_info['total_num_steps'] = 0
        
        self.z_gen.lr_scheduler(episodes)

        for episode in range(episodes):
            obs, share_obs, available_actions = self.envs.reset()
            self.vae_agent.init_first_step(share_obs, obs)

            z_sample_batch, z_old, log_prob_batch = self.z_gen.get_z()

            map_ea2z = {}
            for e in range(self.n_rollout_threads):
                map_ea2z[(e, 0)] = z_sample_batch[e].detach().cpu().numpy()
                map_ea2z[(e, 1)] = z_sample_batch[e].detach().cpu().numpy()
            
            episode_reward = np.zeros(self.n_rollout_threads)
            for step in range(self.episode_length):
                vae_actions = self.vae_agent.step(step, map_ea2z)  
                actions = [[np.array([vae_actions[(e, 0)][0]]), np.array([vae_actions[(e, 1)][0]])]
                        for e in range(self.n_rollout_threads)]
                
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor([total_num_steps] * self.n_rollout_threads)

                self.vae_agent.insert_data(share_obs, obs)
                episode_reward += np.array([r.mean() for r in rewards])

            episode_avg.append(np.mean(episode_reward))

            current_reward = np.max(episode_reward)
            if current_reward > best_reward:
                best_reward = current_reward
                best_env = np.argmax(episode_reward)
                best_z = {'mean': map_ea2z[(best_env, 0)].mean(),
                        'std': map_ea2z[(best_env, 0)].std(),
                        'min': map_ea2z[(best_env, 0)].min(),
                        'max': map_ea2z[(best_env, 0)].max()}
                z_stats = z_distribution(z_sample_batch, print_z_stats=False, plot_z_hist=False)

            train_info = self.z_gen.train_step(episode_reward, z_sample_batch, z_old, log_prob_batch)

            # Log episode information
            env_info['episode_rewards'].append({
                        'mean': np.mean(episode_reward),
                        'std': np.std(episode_reward),
                        'min': np.min(episode_reward),
                        'max': np.max(episode_reward),
                        'overall_avg': np.mean(episode_avg) 
                    })
            env_info['total_num_steps'] += total_num_steps

            # Log best reward information
            env_info['best_reward'].append(best_reward)
            env_info['best_z'] = best_z
            env_info['best_batch_stats'] = z_stats

            # Logging
            end = time.time()
            time_per_episode = (end - start) / (episode + 1)
            time_left = (time_per_episode * (episodes - episode - 1)) / 60
            print("\n Layout '{}' Algo '{}' Exp '{}' updates {}/{} episodes, total num timesteps {}/{}, FPS {}, Estimated time left {:.2f} min.\n"
                  .format(self.all_args.layout_name,
                          self.algorithm_name,
                          self.experiment_name,
                          episode,
                          episodes,
                          total_num_steps,
                          self.num_env_steps,
                          int(total_num_steps / (end - start)),
                          time_left))
            
            self.z_gen.log_train_info(train_info, total_num_steps)
            self.z_gen.log_env_info(env_info, total_num_steps)

            print_env_stats(env_info, episode, best_reward)
            print_training_stats(train_info)
                
            # save model 
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.z_gen.save(total_num_steps)

        import sys
        sys.stdout.flush()
