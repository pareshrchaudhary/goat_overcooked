import itertools
import os
import time
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import wandb

from mapbt.runner.shared.overcooked_runner import OvercookedRunner as Runner
from ..vae_constructor.vae_model import VAEModel
from .vae_agent import VAEAgent
from .z_generator import get_z_generator


class OvercookedRunner(Runner):
    def evaluate_one_episode_with_multi_policy(self, policy_pool: dict, map_ea2p: dict):
        """Evaluate one episode with different policy for each agent.
        Params:
            policy_pool (Dict): a pool of policies. Each policy should support methods 'step' that returns actions given observation while maintaining hidden states on its own, and 'reset' that resets the hidden state.
            map_ea2p (Dict): a mapping from (env_id, agent_id) to policy name
        """
        warnings.warn("Evaluation with multi policy is not compatible with async done.")
        [policy.reset(self.n_eval_rollout_threads, self.num_agents) for policy_name, policy in policy_pool.items()]
        for e in range(self.n_eval_rollout_threads):
            for agent_id in range(self.num_agents):
                if not map_ea2p[(e, agent_id)].startswith("script:"):
                    policy_pool[map_ea2p[(e, agent_id)]].register_control_agent(e, agent_id)

        eval_env_infos = defaultdict(list)
        reset_choose = np.ones(self.n_eval_rollout_threads) == 1
        eval_obs, _, _ = self.eval_envs.reset(reset_choose)

        extract_info_keys = [] # ['stuck', 'can_begin_cook']
        infos = None
        for eval_step in range(self.all_args.episode_length):
            # print("Step", eval_step)
            eval_actions = np.full((self.n_eval_rollout_threads, self.num_agents, 1), fill_value=0).tolist()
            for policy_name, policy in policy_pool.items():
                if len(policy.control_agents) > 0:
                    policy.prep_rollout()
                    policy.to(self.device)
                    obs_lst = [eval_obs[e][a] for (e, a) in policy.control_agents]
                    info_lst = None
                    if infos is not None:
                        info_lst = {k: [infos[e][k][a] for e, a in policy.control_agents] for k in extract_info_keys}
                    agents = policy.control_agents
                    actions = policy.step(np.stack(obs_lst, axis=0), agents, info = info_lst, deterministic = not self.all_args.eval_stochastic)
                    for action, (e, a) in zip(actions, agents):
                        eval_actions[e][a] = action
            # Observe reward and next obs
            eval_actions = np.array(eval_actions)
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)

            infos = eval_infos

        shaped_info_keys = [
            "put_onion_on_X",
            "put_tomato_on_X",
            "put_dish_on_X",
            "put_soup_on_X",
            "pickup_onion_from_X",
            "pickup_onion_from_O",
            "pickup_tomato_from_X",
            "pickup_tomato_from_T",
            "pickup_dish_from_X",
            "pickup_dish_from_D",
            "pickup_soup_from_X",
            "USEFUL_DISH_PICKUP", # counted when #taken_dishes < #cooking_pots + #partially_full_pots and no dishes on the counter
            "SOUP_PICKUP", # counted when soup in the pot is picked up (not a soup placed on the table)
            "PLACEMENT_IN_POT", # counted when some ingredient is put into pot
            "viable_placement",
            "optimal_placement",
            "catastrophic_placement",
            "useless_placement",
            "potting_onion",
            "potting_tomato",
            "delivery",
        ]

        for eval_info in eval_infos:
            # for a in range(self.num_agents):
                # for i, k in enumerate(shaped_info_keys):
                #     eval_env_infos[f'eval_ep_{k}_by_agent{a}'].append(eval_info['episode']['ep_category_r_by_agent'][a][i])
                # eval_env_infos[f'eval_ep_sparse_r_by_agent{a}'].append(eval_info['episode']['ep_sparse_r_by_agent'][a])
                # eval_env_infos[f'eval_ep_shaped_r_by_agent{a}'].append(eval_info['episode']['ep_shaped_r_by_agent'][a])
            eval_env_infos['eval_ep_sparse_r'].append(eval_info['episode']['ep_sparse_r'])
            # eval_env_infos['eval_ep_shaped_r'].append(eval_info['episode']['ep_shaped_r'])
        
        return eval_env_infos
    
    def evaluate_with_multi_policy(self, policy_pool = None, map_ea2p = None, num_eval_episodes = None):
        """Evaluate with different policy for each agent.
        """
        policy_pool = policy_pool or self.policy.policy_pool
        map_ea2p = map_ea2p or self.policy.map_ea2p
        num_eval_episodes = num_eval_episodes or self.all_args.eval_episodes
        eval_infos = defaultdict(list)
        for episode in range(num_eval_episodes // self.n_eval_rollout_threads):
            eval_env_info = self.evaluate_one_episode_with_multi_policy(policy_pool, map_ea2p)
            for k, v in eval_env_info.items():
                for e in range(self.n_eval_rollout_threads):
                    agent0, agent1 = map_ea2p[(e, 0)], map_ea2p[(e, 1)]
                    # for log_name in [f"{agent0}-{agent1}-{k}", f"agent0-{agent0}-{k}", f"agent1-{agent1}-{k}", f"either-{agent0}-{k}", f"either-{agent1}-{k}"]:
                    for log_name in [f"{agent0}-{agent1}-{k}", f"either-{agent0}-{k}", f"either-{agent1}-{k}"]:
                        eval_infos[log_name].append(v[e])
        eval_infos = {k: [np.mean(v),] for k, v in eval_infos.items()}
        return eval_infos
    
    def naive_train_with_multi_policy(self, reset_map_ea2t_fn = None, reset_map_ea2p_fn = None):
        """This is a naive training loop using TrainerPool and PolicyPool. 

        To use PolicyPool and TrainerPool, you should first initialize population in policy_pool, with either:
        >>> self.policy.load_population(population_yaml_path)
        >>> self.trainer.init_population()
        or:
        >>> # mannually register policies
        >>> self.policy.register_policy(policy_name="ppo1", policy=rMAPPOpolicy(args, obs_space, share_obs_space, act_space), policy_config=(args, obs_space, share_obs_space, act_space), policy_train=True)
        >>> self.policy.register_policy(policy_name="ppo2", policy=rMAPPOpolicy(args, obs_space, share_obs_space, act_space), policy_config=(args, obs_space, share_obs_space, act_space), policy_train=True)
        >>> self.trainer.init_population()

        To bind (env_id, agent_id) to different trainers and policies:
        >>> map_ea2t = {(e, a): "ppo1" if a == 0 else "ppo2" for e in range(self.n_rollout_threads) for a in range(self.num_agents)}
        >>> map_ea2p = {(e, a): "ppo1" if a == 0 else "ppo2" for e in range(self.n_eval_rollout_threads) for a in range(self.num_agents)}
        >>> self.trainer.set_map_ea2t(map_ea2t)
        >>> self.policy.set_map_ea2p(map_ea2p)

        Note that map_ea2t is for training while map_ea2p is for policy evaluations

        WARNING: Currently do not support changing map_ea2t and map_ea2p when training. To implement this, we should take the first obs of next episode in the previous buffers and feed into the next buffers.
        """
        
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0
        env_infos = defaultdict(list)
        self.eval_info = dict()
        self.env_info = dict()
        self.init_best_r_logger()

        for episode in range(0, episodes): 
            self.total_num_steps = total_num_steps
            if self.use_linear_lr_decay:
                self.trainer.lr_decay(episode, episodes)
            
            # reset env agents
            if reset_map_ea2t_fn is not None:
                map_ea2t = reset_map_ea2t_fn(episode)
                self.trainer.reset(map_ea2t, self.n_rollout_threads, self.num_agents, n_repeats=None, load_unused_to_cpu=True)
                if self.all_args.use_policy_in_env:
                    load_policy_cfg = np.full((self.n_rollout_threads, self.num_agents), fill_value=None).tolist()
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            trainer_name = map_ea2t[(e, a)]
                            if trainer_name not in self.trainer.on_training:
                                load_policy_cfg[e][a] = self.trainer.policy_pool.policy_info[trainer_name]
                    self.envs.load_policy(load_policy_cfg)
                featurize_type = [[self.policy.featurize_type[map_ea2t[(e, a)]] for a in range(self.num_agents)]for e in range(self.n_rollout_threads)]
                self.envs.reset_featurize_type(featurize_type)

            # init env
            obs, share_obs, available_actions = self.envs.reset()

            # replay buffer
            if self.use_centralized_V:
                share_obs = share_obs
            else:
                share_obs = obs

            self.trainer.init_first_step(share_obs, obs)

            for step in range(self.episode_length):
                # Sample actions
                actions = self.trainer.step(step)
                    
                # Observe reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor(self.trainer.reward_shaping_steps())

                self.trainer.insert_data(share_obs, obs, rewards, dones, infos=infos)

            # update env infos
            episode_env_infos = defaultdict(list)
            if self.env_name == "Overcooked":
                for e, info in enumerate(infos):
                    agent0_trainer = self.trainer.map_ea2t[(e, 0)]
                    agent1_trainer = self.trainer.map_ea2t[(e, 1)]
                    # for log_name in [f"{agent0_trainer}-{agent1_trainer}", f"agent0-{agent0_trainer}", f"agent1-{agent1_trainer}", f"either-{agent0_trainer}", f"either-{agent1_trainer}"]:
                    for log_name in [f"{agent0_trainer}-{agent1_trainer}"]:
                        # episode_env_infos[f'{log_name}-ep_sparse_r_by_agent0'].append(info['episode']['ep_sparse_r_by_agent'][0])
                        # episode_env_infos[f'{log_name}-ep_sparse_r_by_agent1'].append(info['episode']['ep_sparse_r_by_agent'][1])
                        # episode_env_infos[f'{log_name}-ep_shaped_r_by_agent0'].append(info['episode']['ep_shaped_r_by_agent'][0])
                        # episode_env_infos[f'{log_name}-ep_shaped_r_by_agent1'].append(info['episode']['ep_shaped_r_by_agent'][1])
                        episode_env_infos[f'{log_name}-ep_sparse_r'].append(info['episode']['ep_sparse_r'])
                        # episode_env_infos[f'{log_name}-ep_shaped_r'].append(info['episode']['ep_shaped_r'])
                env_infos.update(episode_env_infos)
            self.env_info.update(env_infos)
            
            # compute return and update network
            train_infos = self.trainer.train(sp_size=getattr(self, "n_repeats", 0)*self.num_agents)
            
            # update advantage moving average
            if not hasattr(self, "avg_adv"):
                self.avg_adv = defaultdict(float)
            adv = self.trainer.compute_advantages()
            for (agent0, agent1, a), vs in adv.items():
                agent_pair = (agent0, agent1)
                for v in vs:
                    if agent_pair not in self.avg_adv.keys():
                        self.avg_adv[agent_pair] = v
                    else:
                        self.avg_adv[agent_pair] = self.avg_adv[agent_pair] * 0.99 + v * 0.01
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.trainer.save(episode, save_dir=self.save_dir)

            self.trainer.update_best_r({
                trainer_name: np.mean(self.env_info.get(f'either-{trainer_name}-ep_sparse_r', -1e9))
                for trainer_name in self.trainer.active_trainers
            }, save_dir=self.save_dir)

            # log information
            end = time.time()
            print("\n Layout {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(self.all_args.layout_name,
                            self.algorithm_name,
                            self.experiment_name,
                            episode,
                            episodes,
                            total_num_steps,
                            self.num_env_steps,
                            int(total_num_steps / (end - start))))
            print("average episode rewards is {}".format({k.split('-')[0]: train_infos[k] 
                for k in train_infos.keys() if "average_episode_rewards" in k}))
            if self.all_args.algorithm_name == 'traj':
                if self.all_args.traj_stage == 1:
                    print(f'jsd is {train_infos["average_jsd"]}')
                    print(f'jsd loss is {train_infos["average_jsd_loss"]}')

            if episode % self.log_interval == 0:
                self.log_train(train_infos, total_num_steps)
                print("env_infos", env_infos)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                if reset_map_ea2p_fn is not None:
                    map_ea2p = reset_map_ea2p_fn(episode)
                    self.policy.set_map_ea2p(map_ea2p, load_unused_to_cpu=True)
                eval_info = self.evaluate_with_multi_policy()
                new_eval_info = defaultdict(list)
                for eval_info_key in eval_info:
                    if eval_info_key.startswith("either-") and eval_info_key.endswith("-eval_ep_sparse_r"):
                        new_key = "eval_metrics/" + eval_info_key.split("either-")[1].split("-eval_ep_sparse_r")[0]
                        new_eval_info[new_key].append(eval_info[eval_info_key])
                eval_info.update(new_eval_info)
                for k, v in eval_info.items():
                    if k.startswith("eval_metrics"):
                        print(k, v)
                        self.update_best_r(k.split("eval_metrics/")[1], np.mean(v))
                self.log_env(eval_info, total_num_steps)
                self.eval_info.update(eval_info)
            
            import sys
            sys.stdout.flush()

    def train_coordinator_vs_agents(self):
        assert self.all_args.population_size == len(self.trainer.population)
        self.population_size = self.all_args.population_size
        self.population = sorted(self.trainer.population.keys()) # Note index and trainer name would not match when there are >= 10 agents

        print(f"population_size: {self.all_args.population_size}, {len(self.trainer.population)}")

        # Stage 2: train an agent against population with prioritized sampling
        agent_name = self.trainer.agent_name
        assert (self.n_eval_rollout_threads % self.all_args.eval_env_batch == 0 and self.all_args.eval_episodes % self.all_args.eval_env_batch == 0)
        assert self.n_rollout_threads % self.all_args.train_env_batch == 0
        self.all_args.eval_episodes = self.all_args.eval_episodes * self.n_eval_rollout_threads // self.all_args.eval_env_batch
        self.eval_idx = 0
        all_agent_pairs = list(itertools.product(self.population, [agent_name])) + list(itertools.product([agent_name], self.population))
        print(f"all agent pairs: {all_agent_pairs}")

        running_avg_r = -np.ones((self.population_size * 2,), dtype=np.float32) * 1e9

        def mep_reset_map_ea2t_fn(episode):
            # Randomly select agents from population to be trained
            # 1) consistent with MEP to train against one agent each episode 2) sample different agents to train against
            sampling_prob_np = np.ones((self.population_size * 2, ))
            for i, player_pair in enumerate(all_agent_pairs):
                for player_id in range(self.num_agents):
                    player_name = player_pair[player_id]
                    player_info = self.trainer.policy_pool.policy_info[player_name][1]
                    if player_info.get("held_out", False):
                        sampling_prob_np[i] = 0
                        break
            sampling_prob_np /= sampling_prob_np.sum()
            assert abs(sampling_prob_np.sum() - 1) < 1e-3

            # log sampling prob
            sampling_prob_dict = {}
            for i, agent_pair in enumerate(all_agent_pairs):
                sampling_prob_dict[f"sampling_prob/{agent_pair[0]}-{agent_pair[1]}"] = sampling_prob_np[i]
            if self.all_args.use_wandb:
                wandb.log(sampling_prob_dict, step=self.total_num_steps)

            n_selected = self.n_rollout_threads // self.all_args.train_env_batch
            pair_idx = np.random.choice(2 * self.population_size, size = (n_selected,), p = sampling_prob_np)
            if self.all_args.uniform_sampling_repeat > 0:
                assert n_selected >= 2 * self.population_size * self.all_args.uniform_sampling_repeat
                i = 0
                for r in range(self.all_args.uniform_sampling_repeat):
                    for x in range(2 * self.population_size):
                        pair_idx[i] = x
                        i += 1
            map_ea2t = {(e, a): all_agent_pairs[pair_idx[e % n_selected]][a] for e, a in itertools.product(range(self.n_rollout_threads), range(self.num_agents))}
            
            return map_ea2t
        
        def mep_reset_map_ea2p_fn(episode):
            if self.all_args.eval_policy != "":
                map_ea2p = {(e, a): [self.all_args.eval_policy, agent_name][(e + a) % 2] for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))}
            elif self.all_args.use_evaluation_agents:
                evaluation_pairs = []
                for i, player_pair in enumerate(all_agent_pairs):
                    for player_id in range(self.num_agents):
                        player_name = player_pair[player_id]
                        player_info = self.trainer.policy_pool.policy_info[player_name][1]
                        if player_info.get("evaluation_agent", False):
                            assert player_info.get("held_out", False), "evaluating agents should not be in the training population"
                            evaluation_pairs.append(player_pair)
                            break
                map_ea2p = {(e, a): evaluation_pairs[(self.eval_idx + e // self.all_args.eval_env_batch) % len(evaluation_pairs)][a] for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))}
                self.eval_idx += self.n_eval_rollout_threads // self.all_args.eval_env_batch
                self.eval_idx %= len(evaluation_pairs)
            else:
                map_ea2p = {(e, a): all_agent_pairs[(self.eval_idx + e // self.all_args.eval_env_batch) % (self.population_size * 2)][a] for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))}
                self.eval_idx += self.n_eval_rollout_threads // self.all_args.eval_env_batch
                self.eval_idx %= self.population_size * 2
            featurize_type = [[self.policy.featurize_type[map_ea2p[(e, a)]] for a in range(self.num_agents)]for e in range(self.n_eval_rollout_threads)]
            self.eval_envs.reset_featurize_type(featurize_type)
            return map_ea2p

        self.naive_train_with_multi_policy(reset_map_ea2t_fn=mep_reset_map_ea2t_fn, reset_map_ea2p_fn=mep_reset_map_ea2p_fn)
    
    def vae_train_with_multi_policy(self, reset_map_ea2t_ea2z_fn = None, reset_map_ea2p_fn = None):
        """This is a naive training loop using TrainerPool and PolicyPool. 

        To use PolicyPool and TrainerPool, you should first initialize population in policy_pool, with either:
        >>> self.policy.load_population(population_yaml_path)
        >>> self.trainer.init_population()
        or:
        >>> # mannually register policies
        >>> self.policy.register_policy(policy_name="ppo1", policy=rMAPPOpolicy(args, obs_space, share_obs_space, act_space), policy_config=(args, obs_space, share_obs_space, act_space), policy_train=True)
        >>> self.policy.register_policy(policy_name="ppo2", policy=rMAPPOpolicy(args, obs_space, share_obs_space, act_space), policy_config=(args, obs_space, share_obs_space, act_space), policy_train=True)
        >>> self.trainer.init_population()

        To bind (env_id, agent_id) to different trainers and policies:
        >>> map_ea2t = {(e, a): "ppo1" if a == 0 else "ppo2" for e in range(self.n_rollout_threads) for a in range(self.num_agents)}
        >>> map_ea2p = {(e, a): "ppo1" if a == 0 else "ppo2" for e in range(self.n_eval_rollout_threads) for a in range(self.num_agents)}
        >>> self.trainer.set_map_ea2t(map_ea2t)
        >>> self.policy.set_map_ea2p(map_ea2p)

        Note that map_ea2t is for training while map_ea2p is for policy evaluations

        WARNING: Currently do not support changing map_ea2t and map_ea2p when training. To implement this, we should take the first obs of next episode in the previous buffers and feed into the next buffers.
        """
        
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0
        env_infos = defaultdict(list)
        self.eval_info = dict()
        self.env_info = dict()
        self.init_best_r_logger()

        for episode in range(0, episodes): 
            self.total_num_steps = total_num_steps
            if self.use_linear_lr_decay:
                self.trainer.lr_decay(episode, episodes)
            
            # reset env agents
            if reset_map_ea2t_ea2z_fn is not None:
                map_ea2t, map_ea2z = reset_map_ea2t_ea2z_fn(episode)
                self.trainer.reset(map_ea2t, self.n_rollout_threads, self.num_agents, n_repeats=None, load_unused_to_cpu=True)
                if self.all_args.use_policy_in_env:
                    load_policy_cfg = np.full((self.n_rollout_threads, self.num_agents), fill_value=None).tolist()
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            if (e, a) not in map_ea2t:
                                continue
                            trainer_name = map_ea2t[(e, a)]
                            if trainer_name not in self.trainer.on_training:
                                load_policy_cfg[e][a] = self.trainer.policy_pool.policy_info[trainer_name]
                    self.envs.load_policy(load_policy_cfg)

            # init env
            obs, share_obs, available_actions = self.envs.reset()

            # replay buffer
            if self.use_centralized_V:
                share_obs = share_obs
            else:
                share_obs = obs

            self.trainer.init_first_step(share_obs, obs)
            self.vae_agent.init_first_step(share_obs, obs)

            for step in range(self.episode_length):
                # Sample actions
                actions = self.trainer.step(step)
                vae_actions = self.vae_agent.step(step, map_ea2z[step])
                for (e, a) in vae_actions:
                    actions[e][a] = vae_actions[(e, a)]
                    
                # Observe reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor(self.trainer.reward_shaping_steps())

                self.trainer.insert_data(share_obs, obs, rewards, dones, infos=infos)
                self.vae_agent.insert_data(share_obs, obs)

            # update env infos
            episode_env_infos = defaultdict(list)
            if self.env_name == "Overcooked":
                for e, info in enumerate(infos):
                    agent0_trainer = self.trainer.map_ea2t.get((e, 0), "vae")
                    agent1_trainer = self.trainer.map_ea2t.get((e, 1), "vae")
                    # for log_name in [f"{agent0_trainer}-{agent1_trainer}", f"agent0-{agent0_trainer}", f"agent1-{agent1_trainer}", f"either-{agent0_trainer}", f"either-{agent1_trainer}"]:
                    for log_name in [f"{agent0_trainer}-{agent1_trainer}"]:
                        # episode_env_infos[f'{log_name}-ep_sparse_r_by_agent0'].append(info['episode']['ep_sparse_r_by_agent'][0])
                        # episode_env_infos[f'{log_name}-ep_sparse_r_by_agent1'].append(info['episode']['ep_sparse_r_by_agent'][1])
                        # episode_env_infos[f'{log_name}-ep_shaped_r_by_agent0'].append(info['episode']['ep_shaped_r_by_agent'][0])
                        # episode_env_infos[f'{log_name}-ep_shaped_r_by_agent1'].append(info['episode']['ep_shaped_r_by_agent'][1])
                        episode_env_infos[f'{log_name}-ep_sparse_r'].append(info['episode']['ep_sparse_r'])
                        # episode_env_infos[f'{log_name}-ep_shaped_r'].append(info['episode']['ep_shaped_r'])
                env_infos.update(episode_env_infos)
            self.env_info.update(env_infos)
            
            # compute return and update network
            train_infos = self.trainer.train(sp_size=getattr(self, "n_repeats", 0)*self.num_agents)
            
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.trainer.save(episode, save_dir=self.save_dir)

            self.trainer.update_best_r({
                trainer_name: np.mean(self.env_info.get(f'either-{trainer_name}-ep_sparse_r', -1e9))
                for trainer_name in self.trainer.active_trainers
            }, save_dir=self.save_dir)

            # log information
            end = time.time()
            print("\n Layout {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(self.all_args.layout_name,
                            self.algorithm_name,
                            self.experiment_name,
                            episode,
                            episodes,
                            total_num_steps,
                            self.num_env_steps,
                            int(total_num_steps / (end - start))))
            print("average episode rewards is {}".format({k.split('-')[0]: train_infos[k] 
                for k in train_infos.keys() if "average_episode_rewards" in k}))
            if self.all_args.algorithm_name == 'traj':
                if self.all_args.traj_stage == 1:
                    print(f'jsd is {train_infos["average_jsd"]}')
                    print(f'jsd loss is {train_infos["average_jsd_loss"]}')

            if episode % self.log_interval == 0:
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                if reset_map_ea2p_fn is not None:
                    map_ea2p = reset_map_ea2p_fn(episode)
                    self.policy.set_map_ea2p(map_ea2p, load_unused_to_cpu=True)
                eval_info = self.evaluate_with_multi_policy()
                new_eval_info = defaultdict(list)
                for eval_info_key in eval_info:
                    if eval_info_key.startswith("either-") and eval_info_key.endswith("-eval_ep_sparse_r"):
                        new_key = "eval_metrics/" + eval_info_key.split("either-")[1].split("-eval_ep_sparse_r")[0]
                        new_eval_info[new_key].append(eval_info[eval_info_key])
                        
                eval_info.update(new_eval_info)
                for k, v in eval_info.items():
                    if k.startswith("eval_metrics"):
                        print(k, v)
                        self.update_best_r(k.split("eval_metrics/")[1], np.mean(v))
                self.log_env(eval_info, total_num_steps)
                self.eval_info.update(eval_info)
            
            import sys
            sys.stdout.flush()

    def train_coordinator_vs_vae(self):
        self.vae_model = VAEModel(*self.policy_config, device=self.device)
        self.vae_agent = VAEAgent(self.all_args, self.vae_model)

        assert self.all_args.population_size == len(self.trainer.population)
        self.population_size = self.all_args.population_size
        self.population = sorted(self.trainer.population.keys()) # Note index and trainer name would not match when there are >= 10 agents

        print(f"population_size: {self.all_args.population_size}, {len(self.trainer.population)}")

        # Stage 2: train an agent against population with prioritized sampling
        agent_name = self.trainer.agent_name
        assert (self.n_eval_rollout_threads % self.all_args.eval_env_batch == 0 and self.all_args.eval_episodes % self.all_args.eval_env_batch == 0)
        assert self.n_rollout_threads % self.all_args.train_env_batch == 0
        self.all_args.eval_episodes = self.all_args.eval_episodes * self.n_eval_rollout_threads // self.all_args.eval_env_batch
        self.eval_idx = 0
        all_agent_pairs = list(itertools.product(self.population, [agent_name])) + list(itertools.product([agent_name], self.population))
        print(f"all agent pairs: {all_agent_pairs}")

        running_avg_r = -np.ones((self.population_size * 2,), dtype=np.float32) * 1e9

        self.z_gen = get_z_generator(self.all_args, self.vae_model)
        def vae_reset_map_ea2t_ea2z_fn(episode):
            map_ea2t = {(e, e % 2): agent_name for e in range(self.n_rollout_threads)}
            map_ea2z = {}
            step_map_ea2z = {}
            for t in range(self.episode_length + 1):
                for e in range(self.n_rollout_threads):
                    for a in range(self.num_agents):
                        if (e, a) not in map_ea2t:
                            if t == 0 or np.random.rand() < self.all_args.vae_z_change_prob:
                                step_map_ea2z[(e, a)] = self.z_gen.get_z(e, a)
                map_ea2z[t] = deepcopy(step_map_ea2z)
            return map_ea2t, map_ea2z
        
        def vae_reset_map_ea2p_fn(episode):
            if self.all_args.eval_policy != "":
                map_ea2p = {(e, a): [self.all_args.eval_policy, agent_name][(e + a) % 2] for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))}
            elif self.all_args.use_evaluation_agents:
                evaluation_pairs = []
                for i, player_pair in enumerate(all_agent_pairs):
                    for player_id in range(self.num_agents):
                        player_name = player_pair[player_id]
                        player_info = self.trainer.policy_pool.policy_info[player_name][1]
                        if player_info.get("evaluation_agent"):
                            assert player_info.get("held_out", False), "evaluating agents should not be in the training population"
                            evaluation_pairs.append(player_pair)
                            break
                map_ea2p = {(e, a): evaluation_pairs[(self.eval_idx + e // self.all_args.eval_env_batch) % len(evaluation_pairs)][a] for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))}
                self.eval_idx += self.n_eval_rollout_threads // self.all_args.eval_env_batch
                self.eval_idx %= len(evaluation_pairs)
            else:
                map_ea2p = {(e, a): all_agent_pairs[(self.eval_idx + e // self.all_args.eval_env_batch) % (self.population_size * 2)][a] for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))}
                self.eval_idx += self.n_eval_rollout_threads // self.all_args.eval_env_batch
                self.eval_idx %= self.population_size * 2
            featurize_type = [[self.policy.featurize_type[map_ea2p[(e, a)]] for a in range(self.num_agents)]for e in range(self.n_eval_rollout_threads)]
            self.eval_envs.reset_featurize_type(featurize_type)
            return map_ea2p

        self.vae_train_with_multi_policy(reset_map_ea2t_ea2z_fn=vae_reset_map_ea2t_ea2z_fn, reset_map_ea2p_fn=vae_reset_map_ea2p_fn)

    def init_best_r_logger(self):
        self.best_r = {}

    def update_best_r(self, partner_name, r):
        if partner_name not in self.best_r:
            self.best_r[partner_name] = -1e9
        if r > self.best_r[partner_name]:
            self.best_r[partner_name] = r
            coordinator = self.trainer.trainer_pool[self.trainer.agent_name].policy
            self.save_best_r_coordinator(coordinator, partner_name)

    def save_best_r_coordinator(self, coordinator, partner_name):
        save_dir = str(self.save_dir) + "/coordinator_best_r"
        os.makedirs(save_dir, exist_ok=True)
        policy_actor = coordinator.actor
        torch.save(policy_actor.state_dict(), save_dir + "/actor_best_r_vs_{}.pt".format(partner_name))
        policy_critic = coordinator.critic
        torch.save(policy_critic.state_dict(), save_dir + "/critic_best_r_vs_{}.pt".format(partner_name))
