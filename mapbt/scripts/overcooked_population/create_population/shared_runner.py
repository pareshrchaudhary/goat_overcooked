import os
import pickle
import time
from collections import defaultdict
from copy import deepcopy
from typing import Optional

import h5py
import numpy as np
import torch
from icecream import ic
from tqdm import tqdm

from mapbt.runner.shared.overcooked_runner import OvercookedRunner as Runner


class OvercookedRunner(Runner):
    def run_w_trainer_reward(self):
        ic(self.all_args.trainer_reward_shaping_command)
        trainer_reward_shaping_arguments = []
        if self.all_args.trainer_reward_shaping_command != "":
            if self.all_args.trainer_reward_shaping_command[0] != "(" or self.all_args.trainer_reward_shaping_command[-1] != ")":
                raise ValueError(f"trainer reward shaping command {self.all_args.trainer_reward_shaping_command} format error")
            for single_reward_command in self.all_args.trainer_reward_shaping_command[1:-1].split("), ("):
                step_ep, event, player_group, rew_func, rew_scale = single_reward_command.split(", ")
                if step_ep not in ["step", "episode"]:
                    raise ValueError
                if player_group not in ["player_0", "player_1", "both"]:
                    raise ValueError
                if rew_func not in ["accumulate", "sign", ">0", ">1", ">2", "<0", "<-1", "<-2", "acc:sum<5", "acc:sum>-5"]:
                    raise NotImplementedError
                trainer_reward_shaping_arguments.append((step_ep, event, player_group, rew_func, float(rew_scale)))
        ic(trainer_reward_shaping_arguments)
        
        def get_trainer_reward(episode, episodes, player_id, info, done):
            annealing = self.all_args.trainer_reward_shaping_annealing
            if annealing == "const":
                k = 1
            elif annealing == "linear":
                k = max(0, 1 - episode / episodes)
            elif annealing == "cos_2cycle":
                k = np.cos(2 * np.pi * episode / (episodes / 2))
                k = (k + 1) / 2
            elif annealing == "sin_2cycle":
                k = np.sin(2 * np.pi * episode / (episodes / 2))
                k = (k + 1) / 2
            else:
                raise ValueError(f"annealing function {k} not defined!")
            step_info = info["stepwise_shaped_info_by_agent"][player_id]
            episode_info = info["shaped_info_by_agent"][player_id]
            player_done = done[player_id]
            trainer_reward_sum = 0
            for step_ep, event, player_group, rew_func, rew_scale in trainer_reward_shaping_arguments:
                if step_ep == "episode" and not player_done:
                    continue
                if player_group not in [f"player_{player_id}", "both"]:
                    "does continue"
                    continue
                event_counter = step_info[event] if step_ep == "step" else episode_info[event]
                if rew_func == "accumulate":
                    trainer_reward = rew_scale * event_counter
                elif rew_func == "sign":
                    trainer_reward = rew_scale * np.sign(event_counter)
                elif rew_func == ">0":
                    trainer_reward = rew_scale * (event_counter > 0)
                elif rew_func == ">1":
                    trainer_reward = rew_scale * (event_counter > 1)
                elif rew_func == ">2":
                    trainer_reward = rew_scale * (event_counter > 2)
                elif rew_func == "<0":
                    trainer_reward = rew_scale * (event_counter < 0)
                elif rew_func == "<-1":
                    trainer_reward = rew_scale * (event_counter < -1)
                elif rew_func == "<-2":
                    trainer_reward = rew_scale * (event_counter < -2)
                elif rew_func == "acc:sum<5":
                    trainer_reward = rew_scale * event_counter * (episode_info[event] < 5)
                elif rew_func == "acc:sum>-5":
                    trainer_reward = rew_scale * event_counter * (episode_info[event] > -5)
                else:
                    raise NotImplementedError
                trainer_reward_sum += k * trainer_reward
            return trainer_reward_sum

        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        for episode in range(episodes): 
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            episode_trainer_reward = 0
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                for rollout_id, (info, done) in enumerate(zip(infos, dones)):
                    for player_id in range(self.num_agents):
                        trainer_reward = get_trainer_reward(episode, episodes, player_id, info, done)
                        episode_trainer_reward += trainer_reward / self.n_rollout_threads / self.num_agents
                        rewards[rollout_id, player_id, 0] += trainer_reward
                obs = np.stack(obs)
                total_num_steps += (self.n_rollout_threads)
                self.envs.anneal_reward_shaping_factor([total_num_steps] * self.n_rollout_threads)
                data = obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if episode < 50:
                if episode % 1 == 0:
                    self.save(episode)
            elif episode < 100:
                if episode % 2 == 0:
                    self.save(episode)
            else:
                if (episode % self.save_interval == 0 or episode == episodes - 1):
                    self.save(episode)

            # log information
            if episode % self.log_interval == 0:
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

                
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                print("average episode trainer rewards is {}".format(episode_trainer_reward))

                env_infos = defaultdict(list)
                if self.env_name == "Overcooked":
                    for info in infos:
                        env_infos['ep_sparse_r_by_agent0'].append(info['episode']['ep_sparse_r_by_agent'][0])
                        env_infos['ep_sparse_r_by_agent1'].append(info['episode']['ep_sparse_r_by_agent'][1])
                        env_infos['ep_shaped_r_by_agent0'].append(info['episode']['ep_shaped_r_by_agent'][0])
                        env_infos['ep_shaped_r_by_agent1'].append(info['episode']['ep_shaped_r_by_agent'][1])
                        env_infos['ep_sparse_r'].append(info['episode']['ep_sparse_r'])
                        env_infos['ep_shaped_r'].append(info['episode']['ep_shaped_r'])
                    env_infos['ep_trainer_r'].append(episode_trainer_reward)

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                for info_key in env_infos:
                    print(info_key, np.mean(env_infos[info_key]))

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    @torch.no_grad()
    def collect_one_episode(self, n_rollout_threads, envs, policy_pool: dict, map_ea2p: dict):
        """Collect one episode with different policy for each agent.
        Params:
            policy_pool (Dict): a pool of policies. Each policy should support methods 'step' that returns actions given observation while maintaining hidden states on its own, and 'reset' that resets the hidden state.
            map_ea2p (Dict): a mapping from (env_id, agent_id) to policy name
        """
        episode_obs, episode_rewards, episode_dones, episode_actions = [], [], [], []

        [policy.reset(n_rollout_threads, self.num_agents) for policy_name, policy in policy_pool.items()]
        for e in range(n_rollout_threads):
            for agent_id in range(self.num_agents):
                if not map_ea2p[(e, agent_id)].startswith("script:"):
                    policy_pool[map_ea2p[(e, agent_id)]].register_control_agent(e, agent_id)

        env_infos = defaultdict(list)
        obs, share_obs, _ = envs.reset()

        episode_obs.append(np.array(obs))

        extract_info_keys = [] # ['stuck', 'can_begin_cook']
        infos = None
        for step in range(self.all_args.episode_length):
            # print("Step", step)
            actions = np.full((n_rollout_threads, self.num_agents, 1), fill_value=0).tolist()
            for policy_name, policy in policy_pool.items():
                if len(policy.control_agents) > 0:
                    policy.prep_rollout()
                    # policy.to(self.device)
                    obs_lst = [obs[e][a] for (e, a) in policy.control_agents]
                    info_lst = None
                    if infos is not None:
                        info_lst = {k: [infos[e][k][a] for e, a in policy.control_agents] for k in extract_info_keys}
                    agents = policy.control_agents
                    step_actions = policy.step(np.stack(obs_lst, axis=0), agents, info=info_lst, deterministic=False)
                    for action, (e, a) in zip(step_actions, agents):
                        actions[e][a] = action
            # Observe reward and next obs
            actions = np.array(actions)
            obs, share_obs, rewards, dones, infos, available_actions = envs.step(actions)
            episode_rewards.append(rewards)
            episode_dones.append(dones)
            episode_actions.append(actions)
            episode_obs.append(np.array(obs))

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

        for info in infos:
            for a in range(self.num_agents):
                for i, k in enumerate(shaped_info_keys):
                    env_infos[f'eval_ep_{k}_by_agent{a}'].append(info['episode']['ep_category_r_by_agent'][a][i])
                env_infos[f'eval_ep_sparse_r_by_agent{a}'].append(info['episode']['ep_sparse_r_by_agent'][a])
                env_infos[f'eval_ep_shaped_r_by_agent{a}'].append(info['episode']['ep_shaped_r_by_agent'][a])
            env_infos['eval_ep_sparse_r'].append(info['episode']['ep_sparse_r'])
            env_infos['eval_ep_shaped_r'].append(info['episode']['ep_shaped_r'])

        return {
            "env_infos": env_infos,
            "episode_obs": np.array(episode_obs),
            "episode_actions": np.array(episode_actions),
            "episode_rewards": np.array(episode_rewards),
            "episode_dones": np.array(episode_dones),
        }          
    
    def get_training_policies(self) -> list[str]:
        policies = []
        for policy_name, (_, policy_info) in self.policy.policy_info.items():
            if policy_info.get("held_out") is not True:
                policies.append(policy_name)
        return policies
    
    def get_heldout_policies(self) -> list[str]:
        policies = []
        for policy_name, (_, policy_info) in self.policy.policy_info.items():
            if policy_info.get("held_out"):
                policies.append(policy_name)
        return policies

    def get_policy_pairs(self, policies: list[str]) -> list[tuple[str, str]]:
        if self.num_agents != 2:
            raise NotImplementedError
        pairs = []
        for policy_1 in policies:
            for policy_2 in policies:
                pairs.append((policy_1, policy_2))
        return pairs
    
    def store_data(self, datagroup: h5py.Group, k: str, v: np.ndarray, i: int, n: int, chunk_len: Optional[int]=None):
        if i == 0:
            datagroup.create_dataset(
                k,
                shape=(0,)+v.shape[1:],
                maxshape=(n,)+v.shape[1:],
                dtype=v.dtype,
                chunks=(chunk_len,)+v.shape[1:] if chunk_len else None
            )
        dset = datagroup[k]
        assert i == dset.shape[0]
        dset.resize(i + v.shape[0], axis=0)
        dset[-v.shape[0]:] = v

    def collect_trajectories(self, dataset: h5py.Group, num_traj: int, policy_names: list[str], agent_pairs: list[tuple[str, str]]):
        assert num_traj % len(agent_pairs) == 0

        env_info_dataset = dataset.create_group("env_info")

        pair_sampling_list = []
        for agent_pair in agent_pairs:
            pair_sampling_list += [agent_pair] * (num_traj // len(agent_pairs))
        pair_sampling_list += [agent_pairs[-1]] * self.n_rollout_threads
        for i in tqdm(range(0, num_traj, self.n_rollout_threads)):
            policy_id = np.zeros((self.n_rollout_threads, self.num_agents), dtype=int)
            cur_agent_pairs = pair_sampling_list[i:i+self.n_rollout_threads]
            
            map_ea2p = dict()
            for e, agent_pair in enumerate(cur_agent_pairs):
                for a in range(self.num_agents):
                    policy_id[e, a] = policy_names.index(agent_pair[a])
                    map_ea2p[(e, a)] = agent_pair[a]
            self.policy.set_map_ea2p(map_ea2p)
            
            while True:
                outcome = self.collect_one_episode(self.n_rollout_threads, self.envs, self.policy.policy_pool, map_ea2p)
                obs = outcome["episode_obs"].astype(np.int16)
                try:
                    assert np.all(obs == outcome["episode_obs"])
                    break
                except:
                    print(f'Failed to compress observation into np.int16, with outcome["episode_obs"].max() = {outcome["episode_obs"].max()}, with outcome["episode_obs"].min() = {outcome["episode_obs"].min()}')
                    continue
            
            rst_samples = min(self.n_rollout_threads, num_traj - i)
            obs = np.swapaxes(obs, 0, 1)[:rst_samples]
            rewards = np.swapaxes(outcome["episode_rewards"], 0, 1)[:rst_samples]
            dones = np.swapaxes(outcome["episode_dones"], 0, 1)[:rst_samples]
            actions = np.swapaxes(outcome["episode_actions"], 0, 1)[:rst_samples]
            policy_id = policy_id[:rst_samples]
            env_infos = {}
            for k, v in outcome["env_infos"].items():
                env_infos[k] = np.array(v)[:rst_samples]
            
            self.store_data(dataset, "obs", obs, i, num_traj, chunk_len=1)
            self.store_data(dataset, "rewards", rewards, i, num_traj, chunk_len=512)
            self.store_data(dataset, "dones", dones, i, num_traj, chunk_len=512)
            self.store_data(dataset, "actions", actions, i, num_traj, chunk_len=512)
            self.store_data(dataset, "policy_id", policy_id, i, num_traj)
            for k, v in env_infos.items():
                self.store_data(env_info_dataset, k, v, i, num_traj)
            
        for policy_name in policy_names:
            dataset["policy_id"].attrs[f"policy_id[{policy_name}]"] = policy_names.index(policy_name)

    def create_dataset(self):
        training_policies = self.get_training_policies()
        heldout_policies = self.get_heldout_policies()
        policy_names = {
            "train": training_policies,
            "test": heldout_policies,
        }
        agent_pairs = {
            "train": self.get_policy_pairs(training_policies),
            "test": self.get_policy_pairs(heldout_policies)
        }
        num_trajectoeis = {
            "train": self.all_args.total_trajectories,
            "test": self.all_args.test_trajectories
        }

        os.makedirs(os.path.dirname(self.all_args.dataset_file), exist_ok=True)
        with h5py.File(self.all_args.dataset_file, "w") as hdf5_file:
            for data_split in ["train", "test"]:
                print(f"expected #traj for {data_split}: {num_trajectoeis[data_split]}")
                num_trajectoeis[data_split] = len(agent_pairs[data_split]) * max(1, num_trajectoeis[data_split] // len(agent_pairs[data_split]))
                print(f"final #traj for {data_split}: {num_trajectoeis[data_split]}")

                dataset = hdf5_file.create_group(data_split)
                self.collect_trajectories(dataset, num_trajectoeis[data_split], policy_names[data_split], agent_pairs[data_split])
    
    def add_human_human_trajectories(self, dataset, data):
        num_joint_traj = len(data["ep_states"]) // 2
        num_total_chunks = 0
        obs, rewards, dones, actions = None, None, None, None
        for i in range(num_joint_traj):
            if obs is None:
                obs, rewards, dones, actions = data["ep_states"][i*2:i*2+2], data["ep_rewards"][i*2:i*2+2], data["ep_dones"][i*2:i*2+2], data["ep_actions"][i*2:i*2+2]
                obs, rewards, dones, actinos = deepcopy(obs), deepcopy(rewards), deepcopy(dones), deepcopy(actions)
            else:
                last_obs, last_rewards, last_dones, last_actions = data["ep_states"][i*2:i*2+2], data["ep_rewards"][i*2:i*2+2], data["ep_dones"][i*2:i*2+2], data["ep_actions"][i*2:i*2+2]
                for j in range(2):
                    obs[j] += last_obs[j]
                    rewards[j] = np.concatenate([rewards[j], last_rewards[j]])
                    dones[j] += last_dones[j]
                    actions[j] += last_actions[j]
            if len(obs[0]) < self.episode_length:
                continue
            num_total_chunks += int(np.round(len(obs[0]) // self.episode_length))
            obs, rewards, dones, actions = None, None, None, None
        print("num_total_chunks", num_total_chunks)
        chunk_cur = 0
        obs, rewards, dones, actions = None, None, None, None
        for i in range(num_joint_traj):
            if obs is None:
                obs, rewards, dones, actions = data["ep_states"][i*2:i*2+2], data["ep_rewards"][i*2:i*2+2], data["ep_dones"][i*2:i*2+2], data["ep_actions"][i*2:i*2+2]
                obs, rewards, dones, actinos = deepcopy(obs), deepcopy(rewards), deepcopy(dones), deepcopy(actions)
            else:
                last_obs, last_rewards, last_dones, last_actions = data["ep_states"][i*2:i*2+2], data["ep_rewards"][i*2:i*2+2], data["ep_dones"][i*2:i*2+2], data["ep_actions"][i*2:i*2+2]
                for j in range(2):
                    obs[j] += last_obs[j]
                    rewards[j] = np.concatenate([rewards[j], last_rewards[j]])
                    dones[j] += last_dones[j]
                    actions[j] += last_actions[j]
            if len(obs[0]) < self.episode_length:
                continue
            chunks = int(np.round(len(obs[0]) // self.episode_length))
            obs = np.stack(obs, 1) * 255
            try:
                assert np.all(obs.astype(np.int16) == obs)
                obs = obs.astype(np.int16)
            except Exception as e:
                raise RuntimeError(Exception)
            rewards = np.stack(rewards, 1)
            dones = np.stack(dones, 1)
            actions = np.stack(actions, 1)

            obs_lst, rew_lst, done_lst, act_lst = [], [], [], []
            obs = np.stack(list(obs) + [obs[0]])
            for chunk_start in np.round(np.linspace(0, len(rewards) - self.episode_length, chunks)).astype(int):
                obs_lst.append(obs[chunk_start:chunk_start+self.episode_length+1])
                rew_lst.append(rewards[chunk_start:chunk_start+self.episode_length])
                done_lst.append(dones[chunk_start:chunk_start+self.episode_length])
                act_lst.append(actions[chunk_start:chunk_start+self.episode_length])
            obs_lst = np.stack(obs_lst)
            rew_lst = np.expand_dims(np.stack(rew_lst), -1)
            done_lst = np.stack(done_lst)
            act_lst = np.stack(act_lst)
            self.store_data(dataset, "obs", obs_lst, chunk_cur, num_total_chunks, chunk_len=1)
            self.store_data(dataset, "rewards", rew_lst, chunk_cur, num_total_chunks, chunk_len=1)
            self.store_data(dataset, "dones", done_lst, chunk_cur, num_total_chunks, chunk_len=1)
            self.store_data(dataset, "actions", act_lst, chunk_cur, num_total_chunks, chunk_len=1)
            self.store_data(dataset, "policy_id", i * np.ones((chunks, 2), dtype=int), chunk_cur, num_total_chunks)
            chunk_cur += chunks
            dataset["policy_id"].attrs[f"policy_id[human_human_{i}]"] = i
            obs, rewards, dones, actions = None, None, None, None

    def make_bc_dataset(self):
        os.makedirs(os.path.dirname(self.all_args.dataset_file), exist_ok=True)
        with h5py.File(self.all_args.dataset_file, "w") as hdf5_file:
            dataset = hdf5_file.create_group("train")
            with open(self.all_args.raw_training_data, "rb") as f:
                data = pickle.load(f)
            self.add_human_human_trajectories(dataset, data)
            dataset = hdf5_file.create_group("test")
            with open(self.all_args.raw_test_data, "rb") as f:
                data = pickle.load(f)
            self.add_human_human_trajectories(dataset, data)
                