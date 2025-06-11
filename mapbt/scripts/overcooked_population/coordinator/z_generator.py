import argparse

import numpy as np
import torch

from mapbt.scripts.overcooked_population.vae_constructor.hdf5_dataset import New_HDF5Dataset as HDF5Dataset


class ZGenerator:
    def __init__(self):
        pass
    
    def before_episode(self, episode):
        pass

    def after_episode(self, episode):
        pass

    def get_z(self, e, a):
        """
            e for id[rollout], a for id[agent]
        """
        raise NotImplementedError

class NormalGaussianZ(ZGenerator):
    def __init__(self, args):
        self.z_dim = args.z_dim
    
    def get_z(self, e, a):
        return np.random.normal(0, 1, (self.z_dim))

class HumanBiasedZ(ZGenerator):
    def __init__(self, args, vae_model):
        self.episode_length = args.episode_length
        self.chunk_length = args.vae_chunk_length
        self.z_dim = args.z_dim
        self.human_dataset = HDF5Dataset(args, "train")
        self.load_human_data()

        self.vae_model = vae_model
        self.get_human_distribution(use_std=(args.vae_z_generator == "human_biased_std"))
    
    def load_human_data(self):
        self.ref_obs, self.ref_actions, self.tar_obs, self.tar_actions, self.policy_id = [], [], [], [], []
        for _ in range(self.episode_length // self.chunk_length):
            self.human_dataset.reset()
            for i in range(len(self.human_dataset)):
                ref_obs, ref_actions, tar_obs, tar_actions, policy_id = self.human_dataset[i]
                self.ref_obs.append(ref_obs)
                self.ref_actions.append(ref_actions)
                self.tar_obs.append(tar_obs)
                self.tar_actions.append(tar_actions)
                self.policy_id.append(policy_id)
        self.ref_obs = torch.stack(self.ref_obs)
        self.ref_actions = torch.stack(self.ref_actions)
        self.tar_obs = torch.stack(self.tar_obs)
        self.tar_actions = torch.stack(self.tar_actions)
        self.policy_id = torch.stack(self.policy_id)
        # from icecream import ic
        # ic(self.ref_obs.shape, self.ref_obs.dtype)
        # ic(self.ref_actions.shape, self.ref_actions.dtype)
        # ic(self.tar_obs.shape, self.tar_obs.dtype)
        # ic(self.tar_actions.shape, self.tar_actions.dtype)
        # ic(self.policy_id.shape, self.policy_id.dtype)
    
    def get_human_distribution(self, use_std=False):
        z = self.vae_model.encode(self.ref_obs, self.ref_actions, sampling="mean")
        p_x = self.vae_model.decode(self.ref_obs, z)
        tar_actions = self.tar_actions.to(self.vae_model.device)
        tar_actions = torch.cat([tar_actions[:, :, 0], tar_actions[:, :, 1]], 0).swapaxes(0, 1).squeeze(-1)

        pi_targets = tar_actions.reshape(-1, *tar_actions.shape[2:])
        correct_guess = (p_x.probs.argmax(-1) == pi_targets).to(torch.float32)

        self.mean = z.mean(0).mean(0).detach().clone().cpu().numpy()
        self.std = z.reshape(-1, self.z_dim).std(0).detach().clone().cpu().numpy() if use_std else np.ones_like(self.mean)
        print("correct_guess.mean()", correct_guess.mean())
        print("self.mean", self.mean)
        print("self.std", self.std)
    
    def get_z(self, e, a):
        return np.random.normal(self.mean, self.std)

def get_z_generator(args, vae_model):
    if args.vae_z_generator == "normal_gaussian":
        return NormalGaussianZ(args)
    elif args.vae_z_generator in ["human_biased", "human_biased_std"]:
        return HumanBiasedZ(args, vae_model)
    else:
        raise ValueError(f"Z generator {args.vae_z_generator} not defined")
