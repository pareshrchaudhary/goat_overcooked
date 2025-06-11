import argparse
import os
import numpy as np

# torch 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# logging
import wandb
from tensorboardX import SummaryWriter
import pickle

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
        self.batch_size = args.n_rollout_threads
    
    def get_z(self):
        return np.random.normal(loc=0.0, scale=1.0, size=(self.batch_size, self.z_dim))
    
    def log_z_distribution(self, z_batch, steps):
        log_z_sample = {
                'z_sample/min': z_batch.min(),
                'z_sample/max': z_batch.max(),
                'z_sample/mean': z_batch.mean(),
                'z_sample/std': z_batch.std()
            }
        if self.use_wandb:
            wandb.log(log_z_sample, step=steps)
        else:
            self.writer.add_scalars(log_z_sample['z_sample/max'], steps)
            self.writer.add_scalars(log_z_sample['z_sample/mean'],  steps)
            self.writer.add_scalars(log_z_sample['z_sample/std'],  steps)
            self.writer.add_scalars(log_z_sample['z_sample/min'],  steps)

class AdversarialZ(nn.Module):
    def __init__(self, args, device, run_dir):
        super().__init__()
        self.device = device
        self.train_info = {}
        self.total_episodes = 0
        self.current_episode = 0
        
        # logging
        self.use_wandb = args.use_wandb
        print("use_wandb: ", self.use_wandb)
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = run_dir
            self.log_dir = os.path.join(self.run_dir, "logs")
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)
            self.save_dir = os.path.join(self.run_dir, "models")
            os.makedirs(self.save_dir, exist_ok=True)

        # training parameters
        self.z_dim = args.z_dim
        self.batch_size = args.n_rollout_threads
        self.hidden_dim = args.hidden_dim

        # loss parameters
        if args.use_kl:
            self.kl_coeff = args.kl_coeff
            print(f"Using KL loss with coeff: {self.kl_coeff}")
        else:
            self.kl_coeff = 0

        if args.use_entropy_bonus:
            self.entropy_coeff = args.entropy_coeff
            print(f"Using entropy bonus with coeff: {self.entropy_coeff}")
        else:
            self.entropy_coeff = 0

        # optimizer parameters
        self.adversary_learning_rate = args.adversary_learning_rate
        self.adversary_weight_decay = args.adversary_weight_decay
        self.use_grad_clip = args.use_grad_clip
        self.grad_clip_norm = args.grad_clip_norm

        # lr scheduler parameters
        self.use_lr_scheduler = args.use_lr_scheduler
        self.start_factor = args.start_factor
        self.end_factor = args.end_factor

        # Scaling parameters for mean and log_std
        if args.use_mean_clipping:
            self.use_mean_clipping = args.use_mean_clipping
            self.clip_mean_min = args.clip_mean_min
            self.clip_mean_max = args.clip_mean_max
            # print(f"Using mean clipping with min: {self.clip_mean_min}, max: {self.clip_mean_max}")
        else:
            self.use_mean_clipping = False
            self.clip_mean_min = None
            self.clip_mean_max = None
        if args.use_std_scaling:
            self.use_std_scaling = args.use_std_scaling
            self.scale_log_std = args.scale_log_std
            self.scale_std_offset = args.scale_std_offset
            # print(f"Using std scaling with scale: {self.scale_log_std}, offset: {self.scale_std_offset}")
        else:
            self.use_std_scaling = False
            self.scale_log_std = None
            self.scale_std_offset = None

        self.adversary = nn.Sequential(nn.Linear(self.z_dim, self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim, self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim, self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim, 2 * self.z_dim)
                                   ).to(self.device)

        self.optimizer = optim.Adam(self.adversary.parameters(), 
                                    lr=self.adversary_learning_rate, 
                                    weight_decay=self.adversary_weight_decay)
    
    def lr_scheduler(self, total_episodes):
        self.total_episodes = total_episodes
        if self.use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer,
                                                start_factor=self.start_factor,
                                                end_factor=self.end_factor,  
                                                total_iters=self.total_episodes)
            
    def mean_clip(self, vector, min_mean=-0.5, max_mean=0.5):
        current_mean = torch.mean(vector)
        shift = 0
        if current_mean < min_mean:
            shift = min_mean - current_mean 
        elif current_mean > max_mean:
            shift = max_mean - current_mean
        return vector + shift

    def forward(self, z):
        x = self.adversary(z)
        mean, log_std = torch.chunk(x, 2, dim=-1)

        if self.use_mean_clipping:
            mean = self.mean_clip(mean, min_mean=self.clip_mean_min, max_mean=self.clip_mean_max)
        if self.use_std_scaling:
            log_std = torch.tanh(log_std) * self.scale_log_std + self.scale_std_offset

        return mean, log_std
    
    def get_z(self):
        self.current_episode += 1
        z_prior = torch.distributions.Normal(torch.zeros((self.batch_size, self.z_dim), device=self.device),
                                            torch.ones((self.batch_size, self.z_dim), device=self.device))
        z_sample_old = z_prior.sample()
        mean, log_std = self.forward(z_sample_old)
        std = torch.exp(log_std)
        std = std + 1e-6

        z_current = torch.distributions.Normal(mean, std)
        z_sample_new = z_current.rsample().detach()
        log_prob = z_current.log_prob(z_sample_new)

        with torch.no_grad():
            self.train_info.update({
                'z_sample/mean': z_sample_new.mean().item(),
                'z_sample/std': z_sample_new.std().item(),
                'log_prob': log_prob.mean().item()
            })
        return z_sample_new, z_sample_old, log_prob, z_current, z_prior

    def train_step(self, episode_rewards_vae_vs_vae, episode_rewards_minimax, z_new, z_old, log_probs, current_dist, old_dist):
        self.adversary.train()
        with torch.no_grad():
            z_normal_diag = torch.normal(mean=0, std=1, size=(self.batch_size, self.z_dim), device=self.device)
            old_mean_diag, old_log_std_diag = self.forward(z_normal_diag)
            old_std_diag = torch.exp(old_log_std_diag)
            old_dist_diag = torch.distributions.Normal(old_mean_diag, old_std_diag)
        
        raw_regret = episode_rewards_vae_vs_vae - episode_rewards_minimax
        regret = (raw_regret - raw_regret.mean()) / (raw_regret.std() + 1e-8)   
        weighted_log_probs = log_probs.sum(dim=1) * regret

        # kl divergence 
        kl_div = torch.distributions.kl_divergence(current_dist, old_dist).mean()
        # entropy bonus
        entropy = current_dist.entropy().mean()
        loss = -weighted_log_probs.mean() + self.kl_coeff * kl_div - self.entropy_coeff * entropy
        self.optimizer.zero_grad()
        loss.backward()

        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.adversary.parameters(), max_norm=self.grad_clip_norm)
        
        self.optimizer.step()
        if self.use_lr_scheduler:
            self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']

        entropy_diag, kl_diag = self.compute_policy_stats(old_dist=old_dist_diag)
        
        self.train_info.update({'loss': loss.item(),
                                'lr': current_lr,
                                'episode_rewards_vae_vs_vae': episode_rewards_vae_vs_vae.mean().item(),
                                'episode_rewards_minimax': episode_rewards_minimax.mean().item(),
                                'regret': raw_regret.mean().item(),
                                'weighted_log_probs': weighted_log_probs.mean().item(),
                                'kl_divergence': kl_div.mean(),
                                'policy_update': kl_diag.item(),
                                'entropy': entropy_diag.item(),
                                })

        return self.train_info

    def compute_policy_stats(self, old_dist=None):
        """Compute policy distribution stats after update."""
        with torch.no_grad():
            z_normal = torch.normal(mean=0, std=1, size=(self.batch_size, self.z_dim), device=self.device)
            mean, log_std = self.forward(z_normal)
            std = torch.exp(log_std)

            new_dist = torch.distributions.Normal(mean, std)
            entropy = new_dist.entropy().mean()
            
            kl_div = 0.0
            if old_dist is not None:
                kl_div = torch.distributions.kl_divergence(old_dist, new_dist).mean()
            
            return entropy, kl_div
        
    def save(self, steps):
        os.makedirs(os.path.join(self.save_dir, "adversary"), exist_ok=True)
        torch.save(self.adversary.state_dict(), os.path.join(self.save_dir, "adversary", "adversary_{}.pt".format(steps)))

    def restore(self, steps):
        self.adversary.load_state_dict(torch.load(os.path.join(self.run_dir, "adversary", "adversary_{}.pt".format(steps)), weights_only=True))

    def log_train_info(self, train_info, steps):
        log_data = {
            'adversary/lr': train_info['lr'],
            'adversary/vae_vs_vae_reward': train_info['episode_rewards_vae_vs_vae'],
            'adversary/minimax_reward': train_info['episode_rewards_minimax'],
            'adversary/regret': train_info['regret'],
            'adversary/loss': train_info['loss'],
            'adversary/kl_divergence': train_info['kl_divergence'],
            'adversary/entropy': train_info['entropy'],
            'adversary/weighted_log_probs': train_info['weighted_log_probs'],
            'adversary/z_sample/mean': train_info['z_sample/mean'],
            'adversary/z_sample/std': train_info['z_sample/std'],
            'adversary/z_sample/log_prob': train_info['log_prob'],
            'adversary/policy_update': train_info['policy_update']
        }
        if self.use_wandb:
            wandb.log(log_data, step=steps)
        else:
            for key, value in {**log_data}.items():
                self.writer.add_scalar(key, value, steps)

def get_z_generator(args, vae_model, device=None, run_dir=None):
    if args.vae_z_generator == "normal_gaussian":
        return NormalGaussianZ(args)
    elif args.vae_z_generator == "adversarial":
        return AdversarialZ(args, device, run_dir=run_dir)
    else:
        raise ValueError(f"Z generator {args.vae_z_generator} not defined")
