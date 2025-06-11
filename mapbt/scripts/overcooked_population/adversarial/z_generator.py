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
        self.use_wandb = args.use_wandb
    
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

        # advesary type
        if args.adversary_type == 'max':
            self.adversary_grad_coeff = 1
            print(f"Using max adversary with coeff: {self.adversary_grad_coeff}")
        elif args.adversary_type == 'min':
            self.adversary_grad_coeff = -1
            print(f"Using min adversary with coeff: {self.adversary_grad_coeff}")

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
        self.use_mean_scaling = args.use_mean_scaling
        self.use_std_scaling = args.use_std_scaling
        self.scale_mean = args.scale_mean
        self.scale_log_std = args.scale_log_std
        self.scale_std_offset = args.scale_std_offset

        self.model = nn.Sequential(nn.Linear(self.z_dim, self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim, self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim, self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim, 2 * self.z_dim)
                                   ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.adversary_learning_rate, 
                                    weight_decay=self.adversary_weight_decay)
    
    def lr_scheduler(self, total_episodes):
        self.total_episodes = total_episodes
        if self.use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer,
                                                start_factor=self.start_factor,
                                                end_factor=self.end_factor,  
                                                total_iters=self.total_episodes)
    
    def forward(self, z):
        x = self.model(z)
        mean, log_std = torch.chunk(x, 2, dim=-1)

        if self.use_mean_scaling:
            mean = torch.tanh(mean) * self.scale_mean
        if self.use_std_scaling:
            log_std = torch.tanh(log_std) * self.scale_log_std + self.scale_std_offset
        return mean, log_std
    
    def get_z(self):
        self.current_episode += 1
        z_old = torch.normal(mean=0, std=1, size=(self.batch_size, self.z_dim), device=self.device)
        mean, log_std = self.forward(z_old)
        std = torch.exp(log_std) 
        std = std + 1e-6

        normal = torch.distributions.Normal(mean, std)
        z_sample = normal.rsample().detach()
        log_prob = normal.log_prob(z_sample)

        with torch.no_grad():
            self.train_info.update({'z_sample': {'mean': z_sample.mean().item(),
                                                 'std': z_sample.std().item(), 
                                                 'min': z_sample.min().item(),
                                                 'max': z_sample.max().item()
                                                 },
                                    'log_prob': {'mean': log_prob.mean().item(),
                                                 'std': log_prob.std().item(),
                                                 'min': log_prob.min().item(),
                                                 'max': log_prob.max().item()
                                                 }
                                    })
        return z_sample, z_old, log_prob

    def train_step(self, episode_rewards, z_new, z_old, log_probs):
        self.model.train()

        with torch.no_grad():
            z_normal_diag = torch.normal(mean=0, std=1, size=(self.batch_size, self.z_dim), device=self.device)
            old_mean_diag, old_log_std_diag = self.forward(z_normal_diag)
            old_std_diag = torch.exp(old_log_std_diag)
            old_dist_diag = torch.distributions.Normal(old_mean_diag, old_std_diag)

        episode_rewards = torch.tensor(episode_rewards, device=self.device)
        # advantages = episode_rewards - episode_rewards.mean()
        advantages = self.adversary_grad_coeff * (episode_rewards - episode_rewards.mean())
        advantages = advantages / (advantages.std() + 1e-8)
        weighted_log_probs = log_probs.sum(dim=1) * advantages
        
        # kl divergence
        current_d = torch.distributions.Normal(z_new.mean(dim=1), z_new.std(dim=1))
        old_d = torch.distributions.Normal(z_old.mean(dim=1), z_old.std(dim=1))
        kl_div = torch.distributions.kl_divergence(current_d, old_d).mean()
        
        # entropy bonus
        current_dist = torch.distributions.Normal(z_new.mean(dim=1), z_new.std(dim=1))
        entropy = current_dist.entropy().mean()

        loss = -weighted_log_probs.mean()  + self.kl_coeff * kl_div - self.entropy_coeff * entropy

        self.optimizer.zero_grad()
        loss.backward()

        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

        self.optimizer.step()
        if self.use_lr_scheduler:
            self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        
        entropy, mean_range, std_range, kl_diag = self.compute_policy_stats(old_dist_diag)
        
        self.train_info.update({'loss': loss.item(),
                                'lr': current_lr,
                                'advantages': {'min': advantages.min().item(),
                                               'max': advantages.max().item(),
                                               'mean': advantages.mean().item(),
                                               'std': advantages.std().item()
                                               },
                                'weighted_log_probs': {'mean': weighted_log_probs.mean().item(),
                                                       'std': weighted_log_probs.std().item(), 
                                                       'min': weighted_log_probs.min().item(),
                                                       'max': weighted_log_probs.max().item()
                                                       },
                                'kl_divergence': kl_diag,
                                'entropy': {'value': entropy}
                                })

        return self.train_info

    def compute_policy_stats(self, old_dist=None):
        """Compute policy distribution stats after update."""
        with torch.no_grad():
            z_normal = torch.normal(mean=0, std=1, size=(self.batch_size, self.z_dim), device=self.device)
            mean, log_std = self.forward(z_normal)
            std = torch.exp(log_std)
            new_dist = torch.distributions.Normal(mean, std)
            
            entropy = new_dist.entropy().mean().item()
            mean_range = (mean.min().item(), mean.max().item())
            std_range = (std.min().item(), std.max().item())
            
            kl_div = 0.0
            if old_dist is not None:
                kl_div = torch.distributions.kl_divergence(old_dist, new_dist).mean().item()
            
            return entropy, mean_range, std_range, kl_div  
        
    def save(self, steps):
        os.makedirs(os.path.join(self.save_dir, "adversary"), exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "adversary", "adversary_{}.pt".format(steps)))

    def restore(self, steps):
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, "adversary", "adversary_{}.pt".format(steps))))

    def log_train_info(self, train_info, steps):
        log_data = {
            'adversary/loss': train_info['loss'],
            'adversary/lr': train_info['lr'],
            'adversary/kl_divergence': train_info['kl_divergence'],
            'adversary/entropy': train_info['entropy']['value']
            }
        log_z_sample = {
                'z_sample/min': train_info['z_sample']['min'],
                'z_sample/max': train_info['z_sample']['max'],
                'z_sample/mean': train_info['z_sample']['mean'],
                'z_sample/std': train_info['z_sample']['std']
            }
        log_advantages = {
                'adversary/advantages/min': train_info['advantages']['min'],
                'adversary/advantages/max': train_info['advantages']['max'],
                'adversary/advantages/mean': train_info['advantages']['mean'],
                'adversary/advantages/std': train_info['advantages']['std']
            }
        log_weighted_log_probs = {
                'adversary/weighted_log_probs/mean': train_info['weighted_log_probs']['mean'],
                'adversary/weighted_log_probs/std': train_info['weighted_log_probs']['std'],
                'adversary/weighted_log_probs/min': train_info['weighted_log_probs']['min'],
                'adversary/weighted_log_probs/max': train_info['weighted_log_probs']['max']
            }
        
        if self.use_wandb:
            wandb.log(log_data, step=steps)
            wandb.log(log_z_sample, step=steps)
            wandb.log(log_advantages, step=steps)
            wandb.log(log_weighted_log_probs, step=steps)
        else:
            self.writer.add_scalar('adversary/loss', train_info['loss'], steps)
            self.writer.add_scalar('adversary/lr', train_info['lr'], steps)
            self.writer.add_scalar('adversary/kl_divergence', train_info['kl_divergence'], steps)
            self.writer.add_scalar('adversary/entropy', train_info['entropy']['value'], steps)

            self.writer.add_scalar('z_sample/min', train_info['z_sample']['min'], steps)
            self.writer.add_scalar('z_sample/max', train_info['z_sample']['max'], steps)
            self.writer.add_scalar('z_sample/mean', train_info['z_sample']['mean'], steps)
            self.writer.add_scalar('z_sample/std', train_info['z_sample']['std'], steps)

            self.writer.add_scalar('adversary/advantages/min', train_info['advantages']['min'], steps)
            self.writer.add_scalar('adversary/advantages/max', train_info['advantages']['max'], steps)
            self.writer.add_scalar('adversary/advantages/mean', train_info['advantages']['mean'], steps)
            self.writer.add_scalar('adversary/advantages/std', train_info['advantages']['std'], steps)

            self.writer.add_scalar('adversary/weighted_log_probs/mean', train_info['weighted_log_probs']['mean'], steps)
            self.writer.add_scalar('adversary/weighted_log_probs/std', train_info['weighted_log_probs']['std'], steps)
            self.writer.add_scalar('adversary/weighted_log_probs/min', train_info['weighted_log_probs']['min'], steps)
            self.writer.add_scalar('adversary/weighted_log_probs/max', train_info['weighted_log_probs']['max'], steps)

    def log_env_info(self, eval_infos, steps):
        episode_rewards = {
                'episode_rewards/mean': eval_infos['episode_rewards'][-1]['mean'],
                'episode_rewards/std': eval_infos['episode_rewards'][-1]['std'], 
                'episode_rewards/min': eval_infos['episode_rewards'][-1]['min'],
                'episode_rewards/max': eval_infos['episode_rewards'][-1]['max'],
                'episode_rewards/overall_avg': eval_infos['episode_rewards'][-1]['overall_avg'],
            }
        best_z = {
                'best_z/mean': eval_infos['best_z']['mean'],
                'best_z/std': eval_infos['best_z']['std'],
                'best_z/min': eval_infos['best_z']['min'],
                'best_z/max': eval_infos['best_z']['max']
            }
        z_batch_stats = {
                'z_batch_stats/mean': eval_infos['best_batch_stats']['mean'],
                'z_batch_stats/std': eval_infos['best_batch_stats']['std'],
                'z_batch_stats/min': eval_infos['best_batch_stats']['min'],
                'z_batch_stats/max': eval_infos['best_batch_stats']['max'],
                'z_batch_stats/ranges/top/34%': eval_infos['best_batch_stats']['ranges']['top']['34%'],
                'z_batch_stats/ranges/top/13.6%': eval_infos['best_batch_stats']['ranges']['top']['13.6%'],
                'z_batch_stats/ranges/top/2.1%': eval_infos['best_batch_stats']['ranges']['top']['2.1%'],
                'z_batch_stats/ranges/top/0.1%': eval_infos['best_batch_stats']['ranges']['top']['0.1%'],
                'z_batch_stats/ranges/low/34%': eval_infos['best_batch_stats']['ranges']['low']['34%'],
                'z_batch_stats/ranges/low/13.6%': eval_infos['best_batch_stats']['ranges']['low']['13.6%'],
                'z_batch_stats/ranges/low/2.1%': eval_infos['best_batch_stats']['ranges']['low']['2.1%'],
                'z_batch_stats/ranges/low/0.1%': eval_infos['best_batch_stats']['ranges']['low']['0.1%']
            }
        if self.use_wandb:
            wandb.log(episode_rewards, step=steps)
            wandb.log(best_z, step=steps)
            # wandb.log(z_batch_stats, step=steps)
        else:
            self.writer.add_scalar('episode_rewards/mean', eval_infos['episode_rewards'][-1]['mean'], steps)
            self.writer.add_scalar('episode_rewards/std', eval_infos['episode_rewards'][-1]['std'], steps)
            self.writer.add_scalar('episode_rewards/min', eval_infos['episode_rewards'][-1]['min'], steps)
            self.writer.add_scalar('episode_rewards/max', eval_infos['episode_rewards'][-1]['max'], steps)
            self.writer.add_scalar('episode_rewards/overall_avg', eval_infos['episode_rewards'][-1]['overall_avg'], steps)
            self.writer.add_scalar('best_z/mean', eval_infos['best_z']['mean'], steps)
            self.writer.add_scalar('best_z/std', eval_infos['best_z']['std'], steps)
            self.writer.add_scalar('best_z/min', eval_infos['best_z']['min'], steps)
            self.writer.add_scalar('best_z/max', eval_infos['best_z']['max'], steps)
        
def get_z_generator(args, vae_model, device=None, run_dir=None):
    if args.vae_z_generator == "normal_gaussian":
        return NormalGaussianZ(args)
    elif args.vae_z_generator == "adversarial":
        return AdversarialZ(args, device, run_dir=run_dir)
    else:
        raise ValueError(f"Z generator {args.vae_z_generator} not defined")
