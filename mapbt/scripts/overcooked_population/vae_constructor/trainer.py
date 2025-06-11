from collections import defaultdict

import numpy as np
import torch
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, SubsetRandomSampler

from .hdf5_dataset import HDF5Dataset
from .vae_model import VAEModel


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def memory_usage():
    import os
    import psutil # TODO: remove
    """Return the memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    m_usage = memory_info.rss / (1024 * 1024 * 1024)  # Convert to MB
    print("Memory usage:", m_usage, "GB")

def split_dataset(dataset, validation_ratio):
    random_indices = np.random.permutation(len(dataset))
    validation_split = int(validation_ratio * len(random_indices))
    train_indices, validation_indices = random_indices[:-validation_split], random_indices[-validation_split:]
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    return train_sampler, validation_sampler


def get_model_weight_grad_info(model_param_dict, model_name="model"):
    info = {}
    sum_weight, sum_grad = 0, 0
    for name, x in model_param_dict:
        info[f"{model_name}_weight/{name}"] = x.norm().item()
        sum_weight += x.norm().item() ** 2
        if x.grad is not None:
            sum_grad += x.grad.norm().item() ** 2
    info.update({
        f"{model_name}_weight_norm": np.sqrt(sum_weight),
        f"{model_name}_grad_epoch_mean": np.sqrt(sum_grad),
        f"{model_name}_grad_epoch_max": np.sqrt(sum_grad),
    })
    return info


class VAETrainer:
    def __init__(self, args, device, num_actions: int, vae: VAEModel):
        self.device = device
        self.num_actions = num_actions
        self.vae = vae

        self.z_dim = args.z_dim
        self.lr = args.vae_lr
        self.kl_0 = args.vae_init_kl_penalty or args.vae_kl_penalty
        self.kl_T = args.vae_kl_penalty
        self.kl_penalty = args.vae_kl_penalty
        self.minibatch_size = args.vae_batch_size
        self.evaluation_batch_size = args.evaluation_batch_size
        
        self.vae_chunk_length = args.vae_chunk_length
        self.vae_ego_agents = args.vae_ego_agents
        self.vae_encoder_input = args.vae_encoder_input

        self.freeze_encoder = args.vae_train_freeze_encoder

        self.dataset = HDF5Dataset(args, "train")
        train_sampler, validation_sampler = split_dataset(self.dataset, args.dataset_validation_ratio)
        self.train_loader = DataLoader(self.dataset, batch_size=self.minibatch_size, sampler=train_sampler, num_workers=4, prefetch_factor=4)
        self.validation_loader = DataLoader(self.dataset, batch_size=self.evaluation_batch_size, sampler=validation_sampler, num_workers=8, prefetch_factor=4)
        self.heldout_dataset = HDF5Dataset(args, "test")
        self.test_loader = DataLoader(self.heldout_dataset, batch_size=self.evaluation_batch_size, shuffle=False, num_workers=8, prefetch_factor=4)
        
    def kl_decay(self, epoch, epochs):
        self.kl_penalty = self.kl_0 + epoch / epochs * (self.kl_T - self.kl_0)
    
    def get_vae_loss(self, input_states, input_actions, output_states, target_actions):
        """
        Variables:
            input_states: (L_input, B, *State Shape)
            input_actions: (L_input, B, 1)
            output_states: (L_output, B, *State Shape)
            target_actions: (L_output, B, 1)
        Encoding:
            seq[input_states, input_actions] -> z: [B, Z]
        Reconstruction:
            seq[output_states, z] -> predict_actions: (L_output, B, 1)
            predict_actions <---logp--- target_actions
        """
        pass

    def run_epoch(self, data_loader, update_vae=False):
        epoch_info = defaultdict(list)
        self.lastrun_gradient_steps = 0
        for i, (obs, actions, policy_id) in enumerate(data_loader):
            if update_vae:
                self.lastrun_gradient_steps += 1
            data = self.process_data(obs, actions, policy_id)
            info, sample = self.vae_objective(
                vae_states=data["vae_states"],
                vae_onehot_actions=data["vae_onehot_actions"],
                vae_target_actions=data["vae_target_actions"],
                ego_policies=data["ego_policies"],
                partner_policies=data["partner_policies"],
                update_vae=update_vae,
            )
            for k, v in info.items():
                epoch_info[k].append(v)
        summary_info = {}
        for k, v in epoch_info.items():
            if k.endswith("_epoch_max"):
                summary_info[k] = np.max(v)
            else:
                summary_info[k] = np.mean(v)
        return summary_info

    @torch.no_grad()
    def eval_one_epoch(self):
        self.heldout_dataset.reset()
        info = self.run_epoch(self.test_loader, update_vae=False)
        return info
    
    @torch.no_grad()
    def validate_one_epoch(self):
        self.dataset.reset()
        info = self.run_epoch(self.validation_loader, update_vae=False)
        return info
    
    def train_one_epoch(self):
        self.dataset.reset()
        info = self.run_epoch(self.train_loader, update_vae=True)
        return info

    def process_data(self, obs, actions, policy_id):
        # print("obs.shape", obs.shape)
        # print("actions.shape", actions.shape)
        # print("policy_id.shape", policy_id.shape)
        chunk_length = self.vae_chunk_length if self.vae_chunk_length > 0 else episode_length
        batch_size = obs.shape[0]
        data = {
            "num_env_steps": chunk_length * batch_size,
            "vae_states": [],
            "vae_onehot_actions": [],
            "vae_target_actions": [],
            "ego_policies": [],
            "partner_policies": [],
        }
        if self.vae_encoder_input == "partner_obs":
            device = self.device # self.device
            obs = obs.to(device)
            actions = actions.to(device)
            data["vae_states"] = torch.cat([obs[:, :, 1], obs[:, :, 0]], 0).swapaxes(0, 1)
            data["vae_onehot_actions"] = torch.cat([actions[:, :, 1], actions[:, :, 0]], 0).swapaxes(0, 1).squeeze(-1)
            data["vae_onehot_actions"] = torch.eye(self.num_actions).to(device)[data["vae_onehot_actions"]]
            data["vae_target_actions"] = torch.cat([actions[:, :, 1], actions[:, :, 0]], 0).swapaxes(0, 1).squeeze(-1)
        else:
            raise NotImplementedError
        return data
    
    def vae_objective(self, vae_states, vae_onehot_actions, vae_target_actions, ego_policies, partner_policies, update_vae=True):
        chunk_length, batch_size, _num_actions = vae_onehot_actions.shape

        if vae_states.device == self.device:
            states_batch = vae_states
            actions_batch = vae_onehot_actions
            targets_batch = vae_target_actions
        else:
            states_batch = torch.FloatTensor(vae_states).to(self.device)
            actions_batch = torch.FloatTensor(vae_onehot_actions).to(self.device)
            targets_batch = torch.LongTensor(vae_target_actions).to(self.device)

        states_flatten = _flatten(chunk_length, batch_size, states_batch)
        actions_flatten = _flatten(chunk_length, batch_size, actions_batch)
        targets_flatten = _flatten(chunk_length, batch_size, targets_batch)

        # ic(states_batch.shape, actions_batch.shape, targets_batch.shape)
        # ic| states_batch.shape: torch.Size([L, B, Obs_Shape])
        #     actions_batch.shape: torch.Size([L, B, Num_Action])
        #     targets_batch.shape: torch.Size([L, B])
        # flatten: [L * B, ...]

        encoder_rnn_states = np.repeat([self.vae.encoder.default_hidden_state], batch_size, 0)
        encoder_masks = np.ones((chunk_length * batch_size, 1), dtype=np.float32)
        all_q_z: Normal = self.vae.encoder.get_output_distribution(
            states_flatten,
            rnn_states=encoder_rnn_states,
            masks=encoder_masks,
            z=actions_flatten,
        )
        all_mu, all_std = all_q_z.mean.reshape(chunk_length, batch_size, self.z_dim), all_q_z.stddev.reshape(chunk_length, batch_size, self.z_dim)
        q_z = Normal(all_mu[-1], all_std[-1])
        z = q_z.rsample()
        _z_mean = q_z.mean
        # ic(z.shape)
        # ic| z.shape: torch.Size([B, Z])

        z_batch = z.repeat(chunk_length, 1, 1) # [L, B, Z]
        z_flatten = _flatten(chunk_length, batch_size, z_batch) # [L * B, Z]
        decoder_rnn_states = np.repeat([self.vae.decoder.default_hidden_state], batch_size, 0)
        decoder_masks = np.ones((chunk_length * batch_size, 1), dtype=np.float32)
        p_x = self.vae.decoder.get_output_distribution(
            states_flatten,
            rnn_states=decoder_rnn_states,
            masks=decoder_masks,
            z=z_flatten,
        ) # [B * L, Num_Action] Categorical distribution
        pi_targets = targets_flatten
        
        correct_guess = (p_x.probs.argmax(-1) == pi_targets).to(torch.float32)
        meaningful_action_mask = (pi_targets != 4) # Note! In overcooked, action 4 is the idle action
        acc, meaningful_acc = correct_guess.mean(), (correct_guess * meaningful_action_mask).sum() / (meaningful_action_mask.sum() + 1e-5)
        log_likelihood = p_x.log_prob(pi_targets).reshape(chunk_length, batch_size)
        log_likelihood = log_likelihood.sum(0)
        entropy = p_x.entropy()
        kl = torch.distributions.kl_divergence(
            q_z,
            torch.distributions.Normal(0., 1.,)
        ).sum(-1)

        log_likelihood = log_likelihood.mean()
        kl = kl.mean()
        entropy = entropy.mean()
        loss = -(log_likelihood - self.kl_penalty * kl)
        
        info_log = {
            "log_likelihood": log_likelihood.item(),
            "acc": acc.item(),
            "meaningful_acc": meaningful_acc.item(),
            "entropy": entropy.item(),
            "kl": kl.item(),
            "loss": loss.item(),
        }

        if update_vae:
            self.vae.encoder_optimizer.zero_grad()
            self.vae.decoder_optimizer.zero_grad()
            loss.backward()
            info_log.update(get_model_weight_grad_info(
                model_param_dict=self.vae.encoder.named_parameters(),
                model_name="encoder",
            ))
            info_log.update(get_model_weight_grad_info(
                model_param_dict=self.vae.decoder.named_parameters(),
                model_name="decoder",
            ))
            if not self.freeze_encoder:
                self.vae.encoder_optimizer.step()
            self.vae.decoder_optimizer.step()

        return info_log, {
            "_ego_policies": ego_policies,
            "_partner_policies": partner_policies,
            "_z_mean": list(_z_mean.detach().clone().cpu().numpy()),
            "z": list(z.detach().clone().cpu().numpy()),
            "correct_guess": correct_guess.detach().clone().cpu().numpy(),
        }
