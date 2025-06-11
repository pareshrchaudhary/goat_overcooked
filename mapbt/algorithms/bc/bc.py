import numpy as np
import torch
import torch.nn as nn

from mapbt.algorithms.population.utils import NullTrainer
from .algorithm.BCPolicy import BCPolicy


class BC_Trainer(NullTrainer):
    def __init__(self,
                 args,
                 policy: BCPolicy,
                 device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.bc_validation_split = args.bc_validation_split
        self.batch_size = args.bc_batch_size
        
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
    
    def load_data(self, inputs, targets):
        num_samples = len(inputs)
        p = np.random.permutation(num_samples)
        num_validation_samples = int(num_samples * self.bc_validation_split)
        p_training = p[num_validation_samples:]
        p_validation = p[:num_validation_samples]
        self.training_data = {
            "inputs": inputs[p_training].copy(),
            "targets": targets[p_training].copy(),
        }
        self.validation_data = {
            "inputs": inputs[p_validation].copy(),
            "targets": targets[p_validation].copy(),
        }

    def fit_once(self):
        inputs, targets = self.training_data["inputs"], self.training_data["targets"]
        model = self.policy.actor
        optimizer = self.policy.actor_optimizer
        batch_size = self.batch_size
        num_samples = len(inputs)

        p = np.random.permutation(num_samples)
        correct_count, loss_sum, total_count = 0, 0, 0
        for i in range(0, num_samples, batch_size):
            idx = p[i:i+batch_size]
            input_batch = torch.FloatTensor(inputs[idx]).to(self.device)
            target_batch = torch.LongTensor(targets[idx]).to(self.device).reshape(-1)
            output_batch = model.get_action_logits(input_batch)

            loss = self.loss_fn(output_batch, target_batch)
            correct_count += (output_batch.max(dim=-1).indices == target_batch).sum().item()
            total_count += len(idx)
            loss_sum += loss.sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        info = {
            "training_accuracy": correct_count / total_count,
            "training_loss": loss_sum / total_count
        }
        return info

    @torch.no_grad()
    def validate(self):
        inputs, targets = self.validation_data["inputs"], self.validation_data["targets"]
        model = self.policy.actor
        batch_size = self.batch_size
        num_samples = len(inputs)

        p = np.arange(num_samples)
        correct_count, loss_sum, total_count = 0, 0, 0
        for i in range(0, num_samples, batch_size):
            idx = p[i:i+batch_size]
            input_batch = torch.FloatTensor(inputs[idx]).to(self.device)
            target_batch = torch.LongTensor(targets[idx]).to(self.device).reshape(-1)
            output_batch = model.get_action_logits(input_batch)

            loss = self.loss_fn(output_batch, target_batch)
            correct_count += (output_batch.max(dim=-1).indices == target_batch).sum().item()
            total_count += len(idx)
            loss_sum += loss.sum().item()
        
        info = {
            "validation_accuracy": correct_count / total_count,
            "validation_loss": loss_sum / total_count
        }
        return info
        

    def train(self, buffer, turn_on=True):
        raise NotImplementedError("BC policies should be freezed")

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
    
    def to(self, device):
        self.policy.to(device)
        self.loss_fn.to(device)
