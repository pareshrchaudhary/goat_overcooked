import torch
import torch.nn as nn

from mapbt.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from mapbt.algorithms.bc.algorithm.actor_xout import R_Actor_XOut


class Null_Critic(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Null_Critic, self).__init__()
        self._pad_fc = nn.Linear(1, 1)

class BCPolicy(R_MAPPOPolicy):
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):

        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.actor = R_Actor_XOut(args, self.obs_space, self.act_space, self.device)
        self.critic = Null_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)
