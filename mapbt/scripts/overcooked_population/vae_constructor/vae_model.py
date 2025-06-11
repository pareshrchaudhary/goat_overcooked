import os
from copy import deepcopy

from gymnasium.spaces import Box
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from mapbt.utils.util import update_linear_schedule
from mapbt.algorithms.utils.util import init, check
from mapbt.algorithms.utils.cnn import CNNBase
from mapbt.algorithms.utils.mlp import MLPBase, MLPLayer
from mapbt.algorithms.utils.mix import MIXBase
from mapbt.algorithms.utils.rnn import RNNLayer
from mapbt.algorithms.utils.act import ACTLayer
from mapbt.algorithms.utils.popart import PopArt
from mapbt.utils.util import get_shape_from_obs_space
from mapbt.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from mapbt.algorithms.bc.algorithm.actor_xout import ACTLayer_XOut


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

class Add_Z_MLP(nn.Module):
    def __init__(self, args, feature_dim, z_dim, hidden_dim, output_dim):
        super().__init__()

        active_fn = [nn.Tanh, nn.ReLU, nn.LeakyReLU, nn.ELU][args.activation_id]
        self.z_mlp = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            active_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            active_fn(),
            nn.LayerNorm(hidden_dim),
        )
        self.post_mlp = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, output_dim),
            active_fn(),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, feature, z):
        z_feature = self.z_mlp(z)
        feature = self.post_mlp(torch.cat([feature, z_feature], -1))
        return feature

class Flatten_Z_Actor(R_Actor):
    def __init__(self, args, input_space, z_dim, output_space, device=torch.device("cpu")):
        obs_space, action_space = input_space, output_space
        self._z_dim = z_dim

        nn.Module.__init__(self)
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal 
        self._activation_id = args.activation_id
        self._use_policy_active_masks = args.use_policy_active_masks 
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._influence_layer_N = args.influence_layer_N 
        self._use_policy_vhead = args.use_policy_vhead
        self._use_popart = args.use_popart 
        self._recurrent_N = args.recurrent_N 
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)

        if 'Dict' in obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(args, obs_shape, cnn_layers_params=args.cnn_layers_params)
        else:
            self._mixed_obs = False
            self.base = CNNBase(args, obs_shape, cnn_layers_params=args.cnn_layers_params) if len(obs_shape)==3 \
                else MLPBase(args, obs_shape, use_attn_internal=args.use_attn_internal, use_cat_self=True)
        
        input_size = self.base.output_size

        # argument with z input
        self.add_z_feature_mlp = Add_Z_MLP(
            args,
            feature_dim=input_size,
            z_dim=self._z_dim,
            hidden_dim=input_size,
            output_dim=input_size
        )

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(obs_shape[0], self.hidden_size,
                              self._influence_layer_N, self._use_orthogonal, self._activation_id)
            input_size += self.hidden_size

        self.act = ACTLayer_XOut(action_space, input_size, self._use_orthogonal, self._gain)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        if self._use_policy_vhead:
            if self._use_popart:
                self.v_out = init_(PopArt(input_size, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(input_size, 1))
        
        # in Overcooked, predict shaped info
        self._predict_other_shaped_info = False
        if args.env_name == "Overcooked" and getattr(args, "predict_other_shaped_info", False):
            self._predict_other_shaped_info = True
            self.pred_shaped_info = init_(nn.Linear(input_size, 12))

        self.to(device)

    @property
    def default_hidden_state(self):
        return np.zeros((self._recurrent_N, self.hidden_size), dtype=np.float32)

    def get_output_distribution(self, obs, rnn_states=None, masks=None, available_actions=None, deterministic=False, z=None, return_rnn_states=False):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        if rnn_states is not None:
            rnn_states = check(rnn_states).to(**self.tpdv)
        if masks is not None:
            masks = check(masks).to(**self.tpdv)
        if z is not None:
            z = check(z).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        
        # add latent z
        if z is not None:
            actor_features = self.add_z_feature_mlp(actor_features, z)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        if return_rnn_states:
            return self.act.action_distribution(actor_features, available_actions, deterministic), rnn_states
        return self.act.action_distribution(actor_features, available_actions, deterministic)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        raise NotImplementedError

    def evaluate_transitions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        # ! only work for rnn model
        raise NotImplementedError

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        raise NotImplementedError

    def get_policy_values(self, obs, rnn_states, masks):
        raise NotImplementedError
    
    def get_probs(self, obs, rnn_states, masks, available_actions=None):
        raise NotImplementedError
    
    def get_action_probs(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        raise NotImplementedError

class VAEModel:
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        self.z_dim = args.z_dim
        self.lr = getattr(args, "vae_lr", args.lr)

        self.device = device
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        assert args.vae_encoder_input in ["ego_obs", "partner_obs", "ego_share_obs"]
        self.state_space = share_obs_space if "share_obs" in args.vae_encoder_input else obs_space

        encoder_args = deepcopy(args)
        if getattr(args, "vae_hidden_size", -1) > 0:
            encoder_args.hidden_size = args.vae_hidden_size
        encoder_args.use_naive_recurrent_policy, encoder_args.use_recurrent_policy = False, True
        z_output_space = Box(
            -np.ones(self.z_dim) * float("inf"),
            np.ones(self.z_dim,) * float("inf"),
            dtype=np.float32
        )
        if self.act_space.__class__.__name__ == 'Discrete':
            self.encoder = Flatten_Z_Actor(encoder_args, input_space=self.state_space, z_dim=self.act_space.n, output_space=z_output_space, device=self.device)
        else:
            raise NotImplementedError

        decoder_args = deepcopy(args)
        if getattr(args, "vae_hidden_size", -1) > 0:
            decoder_args.hidden_size = args.vae_hidden_size
        decoder_args.use_naive_recurrent_policy, decoder_args.use_recurrent_policy = False, True
        self.decoder = Flatten_Z_Actor(decoder_args, input_space=self.state_space, z_dim=self.z_dim, output_space=self.act_space, device=self.device)

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def get_q_z(self, states, actions) -> Normal:
        chunk_length, batch_size, num_actions = actions.shape
        states_flatten = _flatten(chunk_length, batch_size, states)
        actions_flatten = _flatten(chunk_length, batch_size, actions)
        encoder_rnn_states = np.repeat([self.encoder.default_hidden_state], batch_size, 0)
        encoder_masks = np.ones((chunk_length * batch_size, 1), dtype=np.float32)
        seq_q_z: Normal = self.encoder.get_output_distribution(
            states_flatten,
            rnn_states=encoder_rnn_states,
            masks=encoder_masks,
            z=actions_flatten,
        )
        seq_mean, seq_std = seq_q_z.mean.reshape(chunk_length, batch_size, self.z_dim), seq_q_z.stddev.reshape(chunk_length, batch_size, self.z_dim)
        q_z = Normal(seq_mean[-1], seq_std[-1])
        return q_z

    def get_p_x(self, states, z) -> Categorical:
        chunk_length, batch_size = states.shape[:2]
        states_flatten = _flatten(chunk_length, batch_size, states)
        z_batch = z.repeat(chunk_length, 1, 1) # [L, B, Z]
        z_flatten = _flatten(chunk_length, batch_size, z_batch) # [L * B, Z]
        decoder_rnn_states = np.repeat([self.decoder.default_hidden_state], batch_size, 0)
        decoder_masks = np.ones((chunk_length * batch_size, 1), dtype=np.float32)
        p_x = self.decoder.get_output_distribution(
            states_flatten,
            rnn_states=decoder_rnn_states,
            masks=decoder_masks,
            z=z_flatten,
        ) 
        return p_x

    def encode(self, obs, actions, sampling="none"):
        """Input Format:
            obs: [batch_size, chunk_length, num_agents, *obs_shape]
            actions: [batch_size, chunk_length, num_actions, 1]
        Output Format:
            z: [batch_size, num_agent, z_dim]
                none: (mean, std)
                rsample: rsample()
                sample: sample()
                mean: mean
        """
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        batch_size, chunk_length, num_agents = obs.shape[:3]
        num_actions = self.act_space.n
        if True: # TODO: partner_obs
            obs = torch.cat([obs[:, :, 0], obs[:, :, 1]], 0).swapaxes(0, 1)
            actions = torch.cat([actions[:, :, 0], actions[:, :, 1]], 0).swapaxes(0, 1).squeeze(-1)
            onehot_actions = torch.eye(num_actions).to(self.device)[actions]
        q_z = self.get_q_z(obs, onehot_actions)
        q_z = Normal(q_z.mean.reshape(batch_size, num_agents, self.z_dim), q_z.stddev.reshape(batch_size, num_agents, self.z_dim))
        if sampling == "none":
            raise NotImplementedError
        elif sampling == "rsample":
            raise NotImplementedError
        elif sampling == "sample":
            raise NotImplementedError
        elif sampling == "mean":
            return q_z.mean
        else:
            raise NotImplementedError
    
    def decode(self, obs, z):
        """Input Format:
            obs: [batch_size, chunk_length, num_agents, *obs_shape]
            z: [batch_size, num_actions, z_dim]
        Output Format:
            p_x: [batch_size, chunk_length, num_agents, num_actions]
        """
        batch_size, chunk_length, num_agents = obs.shape[:3]
        if True: # TODO: partner_obs
            obs = torch.cat([obs[:, :, 0], obs[:, :, 1]], 0).swapaxes(0, 1)
            z = z.reshape(batch_size * num_agents, self.z_dim)
        return self.get_p_x(obs, z)
    
    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.encoder_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.decoder_optimizer, episode, episodes, self.lr)
    
    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
