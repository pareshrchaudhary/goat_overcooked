import torch
import torch.nn as nn

from mapbt.algorithms.utils.util import init, check
from mapbt.algorithms.utils.cnn import CNNBase
from mapbt.algorithms.utils.mlp import MLPBase, MLPLayer
from mapbt.algorithms.utils.mix import MIXBase
from mapbt.algorithms.utils.rnn import RNNLayer
from mapbt.algorithms.utils.act import ACTLayer
from mapbt.algorithms.utils.popart import PopArt
from mapbt.utils.util import get_shape_from_obs_space
from mapbt.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor


class ACTLayer_XOut(ACTLayer):    
    def action_distribution(self, x, available_actions=None, deterministic=False):
        if self.mixed_action:
            action_distribution_list = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_distribution_list.append(action_logit)
            return action_distribution_list
        elif self.multidiscrete_action:
            action_logits = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_logits.append(action_logit)
            return action_logits
        elif self.continuous_action:
            return self.action_out(x)        
        else:
            action_logits = self.action_out(x, available_actions)        
            return action_logits

class R_Actor_XOut(R_Actor):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
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

    def get_action_logits(self, obs, rnn_states=None, masks=None, available_actions=None, deterministic=False):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        if rnn_states is not None:
            rnn_states = check(rnn_states).to(**self.tpdv)
        if masks is not None:
            masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        return self.act.action_distribution(actor_features, available_actions, deterministic).logits
