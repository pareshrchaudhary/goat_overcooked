import os

import numpy as np
import torch

from ..vae_constructor.vae_model import VAEModel


class VAEAgent:
    def __init__(self, args, vae: VAEModel):
        self.vae = vae
        self.load_decoder(args.vae_model_dir)

        self.num_agents = args.num_agents
        self.vae_encoder_input = args.vae_encoder_input
        
        self.share_obs = [None for _ in range(args.episode_length + 1)]
        self.obs = [None for _ in range(args.episode_length + 1)]
    
    def init_first_step(self, share_obs, obs):
        self.share_obs[0] = share_obs.copy()
        self.obs[0] = np.stack(obs).copy()
        self._step = 0
        self._rnn_states = None
    
    def insert_data(self, share_obs, obs):
        self._step += 1
        self.share_obs[self._step] = share_obs.copy()
        self.obs[self._step] = np.stack(obs).copy()
    
    def step(self, step, z):
        vae_state = []
        z_input = []
        ea2i = {}
        for i, (e, a) in enumerate(z):
            assert self.vae_encoder_input in ["ego_obs", "partner_obs", "ego_share_obs"]
            assert self.num_agents == 2
            if self.vae_encoder_input == "ego_obs":
                vae_state.append(self.obs[step][e, 1 - a])
            elif self.vae_encoder_input == "partner_obs":
                vae_state.append(self.obs[step][e, a])
            elif self.vae_encoder_input == "ego_share_obs":
                vae_state.append(self.share_obs[step][e, 1 - a])
            else:
                raise NotImplementedError
            z_input.append(z[(e, a)])
            ea2i[(e, a)] = i
        vae_state = np.array(vae_state)
        z_input = np.array(z_input)
        if self._rnn_states is None:
            self._rnn_states = np.repeat([self.vae.decoder.default_hidden_state], len(vae_state), 0)
        action_distribution, self._rnn_states = self.vae.decoder.get_output_distribution(
            vae_state,
            rnn_states=self._rnn_states,
            masks=np.ones((len(vae_state), 1), dtype=np.float32),
            z=z_input,
            return_rnn_states=True,
        )
        action_sample = action_distribution.sample()
        action_sample = action_sample.detach().clone().cpu().numpy()
        res_actions = {}
        for (e, a), i in ea2i.items():
            res_actions[(e, a)] = action_sample[i]
        return res_actions
    
    def load_decoder(self, model_dir):
        try:
            encoder_dir = os.path.join(model_dir, "encoder.pt")
            self.vae.encoder.load_state_dict(torch.load(encoder_dir))
        except Exception as e:
            print(f"encoder not loaded for exception {e}")
        decoder_dir = os.path.join(model_dir, "decoder.pt")
        self.vae.decoder.load_state_dict(torch.load(decoder_dir))