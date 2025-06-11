import numpy as np
import torch
from tqdm import tqdm

from mapbt.runner.shared.overcooked_runner import OvercookedRunner as Runner
from .vae_model import VAEModel
from .trainer import VAETrainer


class OvercookedRunner(Runner):
    def train_vae(self):
        self.vae_model = VAEModel(*self.policy_config, device=self.device)
        vae_model_dir = self.all_args.vae_model_dir
        if vae_model_dir is not None:
            self.load_vae(vae_model_dir, self.all_args.vae_model_name)
        self.vae_trainer = VAETrainer(self.all_args, device=self.device, num_actions=self.envs.action_space[0].n, vae=self.vae_model)

        kl_spectrum = [int(np.round(kl)) if kl >= 1 else np.round(kl, 1) for kl in np.logspace(2, -1, 19)]
        kl_spectrum = np.unique(kl_spectrum)
        kl_spectrum = [int(kl) if kl >= 1 else kl for kl in kl_spectrum]
        
        best_acc_wrt_kl = {
            kl: 0.0
            for kl in kl_spectrum
        }
        best_log_likelihood_wrt_kl = {
            kl: -1e9
            for kl in kl_spectrum
        }
        total_num_steps, num_epochs = 0, self.all_args.vae_epoch
        for epoch in tqdm(range(num_epochs)):
            self.vae_trainer.kl_decay(epoch, num_epochs)

            train_info = self.vae_trainer.train_one_epoch()
            total_num_steps += self.vae_trainer.lastrun_gradient_steps
            validation_info = self.vae_trainer.validate_one_epoch()

            for kl in best_acc_wrt_kl:
                if validation_info["kl"] < kl and validation_info["acc"] > best_acc_wrt_kl[kl]:
                    best_acc_wrt_kl[kl] = validation_info["acc"]
                    self.save_vae(f"best_acc_kl_{kl if isinstance(kl, int) else np.round(kl, 1)}")
                if validation_info["kl"] < kl and validation_info["log_likelihood"] > best_log_likelihood_wrt_kl[kl]:
                    best_log_likelihood_wrt_kl[kl] = validation_info["log_likelihood"]
                    self.save_vae(f"best_log_likelihood_kl_{kl if isinstance(kl, int) else np.round(kl, 1)}")

            if epoch % self.log_interval == 0:
                print("\n ========== train_info ========== \n")
                print({k: v for k, v in train_info.items() if "/" not in k})
                print("\n ========== validation_info ========== \n")
                print(validation_info)
                self.log_train({f"vae_train/{k}": v for k, v in train_info.items()}, total_num_steps)
                self.log_train({f"vae_validation/{k}": v for k, v in validation_info.items()}, total_num_steps)

            if epoch % self.save_interval == 0 or epoch == num_epochs - 1:
                self.save_vae(f"ckp_epoch_{epoch}")

            if self.use_eval and epoch % self.eval_interval == 0:
                eval_info = self.vae_trainer.eval_one_epoch()
                self.log_train({f"vae_heldout_test/{k}": v for k, v in eval_info.items()}, total_num_steps)
    
    def get_evaluation_policies(self) -> list[str]:
        policies = []
        for policy_name, (_, policy_info) in self.policy.policy_info.items():
            if policy_info.get("evaluation_agent"):
                policies.append(policy_name)
        return policies
    
    @torch.no_grad()
    def evaluate_vae(self):
        self.vae_model = VAEModel(*self.policy_config, device=self.device)
        vae_model_dir = self.all_args.vae_model_dir
        if vae_model_dir is None:
            raise ValueError("vae_model_dir is not specified")
        self.load_vae(vae_model_dir, self.all_args.vae_model_name)
        self.vae_trainer = VAETrainer(self.all_args, device=self.device, num_actions=self.envs.action_space[0].n, vae=self.vae_model)

        # TODO: evaluation

    def save_vae(self, name):
        torch.save(self.vae_model.encoder.state_dict(), str(self.save_dir) + "/encoder_{}.pt".format(name))
        torch.save(self.vae_model.decoder.state_dict(), str(self.save_dir) + "/decoder_{}.pt".format(name))

    def load_vae(self, model_dir, name):
        if name is None:
            name = ""
        else:
            name = f"_{name}"
        self.vae_model.encoder.load_state_dict(torch.load(str(model_dir) + "/encoder{}.pt".format(name)))
        self.vae_model.decoder.load_state_dict(torch.load(str(model_dir) + "/decoder{}.pt".format(name)))

    @torch.no_grad()
    def _render_data_support_vae(self):
        # TODO: a newer version
        self.vae_model = VAEModel(*self.policy_config, device=self.device)
        vae_model_dir = getattr(self.all_args, "vae_model_dir", None)
        if vae_model_dir is None:
            raise ValueError("vae_model_dir is not specified")
        self.load_vae(vae_model_dir, self.all_args.vae_model_name)
        
        try:
            assert self.all_args.eval_stochastic
        except Exception as e:
            print(f"Exception {e}")
            print("highly suggest using stochastic action sampling")

        print("\n==========\nstart generate random vae agents for sp...\n")
        print(f"{self.all_args.render_episodes} episodes with z~N(0,1) will be generated")

        obs, share_obs, available_actions = self.envs.reset()
        from icecream import ic
        obs = np.stack(obs)
        ic(obs.shape)
        assert self.n_rollout_threads == 1
        
        # Copy and render traj
        from .hdf5_dataset import New_HDF5Dataset as HDF5Dataset
        self.dataset = HDF5Dataset(self.all_args, "train")
        ref_obs, ref_actions = self.dataset.get_start_matching_episode(start_obs=obs)
        print("ref_actions.shape", ref_actions.shape)
        for episode in range(self.all_args.render_episodes):
            # z_rand = np.random.normal(0, 1, (1, self.all_args.z_dim)).repeat(self.num_agents, 0)
            # decoder_rnn_states = np.repeat([self.vae_model.decoder.default_hidden_state], self.num_agents, 0)
            # decoder_masks = np.ones((self.num_agents, 1), dtype=np.float32)
            for step in range(self.episode_length):
                actions = np.full((self.n_rollout_threads, self.num_agents, 1), fill_value=0).tolist()
                # action_distribution, decoder_rnn_states = self.vae_model.decoder.get_output_distribution(
                #     np.concatenate(obs),
                #     rnn_states=decoder_rnn_states,
                #     masks=decoder_masks,
                #     z=z_rand,
                #     return_rnn_states = True,
                # )
                # action_sample = action_distribution.sample()
                actions[0] = ref_actions[step]

                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                obs = np.stack(obs)
        
        assert self.all_args.vae_chunk_length == 100
        chunk_obs = ref_obs[:-1].reshape(4, 100, *ref_obs.shape[1:])
        chunk_actions = ref_actions.reshape(4, 100, *ref_actions.shape[1:])
        chunk_obs = torch.FloatTensor(chunk_obs)
        chunk_actions = torch.LongTensor(chunk_actions)
        z_human = self.vae_model.encode(chunk_obs, chunk_actions, sampling="mean")
        print(z_human.shape)
        # Follow Z
        assert self.all_args.render_episodes == 1
        for episode in range(4 * self.all_args.render_episodes):
            z_rand = z_human[episode] # np.random.normal(0, 1, (1, self.all_args.z_dim)).repeat(self.num_agents, 0)
            decoder_rnn_states = np.repeat([self.vae_model.decoder.default_hidden_state], self.num_agents, 0)
            decoder_masks = np.ones((self.num_agents, 1), dtype=np.float32)
            for step in range(self.all_args.vae_chunk_length):
                actions = np.full((self.n_rollout_threads, self.num_agents, 1), fill_value=0).tolist()
                action_distribution, decoder_rnn_states = self.vae_model.decoder.get_output_distribution(
                    np.concatenate(obs),
                    rnn_states=decoder_rnn_states,
                    masks=decoder_masks,
                    z=z_rand,
                    return_rnn_states = True,
                )
                action_sample = action_distribution.sample()
                actions[0] = action_sample.detach().clone().cpu().numpy()

                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                obs = np.stack(obs)
        print("self.gif_dir", self.gif_dir) # + layout + traj_num_?
        # move or copy gifs