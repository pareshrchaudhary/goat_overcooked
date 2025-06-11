import json
import os
import random
import shutil
import string
import sys
import yaml

POP_DATA_TOPDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pop_data")


def generate_random_string(length):
    """Generate a random string of given length."""
    characters = string.hexdigits  # Hexadecimal characters (0-9, a-f)
    return ''.join(random.choice(characters) for _ in range(length))

def clear_folder(folder_path):
    """Clear all files within a folder."""
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # Check if the path is a file (not a directory)
        if os.path.isfile(file_path):
            # Delete the file
            os.remove(file_path)


class PopulationController:
    def __init__(self, pop_name, config):
        self.pop_name = pop_name
        self.layout = config["layout"]
        if "render_scripts" in config:
            self.render_script_name = config["render_scripts"]
        else:
            self.render_script_name = "render_agents_in_pop.sh"
        self.pop = {}
    
    def register_local_policy(self, policy_name, policy_info, policy_level, local_ckp_path):
        name = f"{policy_name}_{policy_level}"
        assert name not in self.pop
        self.pop[name] = {
            "local_ckp_path": local_ckp_path,
            "held_out": policy_info.get("held_out", False)
        }
    
    def copy_all_checkpoints(self, clear_existed_models=True):
        print("\n\n===== start moving all checkpoints to pop_name/models dir =====\n\n")
        
        pop_model_dir = os.path.join(POP_DATA_TOPDIR, self.pop_name, "models")
        if os.path.exists(pop_model_dir) and clear_existed_models:
            if input(f"All models under {pop_model_dir} will be removed, press (Y/N) to continue: ").strip().upper() == "Y":
                clear_folder(pop_model_dir)
                print("All existed models are removed")
        os.makedirs(pop_model_dir, exist_ok=True)

        for policy, info in self.pop.items():
            if "local_ckp_path" in info:
                shutil.copyfile(info["local_ckp_path"], os.path.join(pop_model_dir, f"{policy}_actor.pt"))
            else:
                raise NotImplementedError
        
        print("\n\n===== finish moving all checkpoints to pop_name/models dir =====\n\n")
    
    def visualize_all_policies(self):
        print("\n\n===== start visualization =====\n\n")

        gif_dir = os.path.join(POP_DATA_TOPDIR, self.pop_name, "visualization")
        os.makedirs(gif_dir, exist_ok=True)

        for policy in self.pop:
            exp_name = generate_random_string(20)
            layout_name = self.layout
            model_dir = os.path.join(POP_DATA_TOPDIR, self.pop_name, "models", f"{policy}_actor.pt")
            bash_file_path = os.path.relpath(os.path.join(os.path.dirname(__file__), self.render_script_name))
            print(f"\nvisualzie policy {policy} with command:\nbash {bash_file_path} {exp_name} {layout_name} {model_dir}\n")
            try:
                os.system(f"bash {bash_file_path} {exp_name} {layout_name} {model_dir}")
            except Exception as e:
                print(f"rendering failed with error {e}")
            assert not os.path.exists("./results/Overcooked/{self.layout}/mappo/{exp_name}/run1"), "seems multiple runs under the random render directory exist"
            destination_dir= os.path.join(gif_dir, policy)
            if os.path.exists(destination_dir):
                shutil.rmtree(destination_dir)
            shutil.copytree(
                f"./results/Overcooked/{self.layout}/mappo/{exp_name}/run1/gifs/{self.layout}/traj_num_1",
                destination_dir
            )
        
        print("\n\n===== finish visualization =====\n\n")

    def create_vae_yml_config(self):
        yml_dict = {}
        for policy, info in self.pop.items():
            yml_dict[policy] = {
                "policy_config_path": f"config/{self.layout}/mlp_policy_config.pkl",
                "featurize_type": "ppo",
                "train": False,
                "model_path": {
                    "actor": f"pop_data/{self.pop_name}/models/{policy}_actor.pt"
                },
                "held_out": info["held_out"]
            }
        with open(os.path.join(POP_DATA_TOPDIR, self.pop_name, "pop_for_vae_config.yml"), "w") as f:
            yaml.dump(yml_dict, f, indent=4)

def get_step_rew_info(
    local_summary=None,
    web_summary=None,
    policy_name=None,
):
    if local_summary:
        summary = local_summary
        if policy_name.startswith("sp"):
            for info_key in summary:
                if info_key.endswith("ep_sparse_r/ep_sparse_r"):
                    sparse_info = summary[info_key]
                    print(f"For policy {policy_name}, the attribute {info_key} of the summary file is used to get reward\n")
                    break
            step_rew_info = []
            for _, step, rew in sparse_info:
                step_rew_info.append((step, rew))
            return step_rew_info
        elif policy_name.startswith("mep"):
            for info_key in summary:
                if info_key.endswith(f"{policy_name}-{policy_name}-ep_sparse_r/{policy_name}-{policy_name}-ep_sparse_r"):
                    sparse_info = summary[info_key]
                    print(f"For policy {policy_name}, the attribute {info_key} of the summary file is used to get reward\n")
                    break
            step_rew_info = []
            for _, step, rew in sparse_info:
                step_rew_info.append((step, rew))
            return step_rew_info
        else:
            raise NotImplementedError(f"unspecified policy name [{policy_name}]")
    if web_summary:
        pass
    raise RuntimeError("at least one of local/web summary should be provided")

def get_local_checkpoints(directory_path):
    # Get a list of direct checkpoints
    def is_actor_ckp(s):
        return s.startswith("actor_periodic_") and s.endswith(".pt")
    checkpoints = [f for f in os.listdir(directory_path) if is_actor_ckp(str(f))]

    # Now, 'get_checkpoints' contains the id -> names of the direct checkpoints
    def get_ckp_id(s):
        s = str(s)
        assert is_actor_ckp(s)
        return int(s.split("actor_periodic_")[1].split(".pt")[0])
    return {get_ckp_id(ckp_name): ckp_name for ckp_name in checkpoints}

def process_local_policy(layout, policy_name, policy_info):
    print(f"\n\n===== start local policy [{policy_name}] =====\n\n")

    print("use reward information in log file:", policy_info["log_file"], "\n")
    with open(policy_info["log_file"], "r") as f:
        summary = json.load(f)
    step_rew_info = get_step_rew_info(local_summary=summary, policy_name=policy_name)

    ckp_model_dir = os.path.join(policy_info["model_dir"])
    ckp_path_dict = get_local_checkpoints(ckp_model_dir)

    max_actor_id = max(list(ckp_path_dict.keys()))
    max_actor_id = (max_actor_id - 1) // 5 * 5 + 5

    max_step = 0
    max_rew = 0
    for step, rew in step_rew_info:
        max_step = max(max_step, step)
        max_rew = max(max_rew, rew)
    for policy_level, desired_r_ratio in zip(["init", "mid", "final"], [0, 0.5, 1]):
        desired_r = desired_r_ratio * max_rew
        best_step, best_rew = 0, -1e9
        for step, rew in step_rew_info:
            if abs(rew - desired_r) < abs(best_rew - desired_r):
                best_rew = rew
                best_step = step
        desired_actor_id = best_step / max_step * max_actor_id
        best_actor_id = int(-1e9)
        for actor_id in ckp_path_dict:
            if abs(actor_id - desired_actor_id) < abs(best_actor_id - desired_actor_id):
                best_actor_id = actor_id
        print("policy_level", policy_level)
        print("best_actor_id", best_actor_id, "best_rew", best_rew)
        ckp_path = os.path.abspath(os.path.join(ckp_model_dir, ckp_path_dict[best_actor_id]))
        assert os.path.exists(ckp_path), f"model path {ckp_path} does not exist"
        print(f"copy from local file [{ckp_path}]")
        print()
        yield policy_level, ckp_path,

    print(f"\n\n===== end local policy [{policy_name}] =====\n\n")

def main(pop):
    print(f"start to process population [{pop}]...\n\n\n")
    pop_dir = os.path.join(POP_DATA_TOPDIR, pop)
    with open(os.path.join(pop_dir, "index_config.json"), "r") as f:
        config = json.load(f)

    layout = config["layout"]
    controller = PopulationController(pop, config)
    if "local_policies" in config:
        for policy_name, policy_info in config["local_policies"].items():
            for policy_level, local_ckp_path in process_local_policy(layout, policy_name, policy_info):
                controller.register_local_policy(policy_name, policy_info, policy_level, local_ckp_path)
    if "web_policies" in config:
        raise NotImplementedError
    controller.copy_all_checkpoints()
    controller.create_vae_yml_config()
    controller.visualize_all_policies()

if __name__ == "__main__":
    main(sys.argv[1])
