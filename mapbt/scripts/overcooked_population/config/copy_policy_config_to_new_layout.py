import argparse
import os
import pickle
from copy import deepcopy

import numpy as np
from gymnasium.spaces import Box


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_layout", default="asymmetric_advantages", type=str)
    parser.add_argument("--target_layout", default="target_layout", type=str)
    parser.add_argument("--old_dynamics", default=False, action="store_true")
    parser.add_argument("--obs_shape_width", default=None, type=int)
    parser.add_argument("--obs_shape_height", default=None, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    source_layout_dir = os.path.join(os.path.dirname(__file__), args.source_layout)
    target_layout_dir = os.path.join(os.path.dirname(__file__), args.target_layout)
    os.makedirs(target_layout_dir, exist_ok=True)
    for policy_name in ["bc", "mlp", "rnn"]:
        with open(f"{source_layout_dir}/{policy_name}_policy_config.pkl", "rb") as f:
            data = pickle.load(f)
        new_policy_config = deepcopy(data[0])
        new_policy_config.layout_name = args.target_layout
        new_policy_config.old_dynamics = args.old_dynamics
        if policy_name == "bc":
            new_obs_shape = deepcopy(data[1].shape)
        else:
            new_obs_shape = (args.obs_shape_width or data[1].shape[0], args.obs_shape_height or data[1].shape[1], data[1].shape[2])
        new_obs_space = Box(
            np.ones(new_obs_shape, dtype=np.float32) * data[1].low.reshape(-1)[0],
            np.ones(new_obs_shape, dtype=np.float32) * data[1].high.reshape(-1)[0],
            dtype=np.float32
        )
        new_share_obs_shape = (args.obs_shape_width or data[2].shape[0], args.obs_shape_height or data[2].shape[1], data[2].shape[2])
        new_share_obs_space = Box(
            np.ones(new_share_obs_shape, dtype=np.float32) * data[2].low.reshape(-1)[0],
            np.ones(new_share_obs_shape, dtype=np.float32) * data[2].high.reshape(-1)[0],
            dtype=np.float32
        )
        new_data = (new_policy_config, new_obs_space, new_share_obs_space, data[3])
        with open(f"{target_layout_dir}/{policy_name}_policy_config.pkl", "wb") as f:
            pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
