import os

from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS, OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


def formatted_data_path(args):
    data_dir = os.path.dirname(__file__)
    layout = args.human_layout_name
    os.makedirs(os.path.join(data_dir, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "dataset", "formatted_human_trajectories"), exist_ok=True)
    return os.path.join(data_dir, "dataset", "formatted_human_trajectories", f"{layout}.pickle")

def get_humandata_params(args, data_path):
    DEFAULT_DATA_PARAMS = {
        "layouts": [args.human_layout_name],
        "check_trajectories": False,
        "featurize_states": True,
        "data_path": data_path,
    }

    DEFAULT_MLP_PARAMS = {
        # Number of fully connected layers to use in our network
        "num_layers": 2,
        # Each int represents a layer of that hidden size
        "net_arch": [64, 64],
    }

    DEFAULT_TRAINING_PARAMS = {
        "epochs": args.bc_num_epochs,
        "batch_size": args.bc_batch_size,
        "learning_rate": args.lr,
    }

    DEFAULT_EVALUATION_PARAMS = {
        "ep_length": 400,
        "num_games": 1,
        "display": False,
    }

    DEFAULT_BC_PARAMS = {
        "eager": True,
        "use_lstm": False,
        "cell_size": 256,
        "data_params": DEFAULT_DATA_PARAMS,
        "mdp_params": {"layout_name": args.layout_name, "old_dynamics": args.old_dynamics},
        "env_params": DEFAULT_ENV_PARAMS,
        "mdp_fn_params": {},
        "mlp_params": DEFAULT_MLP_PARAMS,
        "training_params": DEFAULT_TRAINING_PARAMS,
        "evaluation_params": DEFAULT_EVALUATION_PARAMS,
        "action_shape": (len(Action.ALL_ACTIONS),),
    }

    def _make_env(bc_params):
        mdp_params, env_params = bc_params['mdp_params'], bc_params['env_params']

        mdp = OvercookedGridworld.from_layout_name(**mdp_params)
        mdp_fn = lambda _ignored: mdp

        env = OvercookedEnv(mdp_fn, **env_params)
        return env

    def _get_observation_shape(bc_params):
        base_env = _make_env(bc_params)
        dummy_state = base_env.mdp.get_standard_start_state()
        obs_shape = base_env.featurize_state_mdp(dummy_state)[0].shape
        return obs_shape

    DEFAULT_BC_PARAMS["observation_shape"] = _get_observation_shape(DEFAULT_BC_PARAMS)
    return DEFAULT_BC_PARAMS

def extract_human_data(args):
    reload_data = args.human_data_refresh
    import pickle
    if reload_data:
        if args.human_data_split == "2019-train":
            from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN as DATA_SPLIT
        elif args.human_data_split == "2019-test":
            from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TEST as DATA_SPLIT
        elif args.human_data_split == "2020-train":
            from human_aware_rl.static import CLEAN_2020_HUMAN_DATA_TRAIN as DATA_SPLIT
        elif args.human_data_split == "2024-train":
            from human_aware_rl.static import CLEAN_2024_HUMAN_DATA_TRAIN as DATA_SPLIT
        elif args.human_data_split == "2024-test":
            from human_aware_rl.static import CLEAN_2024_HUMAN_DATA_TEST as DATA_SPLIT
        else:
            raise NotImplementedError
        humandata_params = get_humandata_params(args, DATA_SPLIT)
        humandata_params["data_params"]["use_image_state"] = getattr(args, "use_image_state", False)
        training_data = get_human_human_trajectories(**humandata_params["data_params"], silent=True)
        with open(formatted_data_path(args), "wb") as f:
            pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(formatted_data_path(args), "rb") as f:
        training_data = pickle.load(f)
    return training_data
