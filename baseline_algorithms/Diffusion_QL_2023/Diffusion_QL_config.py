Diffusion_QL_config = {
    "algo": "ql", # "bc", "ql"
    "eval_freq": int(1e4),
    "max_timestep": int(1e6),
    "checkpoint_start": int(9e5),
    "checkpoint_every": int(1e4),

    "gamma": 0.99,
    "tau": 0.005,
    "eta": 1.0,
    "lr_decay": True,
    "max_q_backup": False,
    "step_start_ema": 1000,
    "ema_decay": 0.995,
    "update_ema_every": 5,
    "T": 10,
    "beta_schedule": 'vp',
    'ms': 'offline',
    'gn': 10.0,

    "lr": 3e-4,
    "alpha": 0.2,
    "batch_size": 256,
    "hidden_size": 256,
    "reward_tune": "no",

    "print_more_info": False,
    "normalize": False
}


def update_Diffusion_QL_config(env_name, config):
    if env_name == 'pen-human-v1' or env_name == 'pen-cloned-v1' or env_name == 'pen-expert-v1':
        update_config = {"eta": 0.1, "lr": 3e-5}
    elif env_name == 'hammer-human-v1' or env_name == 'hammer-cloned-v1' or env_name == 'hammer-expert-v1':
        update_config = {"eta": 0.1, "lr": 3e-5}
    elif env_name == 'door-human-v1' or env_name == 'door-cloned-v1' or env_name == 'door-expert-v1':
        update_config = {"eta": 0.1, "lr": 3e-5}
    elif env_name == 'relocate-human-v1' or env_name == 'relocate-cloned-v1' or env_name == 'relocate-expert-v1':
        update_config = {"eta": 0.1, "lr": 3e-5}
    elif env_name == 'kitchen-mixed-v0' or env_name == 'kitchen-complete-v0' or env_name == 'kitchen-partial-v0':
        update_config = {"eta": 0.005}
    elif env_name == "antmaze-umaze-v0":
        update_config = {"eta": 0.5, "lr": 3e-4, "max_q_backup": False}
    elif env_name == "antmaze-umaze-diverse-v0":
        update_config = {"eta": 2.0, "lr": 3e-4, "max_q_backup": True}
    elif env_name == "antmaze-medium-play-v0":
        update_config = {"eta": 2.0, "lr": 1e-3, "max_q_backup": True}
    elif env_name == "antmaze-medium-diverse-v0":
        update_config = {"eta": 3.0, "lr": 3e-4, "max_q_backup": True}
    elif env_name == "antmaze-large-play-v0":
        update_config = {"eta": 4.5, "lr": 3e-4, "max_q_backup": True}
    elif env_name == "antmaze-large-diverse-v0":
        update_config = {"eta": 4.5, "lr": 3e-4, "max_q_backup": True}
    else:
        update_config = {}

    config.update(update_config)

    return config
