DMBP_config = {
    "algo": "ql", # "bc", "ql"
    "eval_freq" : 5000,
    "max_timestep": int(3e5),
    "start_testing": int(0),
    "checkpoint_start": int(2e5),
    "checkpoint_every": int(1e4),

    # "eval_freq": 200,
    # "max_timestep": int(600),
    # "start_testing": int(400),
    # "checkpoint_start": int(100),
    # "checkpoint_every": int(100),

    "gamma": 0.99,
    "tau": 0.005,
    "eta": 1.0,
    "lr_decay": True,
    "max_q_backup": True,
    "step_start_ema": 1000,
    "ema_decay": 0.995,
    "update_ema_every": 5,
    "T": 100,

    "beta_schedule": 'self-defined2',
    "beta_training_mode": 'all',   # 'all' or 'partial'
    'loss_training_mode': 'no_act2',    # 'normal' or 'noise' or 'no_act' or 'no_act2'
    "predict_epsilon": True,
    "data_usage": 1.0,
    'ms': 'offline',
    'gn': 10.0,

    # Long Term Buffer Parameter Definition
    "condition_length": 4,
    "T-scheme": "same",  # "random" or "same"

    "non_markovian_step": 6,

    # Attention Hyperparameters
    "attn_hidden_layer": 2,
    "attn_hidden_dim": 128,
    "attn_embed_dim": 64,

    "lr": 3e-4,
    "alpha": 0.2,
    "batch_size": 64,
    "hidden_size": 256,
    "embed_dim": 64,
    "reward_tune": "no",
}

def update_DMBP_config(env_name, config, task="denoise"):
    if task == "denoise":
        config["beta_training_mode"] = "partial"
    elif task == "demask":
        config["beta_training_mode"] = "all"
    else:
        raise NotImplementedError

    if "hopper" in env_name.lower():
        updated_config = {
            "condition_length": 4,
            "non_markovian_step": 6,
        }
    elif "halfcheetah" in env_name.lower():
        updated_config = {
            "condition_length": 4,
            "non_markovian_step": 2,
        }
    elif "walker" in env_name.lower():
        updated_config = {
            "condition_length": 4,
            "non_markovian_step": 4,
        }
    elif "pen" in env_name.lower():
        updated_config = {
            "condition_length": 4,
            "non_markovian_step": 6,
        }
    elif "door" in env_name.lower():
        updated_config = {
            "condition_length": 4,
            "non_markovian_step": 6,
        }
    elif "hammer" in env_name.lower():
        updated_config = {
            "condition_length": 4,
            "non_markovian_step": 6,
        }
    elif "relocate" in env_name.lower():
        updated_config = {
            "condition_length": 4,
            "non_markovian_step": 4,
            "embed_dim": 128,
        }
    elif "kitchen" in env_name.lower():
        updated_config = {
            "condition_length": 4,
            "non_markovian_step": 4,
            "embed_dim": 256,
        }
    else:
        raise NotImplementedError

    config.update(updated_config)
    return config
