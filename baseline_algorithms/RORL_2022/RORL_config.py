import sys

RORL_config={
    "eval_freq": int(2e4),
    # Long Training
    "max_timestep": int(3e6),
    "checkpoint_start": int(28e5),
    "checkpoint_every": int(2e4),

    "print_more_info": False,
    "SAC10": False,

    "gamma": 0.99,
    "soft_tau": 0.005,
    "q_lr": 3e-4,
    'policy_lr': 3e-4,
    "alpha": 1.0,
    "auto_tune_entropy": True,
    "max_q_backup": False,
    "deterministic_backup": False,
    "eta": -1.,
    "batch_size": 256,
    "hidden_size": 256,
    "target_update_interval": 1,
    "normalize": False,
}

def update_RORL_config(env_name0, dataset, config):
    if env_name0 == "halfcheetah":
        updated_config = {
            "beta_Q": 0.0001,
            "beta_P": 0.1,
            "beta_ood": 0.0,
            "eps_Q": 0.001,
            "eps_P": 0.001,
            "eps_ood": 0.0,
            "tau": 0.2,
            "n_sample": 10,
            "lambda_max": 0.0,
            "lambda_min": 0.0,
            "lambda_decay": 0.0
        }
        if dataset == "expert":
            updated_config["eps_Q"] = 0.005
            updated_config["eps_P"] = 0.005
        elif dataset == "random":
            updated_config["n_sample"] = 20

    elif env_name0 == "hopper":
        updated_config = {
            "beta_Q": 0.0001,
            "beta_P": 0.1,
            "beta_ood": 0.5,
            "eps_Q": 0.005,
            "eps_P": 0.005,
            "eps_ood": 0.01,
            "tau": 0.2,
            "n_sample": 20,
            "lambda_max": 0.0,
            "lambda_min": 0.0,
            "lambda_decay": 1e-6,
        }
        if dataset == "expert":
            updated_config["lambda_max"] = 4.0
            updated_config["lambda_min"] = 1.0
        elif dataset == "medium-expert":
            updated_config["lambda_max"] = 3.0
            updated_config["lambda_min"] = 1.0
        elif dataset == "medium-replay":
            updated_config["lambda_max"] = 0.1
            updated_config["lambda_min"] = 0.0
        elif dataset == "medium":
            updated_config["lambda_max"] = 2.0
            updated_config["lambda_min"] = 0.1
        elif dataset == "full-replay":
            updated_config["lambda_max"] = 0.1
            updated_config["lambda_min"] = 0.0
        elif dataset == "random":
            updated_config["lambda_max"] = 1.0
            updated_config["lambda_min"] = 0.5

    elif env_name0 == "walker2d":
        updated_config = {
            "beta_Q": 0.0001,
            "beta_P": 1.0,
            "beta_ood": 0.1,
            "eps_Q": 0.01,
            "eps_P": 0.01,
            "eps_ood": 0.01,
            "tau": 0.2,
            "n_sample": 20,
            "lambda_max": 0.1,
            "lambda_min": 0.1,
            "lambda_decay": 0.0,
        }
        if dataset == "expert":
            updated_config["beta_ood"] = 0.5
            updated_config['eps_Q'] = 0.005
            updated_config['eps_P'] = 0.005
            updated_config["lambda_max"] = 1.0
            updated_config["lambda_min"] = 0.7
            updated_config["lambda_decay"] = 1e-6

        elif dataset == "random":
            updated_config["beta_ood"] = 0.5
            updated_config['eps_Q'] = 0.005
            updated_config['eps_P'] = 0.005
            updated_config["lambda_max"] = 5.0
            updated_config["lambda_min"] = 0.5
            updated_config["lambda_decay"] = 1e-5
    elif env_name0 == "pen" and dataset == "expert":
        updated_config = {
            "beta_Q": 0.0001,
            "beta_P": 1.0,
            "beta_ood": 0.5,
            "eps_Q": 0.005,
            "eps_P": 0.005,
            "eps_ood": 0.01,
            "tau": 0.2,
            "n_sample": 20,
            "lambda_max": 2.0,
            "lambda_min": 2.0,
            "lambda_decay": 0.0,
        }
    elif env_name0 == "hammer" and dataset == "expert":
        updated_config = {
            "beta_Q": 0.0001,
            "beta_P": 1.0,
            "beta_ood": 0.5,
            "eps_Q": 0.005,
            "eps_P": 0.005,
            "eps_ood": 0.01,
            "tau": 0.2,
            "n_sample": 20,
            "lambda_max": 1.0,
            "lambda_min": 0.5,
            "lambda_decay": 1e-6,
        }
    elif env_name0 == "door" and dataset == "expert":
        updated_config = {
            "beta_Q": 0.0001,
            "beta_P": 1.0,
            "beta_ood": 0.5,
            "eps_Q": 0.005,
            "eps_P": 0.005,
            "eps_ood": 0.01,
            "tau": 0.2,
            "n_sample": 20,
            "lambda_max": 1.5,
            "lambda_min": 1.5,
            "lambda_decay": 0.,
        }
    elif env_name0 == "relocate" and dataset == "expert":
        updated_config = {
            "beta_Q": 0.0001,
            "beta_P": 1.0,
            "beta_ood": 0.5,
            "eps_Q": 0.001,
            "eps_P": 0.001,
            "eps_ood": 0.01,
            "tau": 0.2,
            "n_sample": 20,
            "lambda_max": 3.0,
            "lambda_min": 2.0,
            "lambda_decay": 2e-6,
        }
    elif env_name0 == "kitchen":
        updated_config = {
            "beta_Q": 0.0001,
            "beta_P": 1.0,
            "beta_ood": 0.5,
            "eps_Q": 0.005,
            "eps_P": 0.005,
            "eps_ood": 0.01,
            "tau": 0.2,
            "n_sample": 20,
            "lambda_max": 1.0,
            "lambda_min": 0.5,
            "lambda_decay": 1e-6,
        }
    elif env_name0 == "antmaze":
        updated_config = {
            "beta_Q": 0.0001,
            "beta_P": 1.0,
            "beta_ood": 0.5,
            "eps_Q": 0.01,
            "eps_P": 0.03,
            "eps_ood": 0.01,
            "tau": 0.2,
            "n_sample": 20,
            "lambda_max": 1.0,
            "lambda_min": 0.5,
            "lambda_decay": 1e-6,
        }
    else:
        raise NotImplementedError("Check the environment config")

    config.update(updated_config)
    return config
