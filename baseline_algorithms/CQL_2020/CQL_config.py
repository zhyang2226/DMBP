CQL_config = {
    'Conservative_Q': int(3),  # int(0) means no Conservative Q term;
    "eval_freq": int(1e4),
    "max_timestep": int(1e6),
    "checkpoint_start": int(9e5),
    "checkpoint_every": int(1e4),
    "policy": "Gaussian",  # "Gaussian", "Deterministic"
    "automatic_entropy_tuning": False, # Always False for good results
    "iter_repeat_sampling": True,

    "gamma": 0.99,
    "tau": 0.005,

    "q_lr": 3e-4,
    'policy_lr': 3e-4,
    "alpha": 0.2,
    "batch_size": 256,
    "hidden_size": 256,
    "target_update_interval": 1,
    "normalize": False,
}
