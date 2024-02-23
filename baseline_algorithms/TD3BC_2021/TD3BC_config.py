TD3BC_config = {
    "policy": "TD3BC",
    "eval_freq": int(1e4),
    "max_timestep": int(1e6),
    "checkpoint_start": int(9e5),
    "checkpoint_every": int(1e4),

    "expl_noise": 0.1,
    "batch_size": 256,
    "discount": 0.99,
    "tau": 0.005,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "policy_freq": 2,
    "alpha": 2.5,
    "normalize": True,
}
