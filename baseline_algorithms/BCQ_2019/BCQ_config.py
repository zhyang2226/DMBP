BCQ_config = {
    "eval_freq": int(1e4),
    "max_timestep": int(1e6),
    "checkpoint_start": int(9e5),
    "checkpoint_every": int(1e4),

    'batch_size': 256,
    'discount': 0.99,
    'tau': 0.005,
    'lmbda': 0.75,
    'phi': 0.05,
    "normalize": False,
}
