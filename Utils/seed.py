import torch
import numpy as np
import random
from loguru import logger
import contextlib
import os
import gym

def setup_seed(seed=1024): # After doing this, the Training results will always be the same for the same seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"Seed {seed} has been set for all modules!")

def seed_env(env, seed=random.randint(0,1024)):
    env.seed(seed)
    env.action_space.seed()


@contextlib.contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stderr(fnull) as err, contextlib.redirect_stdout(fnull) as out:
            yield (err, out)

def load_environment(name):
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env
