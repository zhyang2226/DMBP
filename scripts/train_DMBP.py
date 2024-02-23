# %% Part 0 Package import
# Core: Predictor is based on Multistep Diffusion with U-net Structure with Self-defined2 Function
# Core: Batch Buffer is used for sequential states
import os.path
# os.chdir(os.path.dirname(os.getcwd()))
import sys

from Utils import environments
import gym
import d4rl
import time
import datetime
import torch
import itertools
import numpy as np
from loguru import logger
import wandb
import argparse

from baseline_algorithms.RORL_2022.RORL_config import RORL_config, update_RORL_config
from baseline_algorithms.RORL_2022.RORL_net import RORL

from baseline_algorithms.BCQ_2019.BCQ_config import BCQ_config
from baseline_algorithms.BCQ_2019.BCQ_net import BCQ

from baseline_algorithms.CQL_2020.CQL_config import CQL_config
from baseline_algorithms.CQL_2020.CQL_net import CQL

from baseline_algorithms.TD3BC_2021.TD3BC_config import TD3BC_config
from baseline_algorithms.TD3BC_2021.TD3BC_net import TD3_BC

from baseline_algorithms.Diffusion_QL_2023.Diffusion_QL_config import Diffusion_QL_config, update_Diffusion_QL_config
from baseline_algorithms.Diffusion_QL_2023.Diffusion_QL_net import Diffusion_QL

from diffusion_predictor.Predictor_config import DMBP_config, update_DMBP_config
from diffusion_predictor.Predictor_net import Diffusion_Predictor
from diffusion_predictor.Evaluation import eval_policy_longcon, render_generation
from diffusion_predictor.render_img import MuJoCoRenderer
from Utils.Batch_Buffer import batch_buffer
from Utils.seed import setup_seed, seed_env


def train_DMBP(args):
    # %% Part 1 Env Parameters Initialization and Buffer Loading
    # Step 1 General Parameter Definition
    eval_Algo = args.algo
    env_name0 = args.env_name
    Dataset = args.dataset
    seed = args.seed
    saving_model = args.not_saving_model
    log_writer = args.not_saving_logwriter
    rendering = args.rendering

    eval_noise = np.linspace(0.05, 0.15, 3)
    eval_episode = 10
    Predictor_config = update_DMBP_config(env_name0, DMBP_config, task=args.task)
    beta_training_mode = Predictor_config['beta_training_mode']
    T_scheme = Predictor_config['T-scheme']
    condition_len = Predictor_config['condition_length']
    non_markov_len = Predictor_config['non_markovian_step']
    Norm = False

    device = torch.device(args.device)
    if device.type == 'cpu':
        device_idx = -1
    else:
        device_idx = device.index
    logger.info(f"Training Device is chosen to be {device}")

    # Step 2 Config Loading
    if env_name0 == "hopper":
        Env_Name = f"hopper-{Dataset}-v2"
    elif env_name0 == "walker2d":
        Env_Name = f"walker2d-{Dataset}-v2"
    elif env_name0 == "halfcheetah":
        Env_Name = f"halfcheetah-{Dataset}-v2"
    elif env_name0 == "ant":
        Env_Name = f"ant-{Dataset}-v2"
    elif env_name0 == "pen":
        Env_Name = f"pen-{Dataset}-v1"
    elif env_name0 == "hammer":
        Env_Name = f"hammer-{Dataset}-v1"
    elif env_name0 == "door":
        Env_Name = f"door-{Dataset}-v1"
    elif env_name0 == "relocate":
        Env_Name = f"relocate-{Dataset}-v1"
    elif env_name0 == 'kitchen':
        Env_Name = f'kitchen-{Dataset}-v0'
    elif env_name0 == "antmaze":
        Env_Name = f"antmaze-{Dataset}-v0"
    else:
        raise ValueError(f"Input Env '{Dataset}' is not included in D4RL")

    if eval_Algo == "BCQ":
        config = BCQ_config
    elif eval_Algo == "CQL":
        config = CQL_config
    elif eval_Algo == "TD3BC":
        config = TD3BC_config
    elif eval_Algo == "RORL":
        config = RORL_config
        config = update_RORL_config(env_name0, Dataset, config)
    elif eval_Algo == "Diffusion_QL":
        config = Diffusion_QL_config
        if env_name0 == "halfcheetah":
            config['max_q_backup'] = True
        else:
            config['max_q_backup'] = False
        config = update_Diffusion_QL_config(Env_Name, config)
    else:
        raise NotImplementedError

    predictor_config = Predictor_config
    eval_freq = int(predictor_config['eval_freq'])
    max_timestep = int(predictor_config['max_timestep'])
    start_testing = int(predictor_config['start_testing'])
    setup_seed(seed)

    if rendering:
        render = MuJoCoRenderer(Env_Name)
        logger.info(f"{Env_Name} has been set up for rendering")

    Buffer = batch_buffer(env_name0, Dataset, device, buffer_mode='normal', buffer_normalization=Norm)
    logger.info("D4RL datasets has been loaded successfully")

    eval_env = gym.make(Env_Name)
    seed_env(eval_env, seed)
    logger.info("Evaluation Environment has been seeded!")

    if Norm:
        norm_info = 'Norm'
    else:
        norm_info = 'No_Norm'

    if log_writer:
        wandb.init(project='ICLR2024_DMBP', group=env_name0, config=predictor_config,
                   tags=[Env_Name, f"T{T_scheme}", f"Con{condition_len}", f"NM{non_markov_len}"],
                   name=f"{Dataset}_{beta_training_mode}_{norm_info}_T{T_scheme}_"
                        f"Con{condition_len}_NM{non_markov_len}")

    denoiser = Diffusion_Predictor(Buffer.obs_dim, Buffer.act_dim, device, predictor_config, log_writer)

    if eval_Algo == "BCQ":
        policy = BCQ(Buffer.obs_dim, Buffer.act_dim, Buffer.max_action, device, config)
        path_head = f"baseline_algorithms/BCQ_2019/"
        policy.load_model(f"{path_head}best_models/{eval_Algo}_{env_name0}_{seed}_{Dataset}")
    elif eval_Algo == "CQL":
        policy = CQL(Buffer.obs_dim, eval_env.action_space, device, config)
        path_head = "baseline_algorithms/CQL_2020/"
        policy.load_model(f"{path_head}best_models/{eval_Algo}_{env_name0}_{seed}_{Dataset}")
    elif eval_Algo == "TD3BC":
        policy = TD3_BC(Buffer, device, config)
        path_head = "baseline_algorithms/TD3BC_2021/"
        policy.load_model(f"{path_head}best_models/{eval_Algo}_{env_name0}_{seed}_{Dataset}")
    elif eval_Algo == "RORL":
        policy = RORL(Buffer.obs_dim, eval_env.action_space, device, config)
        path_head = f"baseline_algorithms/RORL_2022/"
        policy.load_model(f"{path_head}best_models/{eval_Algo}_{env_name0}_{seed}_{Dataset}")
    elif eval_Algo == "Diffusion_QL":
        policy = Diffusion_QL(Buffer.obs_dim, Buffer.act_dim, Buffer.max_action, device, config)
        path_head = f"baseline_algorithms/Diffusion_QL_2023/"
        policy.load_model(f"{path_head}best_models/{eval_Algo}_{env_name0}_{seed}_{Dataset}")
    else:
        raise NotImplementedError(f"No such Algorithm {eval_Algo}")


    setting = f"Diffusion_Predictor_{beta_training_mode}_{norm_info}_" \
              f"{env_name0}_{seed}_{Dataset}_T{T_scheme}_Con{condition_len}_NM{non_markov_len}"


    if beta_training_mode == 'all':
        Policy_filepath = f"diffusion_predictor/best_models2/{setting}"
        Policy_checkpoint_filepath = f"diffusion_predictor/checkpoint_models2/{setting}"
        if not os.path.exists(f"diffusion_predictor/checkpoint_models2"):
            os.makedirs(f"diffusion_predictor/checkpoint_models2")
        if not os.path.exists(f"diffusion_predictor/best_models2"):
            os.makedirs(f"diffusion_predictor/best_models2")
    else:
        Policy_filepath = f"diffusion_predictor/best_models/{setting}"
        Policy_checkpoint_filepath = f"diffusion_predictor/checkpoint_models/{setting}"
        if not os.path.exists(f"diffusion_predictor/checkpoint_models"):
            os.makedirs(f"diffusion_predictor/checkpoint_models")
        if not os.path.exists(f"diffusion_predictor/best_models"):
            os.makedirs(f"diffusion_predictor/best_models")

    checkpoint_idx = 0
    checkpoint_buffer = []
    checkpoint_start = predictor_config['checkpoint_start']
    checkpoint_every = predictor_config['checkpoint_every']
    if rendering:
        render_filepath = f"diffusion_predictor/render_log/{setting}"
        if not os.path.exists(f"diffusion_predictor/render_log"):
            os.makedirs(f"diffusion_predictor/render_log")

    # %% Part 2 Train Diffusion Predictor
    print("="*80)
    print(f"Training Start: {setting}")
    if saving_model:
        print(f"Model will be saved to path {Policy_filepath}")
    else:
        print(f"Model will not be saved")
    print("="*80)

    # Core training Start
    total_train = 0
    best_idx = 0
    best_reward = np.zeros_like(eval_noise)
    max_nominal_reward = -np.inf

    start_time = time.time()
    for i_episode in itertools.count(1):
        if total_train > max_timestep:
            break
        logger.info(f"Training is in process: episode({total_train})")
        mid_time = time.time()
        if total_train != 0:
            logger.info(f"Estimated remaining time of "
                        f"{(mid_time - start_time) / 60 * (max_timestep - total_train) / total_train:.4f} min")

        # Evaluation (only the training reach "start_testing", the file will be built up and the evaluation start)
        if total_train >= start_testing:
            nominal_reward = 0
            record_reward_eval_noise = np.zeros_like(best_reward)
            idx = 0
            for i in eval_noise:
                avg_reward, std_reward = eval_policy_longcon(policy, denoiser, False, eval_env, eval_episode, i)
                nominal_reward += avg_reward * i
                record_reward_eval_noise[idx] = avg_reward
                idx += 1
                if log_writer:
                    wandb.log({f"Evaluation/Noise {i:.2} Reward Mean": avg_reward,
                               f"Evaluation/Noise {i:.2} Reward Std": std_reward,}, step=denoiser.step)
            if nominal_reward > max_nominal_reward:
                max_nominal_reward = nominal_reward
                best_idx = denoiser.step
                best_reward = record_reward_eval_noise
                if saving_model:
                    denoiser.save_model(Policy_filepath)
            if log_writer:
                wandb.log({f"Evaluation/Nominal_reward": nominal_reward,
                           f"Best_saving/Best_idx": best_idx}, step=denoiser.step)
                idx = 0
                for i in eval_noise:
                    wandb.log({f"Best_saving/Noise {i:.2} Reward Mean": best_reward[idx]}, step=denoiser.step)
                    idx += 1
        if total_train >= checkpoint_start and total_train % checkpoint_every == 0 and saving_model:
            denoiser.save_checkpoint(f"{Policy_checkpoint_filepath}_checkpoint{checkpoint_idx}")
            if log_writer:
                wandb.log({f"Checkpoint/reward_mean0": record_reward_eval_noise[0],
                           f"Checkpoint/reward_mean1": record_reward_eval_noise[1],
                           f"Checkpoint/reward_mean2": record_reward_eval_noise[2]}, step=denoiser.step)
            checkpoint_idx += 1

        # Train one epoch (eval_freq steps)
        denoiser.train(Buffer, eval_freq, predictor_config['batch_size'], log_writer)
        total_train += int(eval_freq)

    end_time = time.time()

    print("="*80)
    print(f"Training Finished: {setting}")
    # print(f"Trained Predictor Eval Distance is {min_distance:.4}, with std of {corr_std:.4}")
    print(f"Total Training time: {(end_time - start_time) / 60:.4f} min")
    print("="*80)

    if log_writer:
        wandb.join()

    # %% Part Last Generation
    if rendering:
        idx = 0
        for i in eval_noise:
            avg = best_reward[idx]
            image = render_generation(render, render_filepath, policy, denoiser, False, eval_env, i, avg)
            if log_writer:
                wandb.log({f'Trajectory/Noise Scale {i:.2}': wandb.Image(image)})
            idx += 1

    if beta_training_mode == 'all':
        Policy_filepath = f"diffusion_predictor/best_models2/{setting}_final"
    else:
        Policy_filepath = f"diffusion_predictor/best_models/{setting}_final"
    denoiser.save_model(Policy_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='denoise', type=str,
                        help="Choose from ('denoise' or 'demask')")
    parser.add_argument('--algo', default='Diffusion_QL', type=str,
                        help="Choose from reproduced baseline algorithms ('BCQ', 'CQL', 'TD3BC', 'Diffusion_QL', 'RORL')")
    parser.add_argument('--env_name', default='hopper', type=str,
                        help="Choose from mujoco domain ('halfcheetah', 'hopper', 'walker2d'), "
                             "or adroit domain ('pen', 'hammer', 'relocate', 'door'), "
                             "or franka kitchen domain ('kitchen').")
    parser.add_argument('--dataset', default='medium-replay', type=str,
                        help="Use ('expert', 'medium-expert', 'medium', 'medium-replay', 'full-replay') for mujoco domain, "
                             "or ('expert', 'cloned', 'human') for adroit domain, "
                             "or ('complete', 'mixed', 'partial') for franka kitchen domain.")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--not_saving_model', action='store_false',
                        help="'True' for saving the trained models")
    parser.add_argument('--not_saving_logwriter', action='store_false',
                        help="'True' for saving the training process in wandb")
    parser.add_argument('--rendering', action='store_true',
                        help="'True' for saving the rendering results")
    args = parser.parse_args()
    train_DMBP(args)