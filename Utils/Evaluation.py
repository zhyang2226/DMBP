import numpy as np
import random
import torch
import wandb
import time
from loguru import logger

np.set_printoptions(precision=4, linewidth=300, suppress=True)

# %% Part 1 Markovian Based Policy Evaluation with no DMBP
def eval_policy(policy, eval_env, eval_episodes=10):
    reward_buffer = []

    for _ in range(eval_episodes):
        reward_ep = 0.
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            reward_ep += reward
        reward_buffer.append(reward_ep)

    avg_reward = np.average(reward_buffer)
    avg_reward = eval_env.get_normalized_score(avg_reward) * 100
    std_reward = np.std(reward_buffer)
    std_reward = eval_env.get_normalized_score(std_reward) * 100
    max_reward = np.max(reward_buffer)
    max_reward = eval_env.get_normalized_score(max_reward) * 100
    min_reward = np.min(reward_buffer)
    min_reward = eval_env.get_normalized_score(min_reward) * 100

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Evaluation over {eval_episodes} episodes: mean {avg_reward:.3f} and std {std_reward:.3f}")
    print(f"Max Reward is: {max_reward:.3f}, and Min Reward is {min_reward:.3f}")
    return avg_reward, std_reward, max_reward, min_reward

def eval_policy_Gaussian(policy, eval_env, eval_episodes=10, noise_scale=0, Algo=None):
    reward_buffer = []
    step = 0
    for _ in range(eval_episodes):
        # time1 = time.time()
        reward_ep = 0.
        state, done = eval_env.reset(), False
        state_dim = state.shape[0]
        while not done:
            step += 1
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            state = state + noise_scale * np.random.randn(state_dim)
            reward_ep += reward

        reward_buffer.append(reward_ep)
        # time2 = time.time()
        # print(f"{step} step evaluation time is {time2 - time1:.3f}")

    avg_reward = np.average(reward_buffer)
    avg_reward = eval_env.get_normalized_score(avg_reward) * 100
    std_reward = np.std(reward_buffer)
    std_reward = eval_env.get_normalized_score(std_reward) * 100
    max_reward = np.max(reward_buffer)
    max_reward = eval_env.get_normalized_score(max_reward) * 100
    min_reward = np.min(reward_buffer)
    min_reward = eval_env.get_normalized_score(min_reward) * 100

    nor_reward = eval_env.get_normalized_score(np.stack(reward_buffer)) * 100

    print(f"Eval over {eval_episodes} episodes (noise std {noise_scale:.3f}): mean {avg_reward:.3f} std {std_reward:.3f} "
          f"Max {max_reward:.3f} Min {min_reward:.3f}")
    return avg_reward, std_reward, max_reward, min_reward, nor_reward

def eval_policy_Uniform(policy, eval_env, eval_episodes=10, noise_scale=0, Algo=None):
    reward_buffer = []
    step = 0
    for _ in range(eval_episodes):
        # time1 = time.time()
        reward_ep = 0.
        state, done = eval_env.reset(), False
        state_dim = state.shape[0]
        while not done:
            step += 1
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            state = state + (2 * noise_scale * np.random.rand(state_dim) - noise_scale)
            reward_ep += reward

        reward_buffer.append(reward_ep)
        # time2 = time.time()
        # print(f"{step} step evaluation time is {time2 - time1:.3f}")

    avg_reward = np.average(reward_buffer)
    avg_reward = eval_env.get_normalized_score(avg_reward) * 100
    std_reward = np.std(reward_buffer)
    std_reward = eval_env.get_normalized_score(std_reward) * 100
    max_reward = np.max(reward_buffer)
    max_reward = eval_env.get_normalized_score(max_reward) * 100
    min_reward = np.min(reward_buffer)
    min_reward = eval_env.get_normalized_score(min_reward) * 100

    nor_reward = eval_env.get_normalized_score(np.stack(reward_buffer)) * 100

    print(
        f"Eval over {eval_episodes} episodes (noise std {noise_scale:.3f}): mean {avg_reward:.3f} std {std_reward:.3f} "
        f"Max {max_reward:.3f} Min {min_reward:.3f}")
    return avg_reward, std_reward, max_reward, min_reward, nor_reward

def eval_policy_act_diff(policy, eval_env, eval_episodes=10, noise_scale=0, Algo=None):
    reward_buffer = []
    device = policy.device
    noise_rpt = 20

    for _ in range(eval_episodes):
        reward_ep = 0.
        state, done = eval_env.reset(), False
        state_dim = state.shape[0]
        action = policy.select_action(np.array(state))
        while not done:
            state, reward, done, _ = eval_env.step(action)

            state0 = to_cuda(state, device).reshape(1, -1)
            rpt_state = torch.repeat_interleave(state0, repeats=noise_rpt, dim=0)
            rpt_noised_state = rpt_state + (2 * noise_scale * torch.rand_like(rpt_state) - noise_scale)
            rpt_noised_action = policy.select_action(rpt_noised_state)
            original_action = torch.mean(policy.select_action(rpt_state), dim=0, keepdim=True)
            rpt_action = torch.repeat_interleave(original_action, repeats=noise_rpt, dim=0)
            difference = torch.sum(torch.abs(rpt_action - rpt_noised_action), dim=1)
            idx = torch.argmax(difference)

            action = rpt_noised_action[idx].detach().cpu().numpy()

            reward_ep += reward
        reward_buffer.append(reward_ep)

    avg_reward = np.average(reward_buffer)
    avg_reward = eval_env.get_normalized_score(avg_reward) * 100
    std_reward = np.std(reward_buffer)
    std_reward = eval_env.get_normalized_score(std_reward) * 100
    max_reward = np.max(reward_buffer)
    max_reward = eval_env.get_normalized_score(max_reward) * 100
    min_reward = np.min(reward_buffer)
    min_reward = eval_env.get_normalized_score(min_reward) * 100

    nor_reward = eval_env.get_normalized_score(np.stack(reward_buffer)) * 100

    print(
        f"Eval over {eval_episodes} episodes (noise std {noise_scale:.3f}): mean {avg_reward:.3f} std {std_reward:.3f} "
        f"Max {max_reward:.3f} Min {min_reward:.3f}")
    return avg_reward, std_reward, max_reward, min_reward, nor_reward

def eval_policy_minQ(policy, eval_env, eval_episodes=10, noise_scale=0, Algo=None):
    reward_buffer = []
    device = policy.device
    noise_rpt = 20

    for _ in range(eval_episodes):
        reward_ep = 0.
        state, done = eval_env.reset(), False
        state_dim = state.shape[0]
        action = policy.select_action(np.array(state))
        while not done:
            state, reward, done, _ = eval_env.step(action)

            state0 = to_cuda(state, device).reshape(1, -1)
            rpt_state = torch.repeat_interleave(state0, repeats=noise_rpt, dim=0)
            rpt_noised_state = rpt_state + (2 * noise_scale * torch.rand_like(rpt_state) - noise_scale)
            rpt_noised_action = policy.select_action(rpt_noised_state)
            if Algo == "RORL":
                Q_all = policy.critic_target(rpt_noised_state, rpt_noised_action)
                Q, _ = policy.get_min_Q(Q_all)
                idx = torch.argmin(Q)
                action = rpt_noised_action[idx].detach().cpu().numpy()
            else:
                Q1, Q2 = policy.critic_target(rpt_noised_state, rpt_noised_action)
                Q = torch.min(Q1, Q2)
                idx = torch.argmin(Q)
                action = rpt_noised_action[idx].detach().cpu().numpy()

            reward_ep += reward
        reward_buffer.append(reward_ep)

    avg_reward = np.average(reward_buffer)
    avg_reward = eval_env.get_normalized_score(avg_reward) * 100
    std_reward = np.std(reward_buffer)
    std_reward = eval_env.get_normalized_score(std_reward) * 100
    max_reward = np.max(reward_buffer)
    max_reward = eval_env.get_normalized_score(max_reward) * 100
    min_reward = np.min(reward_buffer)
    min_reward = eval_env.get_normalized_score(min_reward) * 100

    nor_reward = eval_env.get_normalized_score(np.stack(reward_buffer)) * 100

    print(
        f"Eval over {eval_episodes} episodes (noise std {noise_scale:.3f}): mean {avg_reward:.3f} std {std_reward:.3f} "
        f"Max {max_reward:.3f} Min {min_reward:.3f}")
    return avg_reward, std_reward, max_reward, min_reward, nor_reward

# %% Part 2 Markovian Based Policy Evaluation with DMBP
def eval_DMBP_policy_Gaussian(policy, predictor, real_state, eval_env, eval_episodes=10, noise_scale=0, Algo=None, method='mean'):
    reward_buffer = []
    device = policy.device
    warm_start_up = 4
    timestep = int((noise_scale + 0.02) / 0.03)

    for k in range(eval_episodes):
        reward_ep = 0.
        # Note that state0 is for "condition usage" and state1 is for "decision making"
        state1, done = eval_env.reset(), False
        state0 = state1
        state_dim = state1.shape[0]
        long_state_con = torch.zeros([1, predictor.condition_step, state_dim]).to(device)
        step = 0
        while not done:
            step += 1
            action = policy.select_action(np.array(state1))
            next_state_real, reward, done, _ = eval_env.step(action)
            next_state_noise = next_state_real + noise_scale * np.random.randn(state_dim)

            if step >= warm_start_up:
                next_state_noise = to_cuda(next_state_noise, device).reshape(1, -1)
            else:
                next_state_noise = to_cuda(next_state_real, device).reshape(1, -1)

            action = to_cuda(action, device).reshape(1, -1)
            state = to_cuda(state0, device).reshape(1, 1, -1)
            long_state_con = torch.cat([long_state_con[:, 1:], state], dim=1)
            next_state_predict = predictor.denoise_state(next_state_noise, action, long_state_con, timestep, method=method)

            if step <= warm_start_up:
                state1 = next_state_real
                state0 = next_state_real
            else:
                state1 = next_state_predict.reshape(-1)
                state0 = next_state_predict.reshape(-1)

            if real_state:
                state0 = next_state_real

            reward_ep += reward
        reward_buffer.append(reward_ep)

    avg_reward = np.average(reward_buffer)
    avg_reward = eval_env.get_normalized_score(avg_reward) * 100
    std_reward = np.std(reward_buffer)
    std_reward = eval_env.get_normalized_score(std_reward) * 100
    max_reward = np.max(reward_buffer)
    max_reward = eval_env.get_normalized_score(max_reward) * 100
    min_reward = np.min(reward_buffer)
    min_reward = eval_env.get_normalized_score(min_reward) * 100

    nor_reward = eval_env.get_normalized_score(np.stack(reward_buffer)) * 100

    print(f"Eval over {eval_episodes} episodes (noise std {noise_scale:.3f}, timestep {timestep}): "
          f"mean {avg_reward:.3f}; std {std_reward:.3f}; max {max_reward:.3f}; min {min_reward:.3f}")
    return avg_reward, std_reward, max_reward, min_reward, nor_reward

def eval_DMBP_policy_Uniform(policy, predictor, real_state, eval_env, eval_episodes=10, noise_scale=0, Algo=None, method='mean'):
    reward_buffer = []
    device = policy.device
    warm_start_up = 4
    timestep = int((noise_scale + 0.04) / 0.05)

    for k in range(eval_episodes):
        reward_ep = 0.
        # Note that state0 is for "condition usage" and state1 is for "decision making"
        state1, done = eval_env.reset(), False
        state0 = state1
        state_dim = state1.shape[0]
        long_state_con = torch.zeros([1, predictor.condition_step, state_dim]).to(device)
        step = 0
        while not done:
            step += 1
            action = policy.select_action(np.array(state1))
            next_state_real, reward, done, _ = eval_env.step(action)
            next_state_noise = next_state_real + (2 * noise_scale * np.random.rand(state_dim) - noise_scale)

            if step >= warm_start_up:
                next_state_noise = to_cuda(next_state_noise, device).reshape(1, -1)
            else:
                next_state_noise = to_cuda(next_state_real, device).reshape(1, -1)

            action = to_cuda(action, device).reshape(1, -1)
            state = to_cuda(state0, device).reshape(1, 1, -1)
            long_state_con = torch.cat([long_state_con[:, 1:], state], dim=1)
            next_state_predict = predictor.denoise_state(next_state_noise, action, long_state_con, timestep,
                                                         method=method)

            if step <= warm_start_up:
                state1 = next_state_real
                state0 = next_state_real
            else:
                state1 = next_state_predict.reshape(-1)
                state0 = next_state_predict.reshape(-1)

            if real_state:
                state0 = next_state_real

            reward_ep += reward
        reward_buffer.append(reward_ep)

    avg_reward = np.average(reward_buffer)
    avg_reward = eval_env.get_normalized_score(avg_reward) * 100
    std_reward = np.std(reward_buffer)
    std_reward = eval_env.get_normalized_score(std_reward) * 100
    max_reward = np.max(reward_buffer)
    max_reward = eval_env.get_normalized_score(max_reward) * 100
    min_reward = np.min(reward_buffer)
    min_reward = eval_env.get_normalized_score(min_reward) * 100

    nor_reward = eval_env.get_normalized_score(np.stack(reward_buffer)) * 100

    print(f"Eval over {eval_episodes} episodes (noise std {noise_scale:.3f}, timestep {timestep}): "
          f"mean {avg_reward:.3f}; std {std_reward:.3f}; max {max_reward:.3f}; min {min_reward:.3f}")
    return avg_reward, std_reward, max_reward, min_reward, nor_reward

def eval_DMBP_policy_act_diff(policy, predictor, real_state, eval_env, eval_episodes=10, noise_scale=0, Algo=None):
    reward_buffer = []
    device = policy.device
    warm_start_up = 4
    noise_rpt = 50
    timestep = int((noise_scale + 0.03) / 0.04 + 0.0001)

    for k in range(eval_episodes):
        # Note: state0 is for condition and state1 is for decision
        reward_ep = 0.
        state1, done = eval_env.reset(), False
        state0 = state1
        state_dim = state1.shape[0]
        long_state_con = torch.zeros([1, predictor.condition_step, state_dim]).to(device)
        step = 0
        action = policy.select_action(np.array(state1))
        while not done:
            step += 1
            next_state_real, reward, done, _ = eval_env.step(action)

            next_state_real_tensor = to_cuda(next_state_real, device).reshape(1, -1)
            rpt_next_state = torch.repeat_interleave(next_state_real_tensor, repeats=noise_rpt, dim=0)
            # rpt_noised_next_state = rpt_next_state + (2 * noise_scale * torch.rand_like(rpt_next_state) - noise_scale)
            if step >= warm_start_up:
                rpt_noised_next_state = rpt_next_state + (2 * noise_scale * torch.rand_like(rpt_next_state) - noise_scale)
            else:
                rpt_noised_next_state = rpt_next_state
            rpt_con_action = torch.repeat_interleave(to_cuda(action, device).reshape(1, -1), repeats=noise_rpt, dim=0)
            long_state_con = torch.cat([long_state_con[:, 1:], to_cuda(state0, device).reshape(1, 1, -1)], dim=1)
            rpt_long_state_con = torch.repeat_interleave(long_state_con, repeats=noise_rpt, dim=0)
            rpt_next_state_predict = predictor.denoise_state(rpt_noised_next_state, rpt_con_action, rpt_long_state_con, timestep)

            # Select the max difference action:
            rpt_noised_action = policy.select_action(rpt_next_state_predict)
            original_action = torch.mean(policy.select_action(rpt_next_state), dim=0)
            rpt_action = torch.repeat_interleave(original_action.reshape(1, -1), repeats=noise_rpt, dim=0)
            difference = torch.sum(torch.abs(rpt_action - rpt_noised_action), 1)
            idx = torch.argmax(difference)
            action = rpt_noised_action[idx].cpu().detach().numpy()

            if step <= warm_start_up:
                state0 = next_state_real
            else:
                state0 = rpt_next_state_predict.mean(dim=0).cpu().detach().numpy()

            if real_state:
                state0 = next_state_real
            reward_ep += reward
        reward_buffer.append(reward_ep)

    avg_reward = np.average(reward_buffer)
    avg_reward = eval_env.get_normalized_score(avg_reward) * 100
    std_reward = np.std(reward_buffer)
    std_reward = eval_env.get_normalized_score(std_reward) * 100
    max_reward = np.max(reward_buffer)
    max_reward = eval_env.get_normalized_score(max_reward) * 100
    min_reward = np.min(reward_buffer)
    min_reward = eval_env.get_normalized_score(min_reward) * 100

    nor_reward = eval_env.get_normalized_score(np.stack(reward_buffer)) * 100

    print(f"Eval over {eval_episodes} episodes (noise std {noise_scale:.3f}, timestep {timestep}): "
          f"mean {avg_reward:.3f}; std {std_reward:.3f}; max {max_reward:.3f}; min {min_reward:.3f}")
    return avg_reward, std_reward, max_reward, min_reward, nor_reward

def eval_DMBP_policy_minQ(policy, predictor, real_state, eval_env, eval_episodes=10, noise_scale=0, Algo=None):
    reward_buffer = []
    device = policy.device
    warm_start_up = 4
    timestep = int((noise_scale + 0.03) / 0.04 + 0.0001)
    noise_rpt = 20

    for k in range(eval_episodes):
        # Note: state0 is for condition and state1 is for decision
        reward_ep = 0.
        state1, done = eval_env.reset(), False
        state0 = state1
        state_dim = state1.shape[0]
        action = policy.select_action(np.array(state1))
        long_state_con = torch.zeros([1, predictor.condition_step, state_dim]).to(device)
        step = 0
        while not done:
            step += 1
            next_state_real, reward, done, _ = eval_env.step(action)
            next_state_real_tensor = to_cuda(next_state_real, device).reshape(1, -1)
            rpt_next_state = torch.repeat_interleave(next_state_real_tensor, repeats=noise_rpt, dim=0)
            if step >= warm_start_up:
                rpt_noised_next_state = rpt_next_state + (2 * noise_scale * torch.rand_like(rpt_next_state) - noise_scale)
            else:
                rpt_noised_next_state = rpt_next_state
            rpt_con_action = torch.repeat_interleave(to_cuda(action, device).reshape(1, -1), repeats=noise_rpt, dim=0)
            long_state_con = torch.cat([long_state_con[:, 1:], to_cuda(state0, device).reshape(1, 1, -1)], dim=1)
            rpt_long_state_con = torch.repeat_interleave(long_state_con, repeats=noise_rpt, dim=0)
            rpt_next_state_predict = predictor.denoise_state(rpt_noised_next_state, rpt_con_action, rpt_long_state_con, timestep)
            rpt_noised_action = policy.select_action(rpt_next_state_predict)

            if Algo == "RORL":
                Q_all = policy.critic_target(rpt_next_state_predict, rpt_noised_action)
                Q, _ = policy.get_min_Q(Q_all)
                idx = torch.argmin(Q)
                action = rpt_noised_action[idx].detach().cpu().numpy()
                next_state_predict = rpt_next_state_predict[idx].cpu().detach().numpy()
            else:
                Q1, Q2 = policy.critic_target(rpt_next_state_predict, rpt_noised_action)
                Q = torch.min(Q1, Q2)
                idx = torch.argmin(Q)
                action = rpt_noised_action[idx].detach().cpu().numpy()
                next_state_predict = rpt_next_state_predict[idx].cpu().detach().numpy()

            state1 = next_state_predict
            if real_state:
                state0 = next_state_real
            else:
                state0 = next_state_predict
            reward_ep += reward
        reward_buffer.append(reward_ep)

    avg_reward = np.average(reward_buffer)
    avg_reward = eval_env.get_normalized_score(avg_reward) * 100
    std_reward = np.std(reward_buffer)
    std_reward = eval_env.get_normalized_score(std_reward) * 100
    max_reward = np.max(reward_buffer)
    max_reward = eval_env.get_normalized_score(max_reward) * 100
    min_reward = np.min(reward_buffer)
    min_reward = eval_env.get_normalized_score(min_reward) * 100

    nor_reward = eval_env.get_normalized_score(np.stack(reward_buffer)) * 100

    print(f"Eval over {eval_episodes} episodes (noise std {noise_scale:.3f}, timestep {timestep}): "
          f"mean {avg_reward:.3f}; std {std_reward:.3f}; max {max_reward:.3f}; min {min_reward:.3f}")
    return avg_reward, std_reward, max_reward, min_reward, nor_reward

def to_cuda(x, device=torch.device("cuda:0")):
    return torch.FloatTensor(x).to(device)

# %% Part 3 Markovian Based Policy Evaluation in masked environment
def eval_policy_mask(policy, eval_env, eval_episodes=10, mask_dim=None):
    mask = np.ones(eval_env.observation_space.shape[0])
    for idx in mask_dim:
        mask[idx] = 0.

    reward_buffer = []
    for _ in range(eval_episodes):
        reward_ep = 0.
        state, done = eval_env.reset(), False
        state_dim = state.shape[0]
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)
            next_state = next_state * mask
            reward_ep += reward
            state = next_state

        reward_buffer.append(reward_ep)

    avg_reward = np.average(reward_buffer)
    avg_reward = eval_env.get_normalized_score(avg_reward) * 100
    std_reward = np.std(reward_buffer)
    std_reward = eval_env.get_normalized_score(std_reward) * 100
    max_reward = np.max(reward_buffer)
    max_reward = eval_env.get_normalized_score(max_reward) * 100
    min_reward = np.min(reward_buffer)
    min_reward = eval_env.get_normalized_score(min_reward) * 100

    nor_reward = eval_env.get_normalized_score(np.stack(reward_buffer)) * 100

    print(f"Eval over {eval_episodes} episodes (Mask on dim {mask_dim}): mean {avg_reward:.3f}, std {std_reward:.3f}, "
          f"max {max_reward:.3f}, min {min_reward:.3f}")
    return avg_reward, std_reward, max_reward, min_reward, nor_reward

def eval_DMBP_policy_mask(policy, predictor, eval_env, eval_episodes=10, mask_dim=None, reverse_step=None):
    mask = np.ones(eval_env.observation_space.shape[0])
    warm_start_up = 4
    device = policy.device

    for idx in mask_dim:
        mask[idx] = 0.
    mask_tensor = to_cuda(mask, device).reshape(1, -1)

    reward_buffer = []
    for _ in range(eval_episodes):
        reward_ep = 0.
        state, done = eval_env.reset(), False
        state_dim = state.shape[0]
        long_state_con = torch.zeros([1, predictor.condition_step, state_dim]).to(device)
        step = 0
        while not done:
            step += 1
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)
            next_state_mask = next_state * mask

            if step >= warm_start_up:
                next_state_mask = to_cuda(next_state_mask, device).reshape(1, -1)
            else:
                next_state_mask = to_cuda(next_state, device).reshape(1, -1)

            action = to_cuda(action, device).reshape(1, -1)
            state = to_cuda(state, device).reshape(1, 1, -1)
            long_state_con = torch.cat([long_state_con[:, 1:], state], dim=1)
            if np.sum(mask) == eval_env.observation_space.shape[0]:
                next_state_recon = next_state
            else:
                # next_state_recon = next_state_mask.cpu().numpy().reshape(-1)
                next_state_recon = predictor.demask_state(next_state_mask, action, long_state_con, mask_tensor, reverse_step)

            if step <= warm_start_up:
                next_state_recon = next_state

            if np.abs(next_state_recon[mask_dim].any()) > 30:
                next_state_recon[mask_dim] = 0.
                logger.warning("Prediction Distributional Shift Alert")
            # print(next_state_recon)
            state = next_state_recon
            reward_ep += reward

        reward_buffer.append(reward_ep)

    avg_reward = np.average(reward_buffer)
    avg_reward = eval_env.get_normalized_score(avg_reward) * 100
    std_reward = np.std(reward_buffer)
    std_reward = eval_env.get_normalized_score(std_reward) * 100
    max_reward = np.max(reward_buffer)
    max_reward = eval_env.get_normalized_score(max_reward) * 100
    min_reward = np.min(reward_buffer)
    min_reward = eval_env.get_normalized_score(min_reward) * 100

    nor_reward = eval_env.get_normalized_score(np.stack(reward_buffer)) * 100

    print(f"Eval over {eval_episodes} episodes (Mask on dim {mask_dim}): mean {avg_reward:.3f}, std {std_reward:.3f}, "
          f"max {max_reward:.3f}, min {min_reward:.3f}")
    return avg_reward, std_reward, max_reward, min_reward, nor_reward


# %% Part 2 Trajectory Based Policy Evaluation
# Note: eval_DT is for evaluating Decision Transformer without noise
# Note: eval_DT_noise is for evaluating Decision Transformer with Gaussian or Uniformly distributed noise
# Note: evaluate_DT_rtg is called by eval_DT and eval_DT_noise for one episode evaluation

def eval_DT(policy, eval_env, eval_episodes=10, target_reward=3000):
    reward_buffer = []
    length_buffer = []
    for i in range(eval_episodes):
        with torch.no_grad():
            total_return, total_length = evaluate_DT_rtg(eval_env, policy, target_reward)
            reward_buffer.append(total_return)
            length_buffer.append(total_length)

    avg_reward = np.average(reward_buffer)
    avg_reward = eval_env.get_normalized_score(avg_reward) * 100
    std_reward = np.std(reward_buffer)
    std_reward = eval_env.get_normalized_score(std_reward) * 100
    max_reward = np.max(reward_buffer)
    max_reward = eval_env.get_normalized_score(max_reward) * 100
    min_reward = np.min(reward_buffer)
    min_reward = eval_env.get_normalized_score(min_reward) * 100
    target_reward_normalize = eval_env.get_normalized_score(target_reward) * 100

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Target Reward of {target_reward_normalize:.3f}")
    print(f"Eval-Episode over {eval_episodes} episodes: "
          f"mean {avg_reward:.3f} and std {std_reward:.3f}")
    print(f"With Episode length of mean {np.mean(length_buffer):.3f} and std {np.std(length_buffer):.3f}")

    return avg_reward, std_reward, max_reward, min_reward

def eval_DT_noise(policy, eval_env, eval_episodes=10, target_reward=3000, noise_scale=0., Gaussian=True):
    reward_buffer = []
    length_buffer = []
    for i in range(eval_episodes):
        with torch.no_grad():
            total_return, total_length = evaluate_DT_rtg(eval_env, policy, target_reward,
                                                         'noise', noise_scale, Gaussian)
            reward_buffer.append(total_return)
            length_buffer.append(total_length)

    avg_reward = np.average(reward_buffer)
    avg_reward = eval_env.get_normalized_score(avg_reward) * 100
    std_reward = np.std(reward_buffer)
    std_reward = eval_env.get_normalized_score(std_reward) * 100
    max_reward = np.max(reward_buffer)
    max_reward = eval_env.get_normalized_score(max_reward) * 100
    min_reward = np.min(reward_buffer)
    min_reward = eval_env.get_normalized_score(min_reward) * 100
    target_reward_normalize = eval_env.get_normalized_score(target_reward) * 100

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    if Gaussian:
        print(f"Target Reward of {target_reward_normalize:.3f} with Gaussian Noise scale of {noise_scale}")
    else:
        print(f"Target Reward of {target_reward_normalize:.3f} with Uniform Noise scale of {noise_scale}")
    print(f"Eval-Episode over {eval_episodes} episodes: "
          f"mean {avg_reward:.3f} and std {std_reward:.3f}")
    print(f"With Episode length of mean {np.mean(length_buffer):.3f} and std {np.std(length_buffer):.3f}")

    return avg_reward, std_reward, max_reward, min_reward


def evaluate_DT_rtg(
        env,
        policy,
        target_return=None,
        mode='normal',   # 'normal' or 'noise' or 'delayed'
        noise_scale=0.,
        Gaussian=True
    ):

    state_dim = policy.state_dim
    act_dim = policy.act_dim
    state_mean = policy.state_mean
    state_std = policy.state_std
    max_ep_len = policy.max_ep_len
    device = policy.device
    scale = policy.scale

    if target_return != None:
        target_return = target_return/policy.scale

    state = env.reset()
    # if mode == 'noise':
    #     state = state + noise_scale * np.random.randn(state_dim)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0

    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = policy.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        if mode == 'noise':
            if Gaussian:   # Gaussian Noise: Noise scale is the std of noise
                state = state + noise_scale * np.random.randn(state_dim)
            else:          # Uniform Noise: Noise scale is upper bound and lower bound
                state = state + (2 * noise_scale * np.random.rand(state_dim) - noise_scale)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0, -1] - (reward/scale)
        else:
            pred_return = target_return[0, -1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

# Note: eval_DT_with_DMBP is for evaluating Decision Transformer with DMBP against noised observations
# Note: evaluate_DT_with_DMBP_rtg is called by eval_DT_with_DMBP for one episode evaluation

# def eval_DT_with_DMBP(policy, predictor, eval_env, eval_episodes=10, target_reward=3000,
#                       noise_scale=0., Gaussian=True):
#     reward_buffer = []
#     length_buffer = []
#     if Gaussian:
#         timestep = int((noise_scale + 0.02) / 0.03)
#     else:
#         timestep = int((noise_scale + 0.04) / 0.05)
#
#     for i in range(eval_episodes):
#         with torch.no_grad():
#             total_return, total_length = evaluate_DT_with_DMBP_rtg(eval_env, policy, predictor, timestep, target_reward,
#                                                                    'noise', noise_scale, Gaussian)
#             reward_buffer.append(total_return)
#             length_buffer.append(total_length)
#
#     avg_reward = np.average(reward_buffer)
#     avg_reward = eval_env.get_normalized_score(avg_reward) * 100
#     std_reward = np.std(reward_buffer)
#     std_reward = eval_env.get_normalized_score(std_reward) * 100
#     max_reward = np.max(reward_buffer)
#     max_reward = eval_env.get_normalized_score(max_reward) * 100
#     min_reward = np.min(reward_buffer)
#     min_reward = eval_env.get_normalized_score(min_reward) * 100
#     target_reward_normalize = eval_env.get_normalized_score(target_reward) * 100
#
#     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#     if Gaussian:
#         print(f"Target Reward of {target_reward_normalize:.3f} with Gaussian Noise scale of {noise_scale}")
#     else:
#         print(f"Target Reward of {target_reward_normalize:.3f} with Uniform Noise scale of {noise_scale}")
#     print(f'Diffusion Timestep is {timestep}')
#     print(f"Eval-Episode over {eval_episodes} episodes: "
#           f"mean {avg_reward:.3f} and std {std_reward:.3f}")
#     print(f"With Episode length of mean {np.mean(length_buffer):.3f} and std {np.std(length_buffer):.3f}")
#
#     return avg_reward, std_reward, max_reward, min_reward
#
# def evaluate_DT_with_DMBP_rtg(
#         env,
#         policy,
#         predictor,
#         timestep,
#         target_return=None,
#         mode='noise',   # 'normal' or 'noise' or 'delayed'
#         noise_scale=0.,
#         Gaussian=True
#     ):
#
#     state_dim = policy.state_dim
#     act_dim = policy.act_dim
#     state_mean = policy.state_mean
#     state_std = policy.state_std
#     max_ep_len = policy.max_ep_len
#     device = policy.device
#     scale = policy.scale
#
#     if target_return != None:
#         target_return = target_return/policy.scale
#
#     state1 = env.reset()
#     state0 = state1
#     # if mode == 'noise':
#     #     state = state + noise_scale * np.random.randn(state_dim)
#
#     # we keep all the histories on the device
#     # note that the latest action and reward will be "padding"
#     states = torch.from_numpy(state0).reshape(1, state_dim).to(device=device, dtype=torch.float32)
#     actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
#     rewards = torch.zeros(0, device=device, dtype=torch.float32)
#
#     ep_return = target_return
#     target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
#     timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
#
#     sim_states = []
#
#     episode_return, episode_length = 0, 0
#     if method == 'smooth_q' or method == 'noised_condition_smooth_q' or method == "enoise_smooth_q":
#         non_smooth_q = 0
#     for t in range(max_ep_len):
#         # add padding
#         actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
#         rewards = torch.cat([rewards, torch.zeros(1, device=device)])
#         action = policy.get_action(
#             (states.to(dtype=torch.float32) - state_mean) / state_std,
#             actions.to(dtype=torch.float32),
#             rewards.to(dtype=torch.float32),
#             target_return.to(dtype=torch.float32),
#             timesteps.to(dtype=torch.long),
#         )
#         actions[-1] = action
#         action = action.detach().cpu().numpy()
#
#         next_state_real, reward, done, _ = env.step(action)
#         next_state_noise = next_state_real + noise_scale * np.random.randn(state_dim)
#         next_state_noise = to_cuda(next_state_noise, device).reshape(1, -1)
#         action = to_cuda(action, device).reshape(1, -1)
#         state = to_cuda(state0, device).reshape(1, -1)
#
#         if method == 'normal':
#             next_state_predict = predictor.denoise_state(next_state_noise, action, state, timestep)
#         elif method == 'smooth_q' or method == 'noised_condition_smooth_q' or method == "enoise_smooth_q":
#             next_state_predict, non_smooth_q = predictor.denoise_state(next_state_noise, action, state, timestep,
#                                                                        reward, method,
#                                                                        policy, non_smooth_q)
#         else:
#             next_state_predict = predictor.denoise_state(next_state_noise, action, state, timestep, reward, method,
#                                                          policy)
#         # state1 = next_state_predict.reshape(-1)
#         state0 = next_state_predict.reshape(-1)
#
#         # cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
#         states = torch.cat([states, torch.tensor(next_state_predict.reshape(1, -1),
#                                                  device=device, dtype=torch.float32)], dim=0)
#         rewards[-1] = reward
#
#         if mode != 'delayed':
#             pred_return = target_return[0,-1] - (reward/scale)
#         else:
#             pred_return = target_return[0,-1]
#         target_return = torch.cat(
#             [target_return, pred_return.reshape(1, 1)], dim=1)
#         timesteps = torch.cat(
#             [timesteps,
#              torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)
#
#         episode_return += reward
#         episode_length += 1
#
#         if done:
#             break
#
#     return episode_return, episode_length

