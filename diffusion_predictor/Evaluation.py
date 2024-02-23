import numpy as np
import torch
import wandb

def to_cuda(x, device=torch.device("cuda:0")):
    return torch.FloatTensor(x).to(device)

# %% Eval the Denoiser conditioned on long time horizon
def eval_policy_longcon(policy, predictor, real_state, eval_env, eval_episodes=10, noise_scale=0):
    reward_buffer = []
    device = policy.device
    warm_up_start = 0
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
            action = policy.select_action(np.array(state1))
            next_state_real, reward, done, _ = eval_env.step(action)
            next_state_noise = next_state_real + noise_scale * np.random.randn(state_dim)
            if step >= warm_up_start:
                next_state_noise = to_cuda(next_state_noise, device).reshape(1, -1)
            else:
                next_state_noise = to_cuda(next_state_real, device).reshape(1, -1)
            action = to_cuda(action, device).reshape(1, -1)
            state = to_cuda(state0, device).reshape(1, 1, -1)
            long_state_con = torch.cat([long_state_con[:, 1:], state], dim=1)
            next_state_predict = predictor.denoise_state(next_state_noise, action, long_state_con, timestep)
            if step <= warm_up_start:
                state1 = next_state_real
                state0 = next_state_real
            else:
                state1 = next_state_predict.reshape(-1)
                state0 = next_state_predict.reshape(-1)
            if real_state:
                state0 = next_state_real
            reward_ep += reward
            step += 1
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
    print(f"Eval over {eval_episodes} episodes (noise std {noise_scale:.3f}, timestep {timestep}): "
          f"mean {avg_reward:.3f}; std {std_reward:.3f}; max {max_reward:.3f}; min {min_reward:.3f}")
    return avg_reward, std_reward

# %% Part Rendering image
def render_generation(render, render_path,
                      policy, predictor, real_state, eval_env, noise_scale=0, avg=0):
    device = policy.device
    timestep = int((noise_scale + 0.02) / 0.03)

    while True:
        original_state_buffer = []
        noised_state_buffer = []
        recovered_state_buffer = []

        reward_ep = 0.
        # Note that state0 is for "condition usage" and state1 is for "decision making"
        state1, done = eval_env.reset(), False
        state0 = state1

        original_state_buffer.append(state1)
        noised_state_buffer.append(state1)
        recovered_state_buffer.append(state1)

        state_dim = state1.shape[0]
        long_state_con = torch.zeros([1, predictor.condition_step, state_dim]).to(device)
        step = 0
        while not done:
            action = policy.select_action(np.array(state1))
            next_state_real, reward, done, _ = eval_env.step(action)
            original_state_buffer.append(next_state_real)
            next_state_noise = next_state_real + noise_scale * np.random.randn(state_dim)
            noised_state_buffer.append(next_state_noise)

            # Core: Warm Start up test
            if step >= 4:
                next_state_noise = to_cuda(next_state_noise, device).reshape(1, -1)
            else:
                next_state_noise = to_cuda(next_state_real, device).reshape(1, -1)

            action = to_cuda(action, device).reshape(1, -1)
            state = to_cuda(state0, device).reshape(1, 1, -1)
            long_state_con = torch.cat([long_state_con[:, 1:], state], dim=1)
            next_state_predict = predictor.denoise_state(next_state_noise, action, long_state_con, timestep)
            # state1 = next_state_predict.reshape(-1)
            recovered_state_buffer.append(next_state_predict)

            # Core: Warm Start up test
            if step <= 4:
                state1 = next_state_real
                state0 = next_state_real
            else:
                state1 = next_state_predict.reshape(-1)
                state0 = next_state_predict.reshape(-1)

            if real_state:
                state0 = next_state_real

            reward_ep += reward
            step += 1
        reward_ep = eval_env.get_normalized_score(reward_ep) * 100
        if reward_ep >= avg:
            break

    original_states = np.concatenate(original_state_buffer, axis=0).reshape(1, -1, state_dim)
    noised_states = np.concatenate(noised_state_buffer, axis=0).reshape(1, -1, state_dim)
    recovered_states = np.concatenate(recovered_state_buffer, axis=0).reshape(1, -1, state_dim)

    observations = np.concatenate([original_states, noised_states, recovered_states], axis=0)
    image = render.composite(f"{render_path}_noise{noise_scale:.2}.png", observations)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Rendering reward {reward_ep:.3f} and step {step} (noise std {noise_scale:.3f}, timestep {timestep})")
    return image