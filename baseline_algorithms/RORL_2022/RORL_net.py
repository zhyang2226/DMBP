#%% Part 0 import package and Global Parameters
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from torch.distributions import kl_divergence

import numpy as np
import random
from loguru import logger
import wandb

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = 1e-6

#%% Part 1 Global Function Definition

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def soft_update(target, source, tau): # Target will be updated but Source will not change
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):      # Target will be updated but Source will not change
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

#%% Part 2 Network Structure Definition
class QNetwork(nn.Module): # Critic (Judge the S+A with Double Q) -> Double Q are independent
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear11 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear12 = nn.Linear(hidden_dim, hidden_dim)
        self.linear13 = nn.Linear(hidden_dim, hidden_dim)
        self.linear14 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear21 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear22 = nn.Linear(hidden_dim, hidden_dim)
        self.linear23 = nn.Linear(hidden_dim, hidden_dim)
        self.linear24 = nn.Linear(hidden_dim, 1)

        # Q3 architecture
        self.linear31 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear32 = nn.Linear(hidden_dim, hidden_dim)
        self.linear33 = nn.Linear(hidden_dim, hidden_dim)
        self.linear34 = nn.Linear(hidden_dim, 1)

        # Q4 architecture
        self.linear41 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear42 = nn.Linear(hidden_dim, hidden_dim)
        self.linear43 = nn.Linear(hidden_dim, hidden_dim)
        self.linear44 = nn.Linear(hidden_dim, 1)

        # Q5 architecture
        self.linear51 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear52 = nn.Linear(hidden_dim, hidden_dim)
        self.linear53 = nn.Linear(hidden_dim, hidden_dim)
        self.linear54 = nn.Linear(hidden_dim, 1)

        # Q6 architecture
        self.linear61 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear62 = nn.Linear(hidden_dim, hidden_dim)
        self.linear63 = nn.Linear(hidden_dim, hidden_dim)
        self.linear64 = nn.Linear(hidden_dim, 1)

        # Q7 architecture
        self.linear71 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear72 = nn.Linear(hidden_dim, hidden_dim)
        self.linear73 = nn.Linear(hidden_dim, hidden_dim)
        self.linear74 = nn.Linear(hidden_dim, 1)

        # Q8 architecture
        self.linear81 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear82 = nn.Linear(hidden_dim, hidden_dim)
        self.linear83 = nn.Linear(hidden_dim, hidden_dim)
        self.linear84 = nn.Linear(hidden_dim, 1)

        # Q9 architecture
        self.linear91 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear92 = nn.Linear(hidden_dim, hidden_dim)
        self.linear93 = nn.Linear(hidden_dim, hidden_dim)
        self.linear94 = nn.Linear(hidden_dim, 1)

        # Q10 architecture
        self.linear01 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear02 = nn.Linear(hidden_dim, hidden_dim)
        self.linear03 = nn.Linear(hidden_dim, hidden_dim)
        self.linear04 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1) # Network input is [State + Act]

        x1 = self.linear14(F.relu(self.linear13(F.relu(self.linear12(F.relu(self.linear11(xu)))))))
        x2 = self.linear24(F.relu(self.linear23(F.relu(self.linear22(F.relu(self.linear21(xu)))))))
        x3 = self.linear34(F.relu(self.linear33(F.relu(self.linear32(F.relu(self.linear31(xu)))))))
        x4 = self.linear44(F.relu(self.linear43(F.relu(self.linear42(F.relu(self.linear41(xu)))))))
        x5 = self.linear54(F.relu(self.linear53(F.relu(self.linear52(F.relu(self.linear51(xu)))))))
        x6 = self.linear64(F.relu(self.linear63(F.relu(self.linear62(F.relu(self.linear61(xu)))))))
        x7 = self.linear74(F.relu(self.linear73(F.relu(self.linear72(F.relu(self.linear71(xu)))))))
        x8 = self.linear84(F.relu(self.linear83(F.relu(self.linear82(F.relu(self.linear81(xu)))))))
        x9 = self.linear94(F.relu(self.linear93(F.relu(self.linear92(F.relu(self.linear91(xu)))))))
        x10 = self.linear04(F.relu(self.linear03(F.relu(self.linear02(F.relu(self.linear01(xu)))))))

        return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10

class GaussianPolicy(nn.Module):  # Gaussian Actor -> log_prod when sampling is NOT 0 (num_input is state_dim)
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, reparameterize=True):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # logger.info(f"Mean is {mean} and Std is {std}")
        normal = Normal(mean, std)
        if reparameterize == True:  # for reparameterization trick (mean + std * N(0,1))
            x_t = mean + std * Normal(torch.zeros_like(mean), torch.ones_like(std)).sample()
            x_t.requires_grad_()
        else:
            x_t = normal.sample().detach()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t * y_t) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, log_std

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


#%% Part 3 Agent Class Definition
class RORL(object):
    def __init__(self, num_inputs, action_space, device, config):
        self.gamma = config["gamma"]  # 0.99
        self.soft_tau = config["soft_tau"]  # 0.005
        self.alpha = config["alpha"]  # 1.0
        self.auto_tune_entropy = config["auto_tune_entropy"]
        self.max_q_backup = config["max_q_backup"]
        self.deterministic_backup = config["deterministic_backup"]
        self.eta = config["eta"]
        self.num_inputs = num_inputs
        self.action_space = action_space
        self.device = device  # cuda or cpu

        self.beta_Q = config["beta_Q"] # 0.0001
        self.beta_P = config["beta_P"]  # 0.1 ~ 1.0
        self.beta_ood = config["beta_ood"]  # 0.0 ~ 0.5
        self.eps_Q = config["eps_Q"]  # 0.001 ~ 0.01
        self.eps_P = config["eps_P"]  # 0.001 ~ 0.01
        self.eps_ood = config["eps_ood"]  # 0.0 ~ 0.01
        self.tau = config["tau"]  # 0.2
        self.n_sample = config["n_sample"]  # 20

        self.lambda_max = config['lambda_max']
        self.lambda_min = config["lambda_min"]
        self.lambda_decay = config["lambda_decay"]

        self.print_more_info = config['print_more_info']
        self.SAC10 = config["SAC10"]

        if self.auto_tune_entropy:
            self.target_entropy = -np.prod(action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=config["policy_lr"])

        self.critic_criterion = nn.MSELoss(reduction='none')
        self.critic = QNetwork(num_inputs, action_space.shape[0], config["hidden_size"]).to(device=self.device)  # 256
        self.critic_optim = Adam(self.critic.parameters(), lr=config["q_lr"])  # 0.0003
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], config["hidden_size"]).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], config["hidden_size"], action_space)
        self.policy.to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=config["policy_lr"])  # 0.0003

        self.step = 0

    def select_action(self, state, evaluate=True):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        input_dim = state.shape[0]
        if evaluate is False: # With Noise
            action, _, _, _ = self.policy.sample(state, reparameterize=False)
        else:                 # No Noise
            _, _, action, _ = self.policy.sample(state, reparameterize=False)
            action = torch.tanh(action) * self.policy.action_scale + self.policy.action_bias
        if input_dim == 1:
            return action.detach().cpu().numpy()[0]
        else:
            return action

    def get_noise_obs(self, obs, radius):   # finished
        M, N = obs.shape[0], obs.shape[1]
        num_samples = self.n_sample
        delta_s = 2 * radius * (torch.rand(num_samples, N, device=self.device) - 0.5)
        tmp_obs = obs.reshape(-1, 1, N).repeat(1, num_samples, 1).reshape(-1, N)
        delta_s = delta_s.reshape(-1, num_samples, N).repeat(M, 1, 1).reshape(-1, N)
        noised_obs = tmp_obs + delta_s
        return noised_obs

    def get_min_Q(self,q_critic):   # finished
        q_cat = torch.stack(q_critic, dim=1).reshape(-1, 10)
        qmin = torch.min(q_cat, dim=1)
        qmin_value, qmin_index = qmin[0].reshape(-1, 1), qmin[1].reshape(-1, 1)
        return qmin_value, qmin_index

    def cal_critic_loss(self, q_target, q_current):  # finished
        q_current = torch.stack(q_current, dim=0)
        q_target = q_target.reshape(1, -1, 1).repeat(10, 1, 1)
        q_loss = self.critic_criterion(q_target, q_current)
        return q_loss.mean(dim=(1, 2)).sum()

    def cal_critic_noise_loss(self, q_pred_noised, q_current_actions, batch_size):   # finished
        q_pred_noised = torch.stack(q_pred_noised, dim=0)
        q_current_actions = (torch.stack(q_current_actions, dim=0).repeat(1, 1, self.n_sample)
                             .reshape(10, -1, 1))
        diff = q_pred_noised - q_current_actions
        zero_tensor = torch.zeros(diff.shape, device=self.device)
        pos, neg = torch.maximum(diff, zero_tensor), torch.minimum(diff, zero_tensor)
        noise_Q_loss = (1 - self.tau) * pos.square().mean(axis=0) + self.tau * neg.square().mean(axis=0)
        noise_Q_loss = noise_Q_loss.reshape(batch_size, self.n_sample)
        noise_Q_loss_max = noise_Q_loss[np.arange(batch_size), torch.argmax(noise_Q_loss, axis=-1)].mean()
        return noise_Q_loss_max

    def cal_critic_ood_loss(self, ood_q_pred):
        ood_loss = torch.tensor(0., device=self.device)
        if self.lambda_max > 0:
            ood_q_pred = torch.stack(ood_q_pred, dim=0)
            ood_target = ood_q_pred - self.lambda_max * ood_q_pred.std(axis=0)
            ood_loss = self.critic_criterion(ood_target.detach(), ood_q_pred).mean()
        return ood_loss

    def train(self, Buffer, iteration=1000, batch_size=256, logwriter=False):
        act_dim = self.action_space.shape[0]
        obs_dim = self.num_inputs

        for i in range(int(iteration)):
            # Sampling from Buffer
            state_batch, next_state_batch, action_batch, reward_batch, mask_batch = Buffer.sample(batch_size)

            # Core RORL Network Updating
            # Step 1 Policy Loss based on beta_P and eps_P -> Eq(6) on page 5
            # Alpha Loss and Classical Policy Loss
            new_action_batch, log_pi, policy_mean, policy_log_std = self.policy.sample(state_batch)
            if self.auto_tune_entropy:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = self.alpha
            q_new_actions = self.critic(state_batch, new_action_batch)
            q_min_new_actions, q_min_idx_new_actions = self.get_min_Q(q_new_actions)
            policy_loss = (alpha * log_pi - q_min_new_actions).mean()

            # Plus robust term (range eps_P and coef beta_P)
            if not self.SAC10 and self.beta_P > 0 and self.eps_P > 0:
                p_noised_obs = self.get_noise_obs(state_batch, self.eps_P)
                p_noised_mean, p_noised_log_std = self.policy(p_noised_obs)
                p_action_dist = Normal(policy_mean.reshape(-1,1,act_dim).repeat(1,self.n_sample,1).reshape(-1,act_dim),
                                       policy_log_std.exp().reshape(-1,1,act_dim).repeat(1,self.n_sample,1).reshape(-1,act_dim))
                p_noised_action_dist = Normal(p_noised_mean, p_noised_log_std.exp())
                kl_loss = kl_divergence(p_action_dist, p_noised_action_dist).sum(axis=-1) + \
                          kl_divergence(p_noised_action_dist, p_action_dist).sum(axis=-1)
                kl_loss = kl_loss.reshape(batch_size, self.n_sample)
                max_id = torch.argmax(kl_loss, axis=1)
                kl_loss_max = kl_loss[np.arange(batch_size), max_id].mean()
                policy_loss = policy_loss + self.beta_P * kl_loss_max
                kl_loss_max_log = kl_loss_max.detach().cpu().numpy()
            else:
                kl_loss_max_log = 0

            # Step 2 Classical Critic Loss -> Eq(5) on page 5
            with torch.no_grad():
                next_actions, next_log_pi, _, _ = self.policy.sample(next_state_batch, reparameterize=False)
                target_q_values = self.critic_target(next_state_batch, next_actions)
                target_q_min_values, target_q_min_values_idx = self.get_min_Q(target_q_values)
                target_q_min_values = target_q_min_values - alpha * next_log_pi
                future_value = mask_batch * self.gamma * target_q_min_values
                q_target = reward_batch + future_value
            q_current_actions = self.critic(state_batch, action_batch)
            q_loss = self.cal_critic_loss(q_target, q_current_actions)

            # Step 3 Smooth Loss based on beta_Q and eps_Q -> Eq(5) on page 5
            if not self.SAC10 and self.eps_Q > 0 and self.beta_Q > 0:
                qs_noised_obs = self.get_noise_obs(state_batch, self.eps_Q)
                q_pred_noised = self.critic(qs_noised_obs, action_batch.reshape(-1,1,act_dim)
                                              .repeat(1,self.n_sample,1).reshape(-1,act_dim))
                q_noise_loss = self.cal_critic_noise_loss(q_pred_noised, q_current_actions, batch_size)
                q_loss += self.beta_Q * q_noise_loss
                q_noise_loss_log = q_noise_loss.detach().cpu().numpy()
            else:
                q_noise_loss_log = 0

            # Step 4 OOD Loss based on beta_ood and eps_ood -> Eq(5) on page 5
            if not self.SAC10 and self.eps_ood > 0  and self.beta_ood > 0:
                ood_noised_obs = self.get_noise_obs(state_batch, self.eps_ood)
                ood_noised_actions, _, _, _ = self.policy.sample(ood_noised_obs, reparameterize=False)
                ood_q_pred = self.critic(ood_noised_obs, ood_noised_actions)
                ood_loss = self.cal_critic_ood_loss(ood_q_pred)
                q_loss += self.beta_ood * ood_loss
                ood_loss_log = ood_loss.detach().cpu().numpy()
            else:
                ood_loss_log = 0

            # Update lambda value
            if self.lambda_max > 0:
                self.lambda_max = max(self.lambda_max - self.lambda_decay, self.lambda_min)

            # Step 5 Update network
            if self.auto_tune_entropy:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            self.critic_optim.zero_grad()
            q_loss.backward()
            self.critic_optim.step()

            if self.step % 200 == 0 and logwriter:
                if self.auto_tune_entropy:
                    wandb.log({"Loss/alpha": alpha.detach().cpu().numpy(),
                               "Loss/alpha_loss": alpha_loss.detach().cpu().numpy()}, step=self.step)
                else:
                    wandb.log({"Loss/alpha": alpha,
                               "Loss/alpha_loss": alpha_loss}, step=self.step)

                wandb.log({"Loss/Policy_Loss": policy_loss.detach().cpu().numpy(),
                           "Loss/KL_loss_max": kl_loss_max_log,
                           "Loss/Q_mean": torch.stack(q_current_actions).detach().cpu().numpy().mean(),
                           "Loss/Qnet_Loss": q_loss.detach().cpu().numpy(),
                           "Loss/Q_noise_Loss": q_noise_loss_log,
                           "Loss/OOD_Loss": ood_loss_log,
                           "Loss/OOD_lambda": self.lambda_max}, step=self.step)

            soft_update(self.critic_target, self.critic, self.soft_tau)
            self.step += 1


    # Save model parameters
    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, file_name)

    # Load model parameters
    def load_model(self, filename, evaluate=False, device_idx=0):
        logger.info(f'Loading models from {filename}')
        if filename is not None:
            if device_idx == -1:
                checkpoint = torch.load(filename, map_location=f'cpu')
            else:
                checkpoint = torch.load(filename, map_location=f'cuda:{device_idx}')
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()