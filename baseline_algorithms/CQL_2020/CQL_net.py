# %% Part 0 import package and Global Parameters
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
import numpy as np
import random
from loguru import logger
import itertools
import wandb

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# %% Part 1 Global Function Definition
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def soft_update(target, source, tau):  # Target will be updated but Source will not change
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):  # Target will be updated but Source will not change
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# %% Part 2 Network Structure Definition
class QNetwork(nn.Module): # Critic (Judge the S+A with Double Q) -> Double Q are independent
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1) # Network input is [State + Act]

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class GaussianPolicy(nn.Module):  # Gaussian Actor -> log_prod when sampling is NOT 0 (num_input is state_dim)
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

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
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # logger.info(f"Mean is {mean} and Std is {std}")
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class DeterministicPolicy(nn.Module):  # Deterministic Actor -> log_prod when sampling is 0 (num_input is state_dim)
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1) # Noise is Normal(miu=0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

#%% Part 3 Agent Class Definition
class CQL(object):
    def __init__(self, num_inputs, action_space, device, config):
        self.gamma = config["gamma"]  # 0.99
        self.tau = config["tau"]  # 0.005
        self.alpha = config["alpha"]  # 0.2
        self.action_space = action_space

        self.policy_type = config["policy"]  # Gaussian
        self.target_update_interval = config["target_update_interval"]  # 1
        self.automatic_entropy_tuning = config["automatic_entropy_tuning"]  # False
        self.device = device  # cuda or cpu
        self.Conservative_Q = config["Conservative_Q"] # int(0), int(1), int(2), int(3)

        self.critic = QNetwork(num_inputs, action_space.shape[0], config["hidden_size"]).to(device=self.device)  # 256
        self.critic_optim = Adam(self.critic.parameters(), lr=config["q_lr"])  # 0.0003
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], config["hidden_size"]).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.step = 0

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=config["policy_lr"])

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], config["hidden_size"], action_space)
            self.policy.to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=config["policy_lr"])

        else:  # Deterministic Do not allow autotune alpha value as alpha will always be 0
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], config["hidden_size"], action_space)
            self.policy.to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=config["policy_lr"])

    def select_action(self, state, evaluate=True):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        if evaluate is False: # With Noise
            action, _, _ = self.policy.sample(state)
        else:                 # No Noise
            _, _, action = self.policy.sample(state)
        if input_dim == 1:
            return action.detach().cpu().numpy()[0]
        else:
            return action

    def train(self, Buffer, iteration=1000, batch_size=256, logwriter=False):
        for i in range(int(iteration)):
            # Sampling from Buffer
            state_batch, next_state_batch, action_batch, reward_batch, mask_batch = Buffer.sample(batch_size)

            # Network Updating
            # Core Step1 is to update the Critic Network
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            qf1, qf2 = self.critic(state_batch, action_batch)
            qf1_loss = F.mse_loss(qf1, next_q_value)
            qf2_loss = F.mse_loss(qf2, next_q_value)
            # Conservative term loss:
            if self.Conservative_Q == 0: # Core Q0 No Conservative Term
                qf_loss = qf1_loss + qf2_loss

            elif self.Conservative_Q == 1: # Core Q1 Conservative Term is Q itself
                qf1_cql_loss = qf1.mean()
                qf2_cql_loss = qf2.mean()
                qf_loss = qf1_loss + qf2_loss + qf1_cql_loss + qf2_cql_loss

            elif self.Conservative_Q == 2 or self.Conservative_Q == 3: # Core Q2 and Q3 Conservative Term is final version
                SP_act = 10
                cql_temp = 1
                cql_clip_diff_min = -1e6
                cql_clip_diff_max = 1e6
                cql_lagrange = False
                cql_weight = 5.
                with torch.no_grad():
                    cql_random_actions = next_state_action.new_empty((batch_size, self.action_space.shape[0], SP_act), requires_grad=False).uniform_(-1, 1)
                    cql_current_actions = next_state_action.new_empty((batch_size, self.action_space.shape[0], SP_act), requires_grad=False)
                    cql_current_log_pis = next_state_action.new_empty((batch_size, self.action_space.shape[0], SP_act), requires_grad=False)
                    cql_next_actions = next_state_action.new_empty((batch_size, self.action_space.shape[0], SP_act), requires_grad=False)
                    cql_next_log_pis = next_state_action.new_empty((batch_size, self.action_space.shape[0], SP_act), requires_grad=False)
                    for k in range(SP_act):
                        cql_current_actions[:,:,k], cql_current_log_pis[:,:,k], _ = self.policy.sample(state_batch)
                        cql_next_actions[:,:,k],   cql_next_log_pis[:,:,k],   _   = self.policy.sample(next_state_batch)
                cql_q1_rand = qf1.new_empty((batch_size, 1, SP_act))
                cql_q2_rand = qf2.new_empty((batch_size, 1, SP_act))
                cql_q1_current = qf1.new_empty((batch_size, 1, SP_act))
                cql_q2_current = qf2.new_empty((batch_size, 1, SP_act))
                cql_q1_next = qf1.new_empty((batch_size, 1, SP_act))
                cql_q2_next = qf2.new_empty((batch_size, 1, SP_act))

                for k in range(SP_act):
                    cql_q1_rand[:, :, k], cql_q2_rand[:, :, k] = self.critic(state_batch, cql_random_actions[:,:,k])
                    cql_q1_current[:, :, k], cql_q2_current[:, :, k] = self.critic(state_batch, cql_current_actions[:,:,k])
                    cql_q1_next[:, :, k], cql_q2_next[:, :, k] = self.critic(state_batch, cql_next_actions[:,:,k])

                cql_cat_q1 = torch.cat([cql_q1_rand, torch.unsqueeze(qf1, 2), cql_q1_next, cql_q1_current], dim=2)
                cql_cat_q2 = torch.cat([cql_q2_rand, torch.unsqueeze(qf2, 2), cql_q2_next, cql_q2_current], dim=2)


                if self.Conservative_Q == 3:
                    random_density = np.log(0.5 ** self.action_space.shape[0])
                    cql_cat_q1 = torch.cat([cql_q1_rand - random_density, cql_q1_next - torch.mean(cql_next_log_pis, dim=1).unsqueeze(1),
                         cql_q1_current - torch.mean(cql_current_log_pis, dim=1).unsqueeze(1)], dim=2)
                    cql_cat_q2 = torch.cat([cql_q2_rand - random_density, cql_q2_next - torch.mean(cql_next_log_pis, dim=1).unsqueeze(1),
                         cql_q2_current - torch.mean(cql_current_log_pis, dim=1).unsqueeze(1)], dim=2)

                cql_qf1_ood = torch.logsumexp(cql_cat_q1 / cql_temp, dim=2) * cql_temp
                cql_qf2_ood = torch.logsumexp(cql_cat_q2 / cql_temp, dim=2) * cql_temp

                """Subtract the log likelihood of data"""
                cql_qf1_diff = torch.clamp(cql_qf1_ood - qf1, cql_clip_diff_min, cql_clip_diff_max).mean()
                cql_qf2_diff = torch.clamp(cql_qf2_ood - qf2, cql_clip_diff_min, cql_clip_diff_max).mean()

                # if cql_lagrange:
                #     alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
                #     cql_min_qf1_loss = alpha_prime * self.config.cql_min_q_weight * (
                #                 cql_qf1_diff - self.config.cql_target_action_gap)
                #     cql_min_qf2_loss = alpha_prime * self.config.cql_min_q_weight * (
                #                 cql_qf2_diff - self.config.cql_target_action_gap)
                #
                #     self.alpha_prime_optimizer.zero_grad()
                #     alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
                #     alpha_prime_loss.backward(retain_graph=True)
                #     self.alpha_prime_optimizer.step()
                # else:
                cql_min_qf1_loss = cql_qf1_diff * cql_weight
                cql_min_qf2_loss = cql_qf2_diff * cql_weight
                    # alpha_prime_loss = observations.new_tensor(0.0)
                    # alpha_prime = observations.new_tensor(0.0)

                qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss
            # logger.info(f"qf_loss = {qf_loss.cpu()}")
            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            # Core Step2 is to update the Actor Network
            pi, log_pi, _ = self.policy.sample(state_batch)
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
            # logger.info(f"log_pi = {log_pi.mean().cpu()}")
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Core Step3 is to update the Autotune Factor
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp()
            else:
                alpha_loss = torch.tensor(0.).to(self.device)

            # Core Step4 is to Soft-update the Target
            soft_update(self.critic_target, self.critic, self.tau)
            alpha_Value = self.alpha

            if logwriter and self.step % 100 == 0:
                wandb.log({"Q Loss": qf_loss.item(),
                           "Actor Loss": policy_loss.item(),
                           "Alpha Loss": alpha_loss.item(),
                           "Alpha Value": alpha_Value,}, step=self.step)
            self.step += 1

        return 0, qf_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_Value

    # Save model parameters
    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, file_name)

    # Load model parameters
    def load_model(self, filename, device_idx=0, evaluate=False):
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