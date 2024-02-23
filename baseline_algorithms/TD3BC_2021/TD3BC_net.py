import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import wandb

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_BC(object):
    def __init__(
            self,
            Buffer,
            device,
            config,
    ):
        self.state_dim = Buffer.obs_dim
        self.action_dim = Buffer.act_dim
        self.max_action = Buffer.max_action
        self.device = device

        self.discount = config['discount']
        self.tau = config['tau']
        self.policy_noise = config['policy_noise']
        self.noise_clip = config['noise_clip']
        self.policy_freq = config['policy_freq']
        self.alpha = config['alpha']

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.step = 0
        if config['normalize']:
            self.obs_mean = Buffer.obs_mean.to(self.device)
            self.obs_std = Buffer.obs_std.to(self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        normalized_state = (state - self.obs_mean) / self.obs_std
        return self.actor(normalized_state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=256, saving_logwriter=False):

        for it in range(iterations):
            # Sample replay buffer
            state, next_state, action, reward, not_done = replay_buffer.sample(batch_size)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                        torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

                next_action = (
                        self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.step % self.policy_freq == 0:

                # Compute actor loss
                pi = self.actor(state)
                Q = self.critic.Q1(state, pi)
                lmbda = self.alpha / Q.abs().mean().detach()

                BC_loss = F.mse_loss(pi, action)
                actor_loss = -lmbda * Q.mean() + BC_loss

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if saving_logwriter and self.step % 100 == 0:
                wandb.log({"Loss/critic_loss": critic_loss.item(),
                           "Loss/Q_mean": Q.mean().item(),
                           "Loss/lmbda": lmbda.item(),
                           "Loss/BC_loss": BC_loss.item(),
                           "Loss/actor_loss": actor_loss.item()}, step=self.step)
            self.step += 1

    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'actor_target_state_dict': self.actor_target.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict()}, file_name)

    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if device_idx == -1:
            checkpoint = torch.load(file_name, map_location=f'cpu')
        else:
            checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])

        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])