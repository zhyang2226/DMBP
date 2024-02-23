#%% Part 0 import package and Global Parameters
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import random
import copy
import math
from loguru import logger
import itertools
import wandb

from baseline_algorithms.Diffusion_QL_2023.Diffusion_model import Diffusion

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

#%% Part 1 Global Function Definition
def setup_seed(seed=1024): # After doing this, the Training results will always be the same for the same seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"Seed {seed} has been set for all modules!")

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

#%% Part 2 Network Definition
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, device, t_dim=16):
        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)

#%% Part 3 Import Network Definition
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

class Diffusion_QL(object):
    def __init__(self, state_dim, action_dim, max_action, device, config, saving_logwriter=False):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=config['beta_schedule'], n_timesteps=config['T'],).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['lr'])

        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.step_start_ema = config['step_start_ema']
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = config['gamma']
        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.print_more = config['print_more_info']

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            state, next_state, action, reward, not_done = replay_buffer.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_model(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            target_q = (reward + not_done * self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.actor.loss(action, state)
            new_action = self.actor(state)

            q1_new_action, q2_new_action = self.critic(state, new_action)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()


            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer and self.step % 200 == 0:
                if self.grad_norm > 0:
                    wandb.log({"Actor Grad Norm": actor_grad_norms.max().item(),
                               'Critic Grad Norm': critic_grad_norms.max().item()}, step=self.step)
                wandb.log({'BC Loss': bc_loss.item(),
                           'QL Loss': q_loss.item(),
                           'Critic Loss': critic_loss.item(),
                           'Target_Q Mean': target_q.mean().item()}, step=self.step)

            if self.step % 100 == 0 and self.print_more:
                print(f"Step{self.step}:   "
                      f"Critic loss is {critic_loss.item():.4f}; ql loss is {q_loss.item():.4f}; "
                      f"actor loss is {actor_loss.item():.4f}; BC loss is {bc_loss.item():.4f}")

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def select_action(self, state, eval=True):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        if eval:
            with torch.no_grad():
                action = self.actor.sample(state_rpt)
                q_value = self.critic_target.q_min(state_rpt, action).flatten().reshape(input_dim, -1)
                if torch.sum(torch.isinf(q_value)+torch.isnan(q_value)) > 0:
                    q_value = torch.where(torch.isinf(q_value), torch.full_like(q_value, 0), q_value)
                    q_value = torch.where(torch.isnan(q_value), torch.full_like(q_value, 0), q_value)
                    logger.warning("Selection Overflow Alert")
                idx = torch.multinomial(F.softmax(q_value), 1)
        else:
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            re_action = action[idx].clip(-1, 1)
            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)
            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor

    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict()}, file_name)


    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])