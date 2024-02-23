# %% Part 0 import package and Global Parameters
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

import numpy as np
import random
import copy
import math
from loguru import logger
import itertools
import einops
from einops.layers.torch import Rearrange

from diffusion_predictor.Predictor_model import Diffusion
from diffusion_predictor.render_img import MuJoCoRenderer

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# %% Part 1 Global Function Definition
def setup_seed(seed=1024):  # After doing this, the Training results will always be the same for the same seed
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


def soft_update(target, source, tau):  # Target will be updated but Source will not change
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):  # Target will be updated but Source will not change
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# %% Part 2 Network Definition
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


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, device, t_dim=32, embed_dim=64):
        super(MLP, self).__init__()
        self.device = device
        self.t_dim = t_dim
        self.embed_dim = embed_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, 2 * t_dim),
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim)
        )

        input_dim = 2 * t_dim + 3 * embed_dim

        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())
        self.dropout = nn.Dropout(0.1)
        self.final_layer = nn.Linear(256, state_dim)

    def forward(self, noise_state, time, action, state):
        t = self.time_mlp(time)
        x = torch.cat([t, self.action_encoder(action), self.state_encoder(state), self.state_encoder(noise_state)],
                      dim=1)
        x = self.mid_layer(x)
        x = self.dropout(x)
        return self.final_layer(x)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    def __init__(self, state_dim, action_dim, device, cond_dim=8,
                 embed_dim=256, dim_mults=(2, 4), attention=False):
        super(TemporalUnet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.cond_dim = cond_dim
        horizon = self.cond_dim
        self.embed_dim = embed_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 2 * embed_dim),
            nn.Mish(),
            nn.Linear(2 * embed_dim, embed_dim // 2)
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 2 * embed_dim),
            nn.Mish(),
            nn.Linear(2 * embed_dim, embed_dim // 2)
        )

        dims = [embed_dim, *map(lambda m: embed_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        logger.info(f'Models Channel dimensions: {in_out}')

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        time_dim = embed_dim
        horizon_history = []
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            horizon_history.append(horizon)
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=3, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=3, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last and horizon_history[-(ind + 1)] != horizon_history[-(ind + 2)]
                else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon_history[-(ind + 2)]

        self.final_conv = nn.Sequential(
            Conv1dBlock(2 * embed_dim, 2 * embed_dim, kernel_size=3),
            nn.Conv1d(2 * embed_dim, embed_dim // 4, 1),
        )
        if horizon % 2 != 0:
            out_horizon = horizon + 1
        else:
            out_horizon =horizon
        self.mid_layer = nn.Sequential(nn.Linear(out_horizon * embed_dim // 4 + (embed_dim * 3) // 2 + embed_dim, 512),
                                       nn.Mish(),
                                       nn.Linear(512, 512),
                                       nn.Mish(),
                                       nn.Linear(512, 512),
                                       nn.Mish())

        self.final_layer = torch.nn.Linear(512, self.state_dim)

    def forward(self, x, time, action, state_condition, mask=None):
        '''
            x : [ batch x horizon x transition ]
        '''
        batch_size = x.shape[0]
        horizon = state_condition.shape[1]

        encoded_noised_state = self.state_encoder(x)
        encoded_action = self.action_encoder(action)
        encoded_state_conditions = self.state_encoder(state_condition)

        noised_state_rpt = torch.repeat_interleave(encoded_noised_state.reshape(batch_size, 1, -1), repeats=horizon,
                                                   dim=1)

        x = torch.cat([noised_state_rpt, encoded_state_conditions], dim=2)


        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        info = x.reshape(batch_size, -1)
        output = self.mid_layer(torch.cat([info, encoded_noised_state, encoded_action,
                                           encoded_state_conditions[:, -1], t], dim=1))
        output = self.final_layer(output)
        return output


# %% Part 3 Import Network Definition
class Diffusion_Predictor(object):
    def __init__(self, state_dim, action_dim, device, config, log_writer=False):

        self.model = TemporalUnet(state_dim=state_dim, action_dim=action_dim, device=device,
                                  cond_dim=config['condition_length'], embed_dim=config['embed_dim']).to(device)

        self.predictor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model,
                                   beta_schedule=config['beta_schedule'], beta_mode=config["beta_training_mode"],
                                   n_timesteps=config['T'], predict_epsilon=config['predict_epsilon']).to(device)

        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=config['lr'])

        if log_writer:
            wandb.watch(self.predictor)
            wandb.save('Diffusion_Predictor_model.h5')

        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']
        self.n_timestep = config['T']

        self.step = 0
        self.step_start_ema = config['step_start_ema']
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.predictor)
        self.update_ema_every = config['update_ema_every']

        if self.lr_decay:
            self.predictor_lr_scheduler = CosineAnnealingLR(self.predictor_optimizer,
                                                            T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = config['gamma']
        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']
        self.NonM_step = config['non_markovian_step']
        self.condition_step = config['condition_length']
        self.buffer_sample_length = self.NonM_step + self.condition_step
        self.T_scheme = config['T-scheme']

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.predictor)

    def train(self, replay_buffer, iterations, batch_size, log_writer=False):

        metric = {'pred_loss': []}
        for _ in range(iterations):
            pred_loss_list = []
            # Sample replay buffer / batch
            if self.T_scheme == "same" and self.predictor.beta_mode == "partial":
                t = torch.randint(0, int(self.n_timestep * 0.1), (batch_size,), device=self.device).long()
            elif self.T_scheme == "same" and self.predictor.beta_mode == "all":
                t = torch.randint(0, int(self.n_timestep), (batch_size,), device=self.device).long()
            else:
                t = None

            s, a, r, d, rtg, timesteps, mask0 = replay_buffer.sample(batch_size, self.buffer_sample_length,
                                                                     pad_front=True, pad_front_len=self.condition_step)
            pre_state_condition = s[:, 0:self.condition_step]
            next_state = s[:, self.condition_step].reshape(batch_size, -1)
            action = a[:, self.condition_step - 1].reshape(batch_size, -1)
            mask = mask0[:, 0:self.condition_step]
            pred_loss, state_recon = self.predictor.loss(next_state, action, pre_state_condition, mask, t, weights=1.0)
            pred_loss_list.append(pred_loss.item())

            for i in range(1, self.NonM_step):
                pre_state_condition = torch.cat([pre_state_condition[:, 1:], state_recon.reshape(batch_size, 1, -1)],
                                                dim=1)
                next_state = s[:, self.condition_step + i].reshape(batch_size, -1)
                action = a[:, self.condition_step - 1 + i].reshape(batch_size, -1)
                mask = mask0[:, i:(self.condition_step + i)]
                weights = torch.ones_like(state_recon)

                # weights[t <= (i - 1)] = 0.
                # w = weights.cpu().numpy()
                # TT = t.cpu().numpy().reshape(-1, 1)

                pred_loss_plus, state_recon = self.predictor.loss(next_state, action, pre_state_condition, mask, t,
                                                                  weights=weights)
                pred_loss += pred_loss_plus * (i + 1)
                # pred_loss += pred_loss_plus
                pred_loss_list.append(pred_loss_plus.item())

            total_loss = pred_loss

            self.predictor_optimizer.zero_grad()
            total_loss.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.predictor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if log_writer and self.step % 100 == 0:
                wandb.log({"Predictor_Loss/Total_Loss": pred_loss.item()}, step=self.step)
                for loss_num in range(len(pred_loss_list)):
                    wandb.log({f"Predictor_Loss/Step{loss_num}_loss": pred_loss_list[loss_num]}, step=self.step)

            self.step += 1
            metric['pred_loss'].append(pred_loss.item())

        if self.lr_decay:
            self.predictor_lr_scheduler.step()

        return metric

    def denoise_state(self, noise_next_state, current_action, condition_states, timestep,
                      reward=None, method='mean', policy=None,
                      non_smooth=None):

        # Core: Method 1 is to Average 50 results
        input_dim = noise_next_state.shape[0]
        if input_dim == 1:
            current_action_rpt = torch.repeat_interleave(current_action, repeats=50, dim=0)
            condition_states_rpt = torch.repeat_interleave(condition_states, repeats=50, dim=0)
            if timestep >= 1:
                noise_next_state = self.ema_model.recover(noise_next_state, timestep - 1)
            noise_next_state_rpt = torch.repeat_interleave(noise_next_state, repeats=50, dim=0)
            with torch.no_grad():
                return_state = self.ema_model(noise_next_state_rpt, current_action_rpt, condition_states_rpt, timestep
                                              ).cpu().detach().numpy()
            if method == "mean":
                final_state = np.mean(return_state, axis=0)
            elif method == "filter":
                mean_state = np.mean(return_state, axis=0)
                bias = np.abs(return_state - mean_state).sum(axis=1)
                state_after_filter = return_state[bias.argsort()[:25]]
                final_state = np.mean(state_after_filter, axis=0)
            else:
                raise NotImplementedError
            return final_state

        else:
            current_action_rpt = (torch.repeat_interleave(current_action.reshape(1, input_dim, -1), repeats=50, dim=0)
                                  .reshape(50 * input_dim, -1))
            condition_states_rpt = torch.repeat_interleave(condition_states.reshape(1, input_dim, self.condition_step, -1)
                                              , repeats=50, dim=0).reshape(50 * input_dim, self.condition_step, -1)
            if timestep >= 1:
                noise_next_state = self.ema_model.recover(noise_next_state, timestep - 1)
            noise_state_rpt = (torch.repeat_interleave(noise_next_state.reshape(1, input_dim, -1), repeats=50, dim=0)
                               .reshape(50 * input_dim, -1))
            with torch.no_grad():
                return_state = self.ema_model(noise_state_rpt, current_action_rpt, condition_states_rpt, timestep
                                              ).reshape(50, input_dim, -1)
            final_state = torch.mean(return_state, dim=0)
            return final_state

    def demask_state(self, masked_next_state, action, states, mask, reverse_step=2):
        repeat = 50
        masked_next_state = torch.repeat_interleave(masked_next_state, repeats=repeat, dim=0)
        action = torch.repeat_interleave(action, repeats=repeat, dim=0)
        states = torch.repeat_interleave(states, repeats=repeat, dim=0)
        mask = torch.repeat_interleave(mask.reshape(1, -1), repeats=repeat, dim=0)
        mask_reverse = torch.ones_like(mask) - mask

        total_tstp = self.predictor.n_timesteps
        xt = torch.randn_like(masked_next_state)
        with torch.no_grad():
            for i in reversed(range(0, total_tstp)):
                timesteps = torch.full((repeat,), i, device=self.device, dtype=torch.long)
                for k in reversed(range(reverse_step)):
                    # denoise "xt" for one diffusion timestep
                    xt_1_unkown = self.predictor.p_sample(xt, timesteps, action, states) * mask_reverse

                    if i != 0:
                        # TODO: Check the timestep here (Derive the equation)
                        # Adding noise directly from masked_next_state
                        xt_1_known = self.predictor.q_sample(masked_next_state, timesteps - 1) * mask
                        xt_1_recon = xt_1_unkown + xt_1_known
                    else:
                        xt_1_recon = xt_1_unkown + masked_next_state

                    if k != 0 and i != 0:
                        xt = self.predictor.q_onestep_sample(xt_1_recon, timesteps - 1)
                    else:
                        xt = xt_1_recon
                        break

        demasked_state = torch.mean(xt, dim=0).cpu().numpy()
        return demasked_state

    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.predictor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'actor_optimizer_state_dict': self.predictor_optimizer.state_dict()}, file_name)

    def save_checkpoint(self, file_name):
        logger.info('Saving Checkpoint model to {}'.format(file_name))
        torch.save({'ema_state_dict': self.ema_model.state_dict()}, file_name)

    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.predictor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.predictor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_checkpoint(self, file_name, device_idx=0):
        if file_name is not None:
            checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.predictor = copy.deepcopy(self.ema_model)
