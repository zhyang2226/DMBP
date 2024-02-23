# %% Part 0 import Package
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Part 1 Global function Definition
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(beta_start, beta_end, timesteps)
    return torch.tensor(betas, dtype=dtype)

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def vp_beta_schedule(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)

def self_defined_beta_schedule(timesteps, dtype=torch.float32):
    if timesteps != 100:
        raise NotImplementedError("Self-Defined beta function only supports T=100")
    t = np.arange(1, timesteps + 1)
    a, b, c = 2.1109, 25.06, -2.5446
    betas = np.exp(-b/(t+a)+c)
    return torch.tensor(betas, dtype=dtype)

def self_defined_beta_schedule2(timesteps, dtype=torch.float32):
    if timesteps != 100:
        raise NotImplementedError("Self-Defined beta function only supports T=100")
    t = np.arange(1, timesteps + 1)
    a, b, c = 3.0651, 24.552, -3.1702
    betas = np.exp(-b/(t+a)+c)
    return torch.tensor(betas, dtype=dtype)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class WeightedLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        loss = self._loss(pred, targ)
        weighted_loss = (loss * weights).mean()
        return weighted_loss

class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
}

# %% Part 2 Network Structure
class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model,
                 beta_schedule='vp', beta_mode='all', n_timesteps=20,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = model

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)
        elif beta_schedule == 'self-defined':
            betas = self_defined_beta_schedule(n_timesteps)
        elif beta_schedule == 'self-defined2':
            betas = self_defined_beta_schedule2(n_timesteps)
        else:
            raise ValueError(f"No such beta_schedule exists {beta_schedule}")
        self.beta_mode = beta_mode  # Train partial or all timesteps

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, next_obs_recon, next_obs, timestep):
        posterior_mean = (
                extract(self.posterior_mean_coef1, timestep, next_obs.shape) * next_obs_recon +
                extract(self.posterior_mean_coef2, timestep, next_obs.shape) * next_obs
        )
        posterior_variance = extract(self.posterior_variance, timestep, next_obs.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, timestep, next_obs.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, next_obs, timesteps, action, state):
        next_obs_recon = self.predict_start_from_noise(next_obs, t=timesteps,
                                                       noise=self.model(next_obs, timesteps, action, state))
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(next_obs_recon, next_obs, timesteps)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, next_obs, timesteps, action, state):
        b, *_, device = *next_obs.shape, next_obs.device
        model_mean, _, model_log_variance = self.p_mean_variance(next_obs, timesteps, action, state)
        noise = torch.randn_like(next_obs)
        # no noise when t == 0
        nonzero_mask = (1 - (timesteps == 0).float()).reshape(b, *((1,) * (len(next_obs.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        # return model_mean

    # @torch.no_grad()
    def p_sample_loop(self, noised_next_state, action, state, tstp, shape, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = noised_next_state
        for i in reversed(range(0, tstp)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, action, state)

        return x

    # @torch.no_grad()
    # Core not finished
    def predict(self, noised_next_state, action, state, tstp, *args, **kwargs):
        if len(noised_next_state.shape) ==1:
            batch_size =1
        else:
            batch_size = noised_next_state.shape[0]
        shape = (batch_size, self.state_dim)
        pred_next_state = self.p_sample_loop(noised_next_state, action, state, tstp,
                                             shape, *args, **kwargs)
        return pred_next_state

    # ------------------------------------------ training ------------------------------------------#
    def recover(self, noised_state, t):
        batch_size = noised_state.shape[0]
        timesteps = torch.full((batch_size,), t, device=noised_state.device, dtype=torch.long)
        x_t = noised_state * extract(self.sqrt_alphas_cumprod, timesteps, noised_state.shape)
        return x_t

    def q_sample(self, next_state, t, noise=None):
        if noise is None:
            noise = torch.randn_like(next_state)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, next_state.shape) * next_state +
                extract(self.sqrt_one_minus_alphas_cumprod, t, next_state.shape) * noise
        )

        return sample

    # TODO: Check if this is correct
    def q_onestep_sample(self, next_state, t, noise=None):
        if noise == None:
            noise = torch.randn_like(next_state)
        sample = (
                torch.sqrt(torch.ones_like(next_state) - extract(self.betas, t, next_state.shape)) * next_state +
                torch.sqrt(extract(self.betas, t, next_state.shape)) * noise
        )
        return sample  # x_(t+1) = sqrt(alpha_t) * x_(t) + sqrt(beta_t) * \epsilon

    def q_inverse_sample(self, noised_next_state, t, noise_pred):
        next_state_recon = (noised_next_state - extract(self.sqrt_one_minus_alphas_cumprod, t, noised_next_state.shape)
                            * noise_pred)/extract(self.sqrt_alphas_cumprod, t, noised_next_state.shape)
        return next_state_recon

    def p_losses(self, next_state, action, state_condition, mask, t, weights=1.0):
        noise = torch.randn_like(next_state)
        next_state_noisy = self.q_sample(next_state, t, noise)
        next_state_recon = self.model(next_state_noisy, t, action, state_condition, mask)

        assert noise.shape == next_state_recon.shape
        if self.predict_epsilon:
            loss = self.loss_fn(next_state_recon, noise, weights)
            next_state_reco = self.predict_start_from_noise(next_state, t, next_state_recon)
        else:
            loss = self.loss_fn(next_state_recon, next_state, weights)
            next_state_reco = 0
        return loss, next_state_reco

    def loss(self, next_state, action, pres_state_condition, mask, t=None, weights=1.0, mode=None):
        batch_size = len(action)
        if mode == None:
            mode = self.beta_mode
        if t == None:
            if mode == 'all':
                t = torch.randint(0, self.n_timesteps, (batch_size,), device=action.device).long()
            elif mode == 'partial':
                t = torch.randint(0, int(self.n_timesteps * 0.1), (batch_size,), device=action.device).long()
            else:
                raise NotImplementedError

        return self.p_losses(next_state, action, pres_state_condition, mask, t, weights)

    # def act_q_loss(self, policy, next_state, action, state, weights=1.0, mode="normal"):
    #     batch_size = len(action)
    #     t = torch.randint(0, int(self.n_timesteps * 0.1), (batch_size,), device=policy.device).long()
    #     noise = torch.randn_like(next_state)
    #     next_state_noisy = self.q_sample(next_state, t, noise)
    #     if mode == 'normal' or mode == 'no_act':
    #         pred_noise = self.model(next_state_noisy, t, action, state)
    #     elif mode == 'noise':
    #         pred_noise = self.model(next_state_noisy, t, action, state + (0.01 * torch.rand_like(state) - 0.005))
    #     else:
    #         raise NotImplementedError("Check the Act_Loss Method")
    #
    #     next_state_recon = self.q_inverse_sample(next_state_noisy, t, pred_noise)
    #     if mode == 'no_act':
    #         act_on_real_next_state, q1 = policy.select_action(next_state, eval=True)
    #         act_on_recon_next_state, q2 = policy.select_action(next_state_recon, eval=True)
    #     else:
    #         act_on_real_next_state, q1 = policy.select_action(next_state, eval=False)
    #         act_on_recon_next_state, q2 = policy.select_action(next_state_recon, eval=False)
    #     act_loss = self.loss_fn(act_on_real_next_state, act_on_recon_next_state, weights)
    #     q_loss = self.loss_fn(q1, q2, weights)
    #
    #     return act_loss, q_loss


    # Core not finished
    def forward(self, noised_next_state, action, state, tstp, *args, **kwargs):
        return self.predict(noised_next_state, action, state, tstp, *args, **kwargs)