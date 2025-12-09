from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hdlm.utils import sample_categorical


def sample_t(config, batch_size, eps=None, device=None):
    if eps is None:
        eps = config.model.t_eps

    if config.training.low_discrepancy_sampling:
        t = torch.arange(batch_size, device=device) / batch_size
        t = (t + torch.rand(1, device=device)).fmod(1.0)
    else:
        t = torch.rand(batch_size, device=device)

    t = (1 - 2 * eps) * t + eps
    return t


class NoiseSchedule(nn.Module, ABC):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)

        self.register_buffer("log_prior", self.get_log_prior())

    def get_log_prior(self):
        pr = torch.full((self.vocab_size,), -1e3)
        pr[self.mask_id] = 0
        return pr - pr.logsumexp(-1, keepdim=True)
    
    def sample_prior(self, shape):
        return torch.full(shape, self.mask_id, dtype=torch.long, device=self.log_prior.device)
    
    @abstractmethod
    def logits_at_t(self, features, t):
        raise NotImplementedError
    
    @abstractmethod
    def probs_at_t(self, prs, t):
        raise NotImplementedError

    @abstractmethod
    def sample_zt(self, input_ids, t):
        raise NotImplementedError


class HybridDiffusion(NoiseSchedule):
    def __init__(self, tokenizer, clip_noise=20, gamma=1.0, p_uniform=0.0):
        super().__init__(tokenizer)
        self.clip_noise = clip_noise
        self.p_uniform = max(np.exp(-clip_noise), p_uniform)

        log_B = -np.log1p((1 - self.p_uniform) / self.p_uniform * self.vocab_size / 2)
        mask = torch.zeros(self.vocab_size)
        mask[self.mask_id] = 1
        self.register_buffer("mask", mask, persistent=False)
        self.register_buffer("log_B", torch.tensor(float(log_B)).clip(-clip_noise))
        self.register_buffer("log_gamma", torch.tensor(float(gamma)).log())
    
    def get_alpha_betapi(self, t, eps=1e-4):
        t = t[:, None]
        t1m = 1 - t

        gamma = self.log_gamma.exp()
        # .pow() autocasts to fp32
        t_gamma = t.pow(gamma)
        t1m_gamma = t1m.pow(gamma)

        B = self.log_B.exp()
        c_t = t_gamma.sqrt() * t1m_gamma.sqrt() * B
        C_t = t_gamma + t1m_gamma + (self.vocab_size - 2) * c_t
        # C_t should never be much smaller than 1,
        # but just in case it is, we clip it to avoid numerical instability
        C_t = C_t.clip(eps)

        alpha_t = (t1m_gamma - c_t) / C_t
        beta_pi = (t_gamma * self.mask + c_t * (1 - self.mask)) / C_t
        return alpha_t, beta_pi

    def logits_at_t(self, features, t):
        t = t[..., None, None]
        gamma = self.log_gamma.exp().to(t.dtype)
        log_B = self.log_B.to(t.dtype)
        xi_t = gamma / 2 * torch.log((1 - t) / t).clip(-self.clip_noise, self.clip_noise)
        logits = features.mul(xi_t - log_B)
        logits.add_(log_B)
        logits[..., self.mask_id] = -xi_t.squeeze(-1).expand_as(logits[..., self.mask_id])
        return logits
    
    def probs_at_t(self, prs, t, eps=1e-4):
        orig_dtype = prs.dtype
        t = t[:, None]
        t1m = 1 - t

        gamma = self.log_gamma.exp()
        # .pow() autocasts to fp32
        t_gamma = t.pow(gamma)
        t1m_gamma = t1m.pow(gamma)

        B = self.log_B.exp()
        c_t = t_gamma.sqrt() * t1m_gamma.sqrt() * B
        C_t = t_gamma + t1m_gamma + (self.vocab_size - 2) * c_t
        # C_t should never be much smaller than 1, but just in case it is, we clip it to avoid numerical instability
        C_t = C_t.clip(eps)

        alpha_t = (t1m_gamma - c_t) / C_t

        # beta_pi_hat = (t_gamma * mask + c_t * (1 - mask)) / C_t
        probs = prs.mul(alpha_t.unsqueeze(-1))
        probs.add_((c_t / C_t).unsqueeze(-1))
        probs[..., self.mask_id] = t_gamma / C_t
        probs[..., self.vocab_size:] = 0
        return probs.to(orig_dtype)
    
    def sample_zt(self, input_ids, t):
        x = F.one_hot(input_ids, num_classes=self.vocab_size).to(dtype=t.dtype)
        probs = self.probs_at_t(x, t)
        z_t = sample_categorical(probs)
        return z_t
    

class HierarchicalDiffusion(NoiseSchedule):
    def __init__(self, tokenizer, config, clip_noise=20):
        super().__init__(tokenizer)
        p_uniform = config.model.get('p_uniform', 0.0)
        gamma = config.model.get('gamma', 1.0)
        cluster_size = config.model.cluster_size
        cluster_dict = config.model.cluster_dict_path
        self.clip_noise = clip_noise
        self.p_uniform = max(np.exp(-clip_noise), p_uniform)
        self.p_perturb = config.model.get('p_perturb', 0.0)  # 1 - \xi
        self.num_levels = config.model.get("num_levels", 2)
        self.cluster_size = cluster_size
        self.register_buffer("cluster_dict", torch.load(cluster_dict))
        # assert self.cluster_dict.shape == (self.vocab_size)
        if self.cluster_dict.max() == self.cluster_size - 1:   
            self.cluster_dict += self.vocab_size
        assert self.cluster_dict.max() == self.vocab_size + self.cluster_size - 1 and self.cluster_dict.min() == self.vocab_size

        log_B = -np.log1p((1 - self.p_uniform) / self.p_uniform * self.vocab_size / 2)
        mask = torch.zeros(self.vocab_size + self.cluster_size)
        mask[self.mask_id] = 1
        self.register_buffer("mask", mask, persistent=False)
        self.register_buffer("log_B", torch.tensor(float(log_B)).clip(-clip_noise))
        self.register_buffer("gamma", torch.tensor(float(gamma)))
        self.register_buffer("log_gamma", torch.tensor(float(gamma)).log())
        mask_clusters = torch.zeros(self.vocab_size + self.cluster_size)
        mask_clusters[self.vocab_size:] = 1
        self.register_buffer("mask_clusters", mask_clusters, persistent=False)
        # Precompute cluster_matrix
        # Assuming cluster_dict_map maps vocab indices to global cluster indices (vocab_size -1 to vocab_size + cluster_size -1)
        # We need local cluster indices (0 to cluster_size-1) for one_hot
        cluster_matrix = F.one_hot(self.cluster_dict - self.vocab_size, num_classes=self.cluster_size).float() # Store as float
        self.register_buffer("cluster_matrix", cluster_matrix, persistent=False)
    
    def get_alpha_betapi(self, t, eps=1e-4):
        # we currently do not add uniform noise to both levels
        t = t[:, None]
        t1m = 1 - t

        gamma = self.log_gamma.exp()
        # .pow() autocasts to fp32
        t_gamma = t.pow(gamma)
        t1m_gamma = t1m.pow(gamma)

        alpha_t = t1m_gamma

        if gamma == 1.0:
            c_t = -t1m * torch.log(t1m)
            m_t = t + t1m * torch.log(t1m)
        else:
            c_t = gamma * t1m * (1 - t1m.pow(gamma - 1)) / (gamma - 1)
            m_t = 1 - c_t - alpha_t
        beta_pi_mask = self.mask * m_t
        beta_pi_clusters = self.mask_clusters * c_t
        
        return alpha_t, beta_pi_clusters, beta_pi_mask

    def logits_at_t(self, features, t):
        return self.probs_at_t(features, t)
    
    def probs_at_t(self, prs, t, eps=1e-4):
        orig_dtype = prs.dtype
        with torch.no_grad():
            alpha_t, beta_pi_clusters, beta_pi_mask = self.get_alpha_betapi(t)

        prs_clusters = torch.einsum('b s v, v c -> b s c', prs[..., :self.vocab_size-1], self.cluster_matrix)
        
        if self.p_perturb > 0:
            prs_clusters.mul_(1 - self.p_perturb * (1 + 1 / self.cluster_size))
            prs_clusters.add_(self.p_perturb / self.cluster_size)
        prs.mul_(alpha_t.to(orig_dtype).unsqueeze(-1))
        prs[..., self.vocab_size:self.vocab_size + self.cluster_size].add_(prs_clusters.to(orig_dtype) * beta_pi_clusters[:, -1].to(orig_dtype).unsqueeze(-1).unsqueeze(-1))
        prs[..., self.mask_id].add_(beta_pi_mask[:, self.mask_id].to(orig_dtype).unsqueeze(-1))
        return prs.to(orig_dtype)
    
    # forward process
    def sample_zt(self, input_ids, t):
        x = F.one_hot(input_ids, num_classes=self.vocab_size + self.cluster_size).to(dtype=t.dtype)
        c = self.cluster_dict[input_ids]
        c = F.one_hot(c, num_classes=self.vocab_size + self.cluster_size).to(dtype=t.dtype)
        if self.p_perturb > 0:
            c = c * (1 - self.p_perturb * (1 + 1 / self.cluster_size))
            c[..., self.vocab_size:] += self.p_perturb / self.cluster_size
            c = c.to(dtype=t.dtype)
        alpha_t, beta_pi_clusters, beta_pi_mask = self.get_alpha_betapi(t)
        probs = x * alpha_t.unsqueeze(-1) + c * beta_pi_clusters.unsqueeze(1).expand_as(x) + beta_pi_mask.unsqueeze(1).expand_as(x)

        z_t = sample_categorical(probs)
        return z_t
    

class MaskedDiffusion(NoiseSchedule):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # required to be able to interchangeably mix our/mdlm schedule/loss
        self.register_buffer("log_gamma", torch.tensor(0.0))
        self.register_buffer("log_B", torch.tensor(-20.0))

    def get_sigmas(self, t, eps=1e-4):
        dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
        sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
        return dsigma, sigma

    def logits_at_t(self, features, t):
        _, sigma = self.get_sigmas(t)
        move_chance = 1 - torch.exp(-sigma)
        log_1m_move_chance = -sigma
        logits = (features + 1e-8).clip(1e-8).log().log_softmax(-1) + log_1m_move_chance[..., None, None]
        logits[:, :, self.mask_id] = move_chance.log().clip(-1e6)[..., None]
        return logits
    
    def probs_at_t(self, prs, t):
        _, sigma = self.get_sigmas(t)
        alpha_t = torch.exp(-sigma)
        probs = alpha_t[..., None, None] * prs
        probs[..., self.mask_id] = 1 - alpha_t.unsqueeze(-1)
        return probs

    def sample_zt(self, input_ids, t):
        _, sigma = self.get_sigmas(t)
        move_chance = 1 - torch.exp(-sigma)
        is_masked = torch.rand_like(input_ids.float()) < move_chance.unsqueeze(-1)
        z_t = torch.where(is_masked, self.mask_id, input_ids)
        return z_t


def get_noise_schedule(config, tokenizer):
    if config.model.type == "autoregressive":
        return None
    elif config.model.diffusion_process == "gidd":
        noise_schedule = HybridDiffusion(tokenizer, p_uniform=config.model.p_uniform)
    elif config.model.diffusion_process == "hdlm":
        noise_schedule = HierarchicalDiffusion(tokenizer, config=config)
    elif config.model.diffusion_process == "mdlm":
        noise_schedule = MaskedDiffusion(tokenizer)
    else:
        raise ValueError(f"Unknown diffusion process: {config.model.diffusion_process}")

    return noise_schedule