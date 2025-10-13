

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm

from hdlm.diffusion_process import NoiseSchedule
from hdlm.utils import sample_categorical

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits / gumbel_noise


class Sampler(nn.Module):
    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, t_eps: float = 1e-4):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.t_eps = t_eps

    @abstractmethod
    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, num_samples=1, num_denoising_steps=1000, max_length=512, decode=True, show_progress=True, **kwargs):
        max_length = max_length or self.model.config.max_seq_len
        device = next(self.model.parameters()).device

        z_t = self._do_generate(num_samples, num_denoising_steps, max_length, show_progress=show_progress, device=device, **kwargs)

        if decode:
            texts = self.tokenizer.batch_decode(z_t, skip_special_tokens=True)
            return texts
        else:
            return z_t


class GiddSampler(Sampler):
    class DenoisingStep(nn.Module):
        def __init__(self, model, noise_schedule, tokenizer, min_p=0.0):
            super().__init__()
            self.model = model
            self.noise_schedule = noise_schedule
            self.tokenizer = tokenizer
            self.min_p = min_p

        def forward(self, z_t, t, s):
            logits = self.model(z_t, t)
            logits[..., self.tokenizer.mask_token_id] = -1e6
            logits = logits[..., :len(self.tokenizer)]

            # if i > 0:
            q_s = self.noise_schedule.probs_at_t(logits.softmax(-1), s)
            q_t = self.noise_schedule.probs_at_t(logits.softmax(-1), t)
            q_zt = q_t.gather(-1, z_t.unsqueeze(-1))

            alpha_t, beta_pi_t = self.noise_schedule.get_alpha_betapi(t)
            alpha_s, beta_pi_s = self.noise_schedule.get_alpha_betapi(s)

            alpha_ts = alpha_t / alpha_s
            beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

            vz_t = F.one_hot(z_t, num_classes=len(self.tokenizer))
            beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
            q_ts = (alpha_ts * vz_t + beta_pi_ts_at_zt)

            q_st = q_ts * q_s / q_zt
            if self.min_p > 0.0:
                is_small = (q_st < self.min_p).float()
                q_st = (1 - is_small) * q_st
                q_st = q_st / q_st.sum(-1, keepdim=True)
            return sample_categorical(q_st)

    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, t_eps=1e-4, compile_step=True, min_p=0.0):
        super().__init__(model, tokenizer, noise_schedule, t_eps=t_eps)
        self.sampling_step = self.DenoisingStep(model, noise_schedule, tokenizer, min_p=min_p)
        if compile_step:
            self.sampling_step = torch.compile(self.sampling_step)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):

        ts = torch.linspace(0, 1, num_denoising_steps + 1, device=device).unsqueeze(-1)
        ts = (1 - 2 * self.t_eps) * ts + self.t_eps

        # zt = sample_categorical(p_zt)
        z_t = self.noise_schedule.sample_prior((num_samples, max_length)).to(device, non_blocking=True)
        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress, dynamic_ncols=True):
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)]).clone()
        return z_t
    

class HDLMSampler(Sampler):
    class DenoisingStep(nn.Module):
        def __init__(self, model, noise_schedule, tokenizer, min_p=0.0, config=None, cluster_dict=None, cluster_size=None):
            super().__init__()
            self.model = model
            self.noise_schedule = noise_schedule
            self.tokenizer = tokenizer
            self.mask_id = tokenizer.mask_token_id
            self.min_p = min_p
            cluster_dict_tensor = torch.load(cluster_dict)
            if not isinstance(cluster_dict_tensor, torch.Tensor):
                cluster_dict_tensor = torch.tensor(cluster_dict_tensor)
            self.register_buffer("cluster_dict", cluster_dict_tensor)
            
            self.cluster_size = cluster_size
            self.vocab_size = len(tokenizer)
            # assert self.cluster_dict.shape == (self.vocab_size)
            if self.cluster_dict.max() == self.cluster_size - 1:
                self.cluster_dict += self.vocab_size
            assert self.cluster_dict.max() == self.vocab_size + self.cluster_size - 1 and self.cluster_dict.min() == self.vocab_size
            self.register_buffer("xi", torch.tensor(float(1 - self.noise_schedule.p_perturb)))
            self.force_transitting_within = config.loss.get('force_transitting_within', True)  # set one of them to True during evaluation
            self.force_transitting_between = config.loss.get('force_transitting_between', False)  # set one of them to True during evaluation
            assert not (self.force_transitting_within and self.force_transitting_between), "force_transitting_within and force_transitting_between cannot be both True"
            self.parameterization = config.get('sampling_parameterization', 'mdlm')
            self.temperature = config.get('temperature', 1.0)
            self.use_auxiliary = config.model.get('use_auxiliary', False)

        def forward(self, z_t, t, s, last_step=False):
            logits, logits_clusters = self.model(z_t, t)
            logits[..., self.tokenizer.mask_token_id] = -1e6
            logits[..., self.vocab_size:] = -1e6
            logits_clusters[..., :self.vocab_size] = -1e6
            logits_clusters[..., self.vocab_size + self.cluster_size:] = -1e6

            with torch.no_grad():
                mask_cluster_ids = (self.vocab_size <= z_t) & (z_t < self.vocab_size + self.cluster_size)
            
            if (self.force_transitting_within or self.force_transitting_between) and mask_cluster_ids.any():
                with torch.no_grad():
                    cluster_dict_device = self.cluster_dict.to(z_t.device)
                    in_cluster_mask = (cluster_dict_device.unsqueeze(0) == z_t[mask_cluster_ids].unsqueeze(1))
                    in_cluster_mask = torch.cat([in_cluster_mask, torch.zeros(in_cluster_mask.shape[0], logits.shape[-1] - in_cluster_mask.shape[1], dtype=in_cluster_mask.dtype, device=in_cluster_mask.device)], dim=-1)
                    in_cluster_mask_ = in_cluster_mask.float()
                    xi = self.xi.to(z_t.device) if self.force_transitting_between else torch.tensor(1.0, device=z_t.device)
                probs = logits[mask_cluster_ids].softmax(dim=-1).to(logits.dtype)
                in_cluster_adjustment = (xi / (probs * in_cluster_mask_).sum(dim=-1, keepdim=True))
                out_cluster_adjustment = ((1. - xi) / (probs * (1-in_cluster_mask_)).sum(dim=-1, keepdim=True))
                probs = torch.where(in_cluster_mask, probs * in_cluster_adjustment, probs * out_cluster_adjustment).clip_(min=1e-30)
                logits[mask_cluster_ids] = probs.log_().clip_(min=-1e6).to(logits.dtype)

            logits = logits[..., :self.vocab_size + self.cluster_size]
            logits_clusters = logits_clusters[..., :self.vocab_size + self.cluster_size]
            if self.parameterization == "gidd":
                if s == 0 or last_step:
                    copy_flag = ((z_t < self.vocab_size) & (z_t != self.mask_id)).to(z_t.dtype)
                    z_tm1 = logits.argmax(-1)
                    z_tm1 = copy_flag * z_t + (1 - copy_flag) * z_tm1
                    return z_tm1
                q_s = self.noise_schedule.probs_at_t(logits.softmax(-1), s)
                q_t = self.noise_schedule.probs_at_t(logits.softmax(-1), t)
                q_zt = q_t.gather(-1, z_t.unsqueeze(-1))

                alpha_t, beta_pi_t_clusters, beta_pi_t_mask = self.noise_schedule.get_alpha_betapi(t)
                alpha_s, beta_pi_s_clusters, beta_pi_s_mask = self.noise_schedule.get_alpha_betapi(s)

                beta_pi_t = beta_pi_t_mask + beta_pi_t_clusters
                beta_pi_s = beta_pi_s_mask + beta_pi_s_clusters

                alpha_ts = alpha_t / alpha_s
                beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

                vz_t = F.one_hot(z_t, num_classes=logits.shape[-1])
                beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
                q_ts = (alpha_ts * vz_t + beta_pi_ts_at_zt)

                q_st = q_ts * q_s / q_zt
                if self.min_p > 0.0:
                    is_small = (q_st < self.min_p).float()
                    q_st = (1 - is_small) * q_st
                    q_st = q_st / q_st.sum(-1, keepdim=True)
                return sample_categorical(q_st)
            elif self.parameterization == "mdlm":
                if s == 0 or last_step:
                    copy_flag = ((z_t < self.vocab_size) & (z_t != self.mask_id)).to(z_t.dtype)
                    z_tm1 = logits.argmax(-1) 
                    z_tm1 = copy_flag * z_t + (1 - copy_flag) * z_tm1
                    return z_tm1
                else:
                    z_tm1 = z_t.clone()
                    alpha_t, beta_pi_t_clusters, beta_pi_t_mask = self.noise_schedule.get_alpha_betapi(t)
                    alpha_s, beta_pi_s_clusters, beta_pi_s_mask = self.noise_schedule.get_alpha_betapi(s)

                    mask_masks = (z_t == self.mask_id)
                    mask_tokens = ~(mask_masks | mask_cluster_ids)

                    probs = torch.zeros_like(logits, device=logits.device, dtype=logits.dtype)
                    
                    x_hat = logits.softmax(-1).to(logits.dtype)  # prevent automatic upcasting
                    probs_masks_2_clusters = torch.einsum('b s v, v c -> b s c', x_hat[..., :self.noise_schedule.cluster_matrix.shape[0]], self.noise_schedule.cluster_matrix).to(logits.dtype)
                    
                    threshold_mask = (beta_pi_t_mask[0, self.mask_id] - beta_pi_s_mask[0, self.mask_id]) / beta_pi_t_mask[0, self.mask_id]
                    change_mask = torch.rand_like(z_tm1[mask_masks], dtype=torch.float32) < threshold_mask
                    if not self.use_auxiliary:
                        sampled_masks = add_gumbel_noise(probs_masks_2_clusters, temperature=self.temperature).argmax(-1) + self.vocab_size
                    else:
                        sampled_masks = add_gumbel_noise(logits_clusters, temperature=self.temperature).argmax(-1)
                    
                    mask_positions = torch.where(mask_masks)
                    change_positions = torch.where(change_mask)[0]
                    if len(change_positions) > 0:
                        final_batch_idx = mask_positions[0][change_positions]
                        final_seq_idx = mask_positions[1][change_positions]
                        z_tm1[final_batch_idx, final_seq_idx] = sampled_masks[mask_masks][change_mask]
                    
                    if mask_cluster_ids.any():
                        probs_clusters = x_hat[..., :self.vocab_size]
                        threshold_cluster = (alpha_s - alpha_t) / beta_pi_t_clusters[0, self.vocab_size]
                        change_cluster = torch.rand_like(z_t[mask_cluster_ids], dtype=torch.float32) < threshold_cluster.squeeze(0)
                        sampled_clusters = add_gumbel_noise(probs_clusters, temperature=self.temperature).argmax(-1)                        
                        cluster_positions = torch.where(mask_cluster_ids)
                        change_cluster_positions = torch.where(change_cluster)[0]
                        if len(change_cluster_positions) > 0:
                            final_batch_idx = cluster_positions[0][change_cluster_positions]
                            final_seq_idx = cluster_positions[1][change_cluster_positions]
                            z_tm1[final_batch_idx, final_seq_idx] = sampled_clusters[mask_cluster_ids][change_cluster]
                    
                    if self.min_p > 0.0:
                        is_small = (probs < self.min_p).float()
                        probs = (1 - is_small) * probs
                        probs = probs / probs.sum(-1, keepdim=True)

                return z_tm1
            else:
                raise ValueError(f"Unsupported parameterization: {self.parameterization}")

    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, t_eps=1e-4, compile_step=True, min_p=0.0, config=None, cluster_dict=None, cluster_size=None):
        super().__init__(model, tokenizer, noise_schedule, t_eps=t_eps)
        self.sampling_step = self.DenoisingStep(model, noise_schedule, tokenizer, min_p=min_p, config=config, cluster_dict=cluster_dict, cluster_size=cluster_size)
        if compile_step:
            self.sampling_step = torch.compile(self.sampling_step)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None, **kwargs):

        ts = torch.linspace(0, 1, num_denoising_steps + 1, device=device).unsqueeze(-1)
        ts = (1 - 2 * self.t_eps) * ts + self.t_eps

        z_t = self.noise_schedule.sample_prior((num_samples, max_length)).to(device, non_blocking=True)
        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress, dynamic_ncols=True):  
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)], last_step=(i == 0), **kwargs).clone()
        return z_t


class MDLMSampler(Sampler):
    class DenoisingStep(nn.Module):
        def __init__(self, model, noise_schedule, mask_id, min_p=0.0):
            super().__init__()
            self.model = model
            self.noise_schedule = noise_schedule
            self.mask_id = mask_id
            self.min_p = min_p

        def get_sigmas(self, t, eps=1e-4):
            dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
            sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
            return dsigma, sigma

        def forward(self, z_t, t, tm1, i=None, eps=1e-4):
            logits = self.model(z_t, t)
            logits[..., self.mask_id] = -1e6

            if i == 0:
                z_tm1 = logits.argmax(-1)
            else:
                _, sigma_t = self.get_sigmas(t, eps=eps)
                _, sigma_tm1 = self.get_sigmas(tm1, eps=eps)

                move_chance_t = 1 - torch.exp(-sigma_t)
                move_chance_tm1 = 1 - torch.exp(-sigma_tm1)
                move_chance_t = move_chance_t[:, None, None]
                move_chance_tm1 = move_chance_tm1[:, None, None]
                probs = logits.softmax(-1) * (move_chance_t - move_chance_tm1)
                probs[:, :, self.mask_id] = move_chance_tm1[:, :, 0]
                probs /= move_chance_t
                if self.min_p > 0.0:
                    is_small = (probs < self.min_p).float()
                    probs = (1 - is_small) * probs
                    probs = probs / probs.sum(-1, keepdim=True)
                z_tm1 = sample_categorical(probs)
                # z_tm1 = torch.distributions.Categorical(probs=probs).sample()
                # z_tm1 = _sample_categorical(probs)

            copy_flag = (z_t != self.mask_id).to(z_t.dtype)
            z_t = copy_flag * z_t + (1 - copy_flag) * z_tm1
            return z_t

    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, t_eps=1e-4, compile_step=True, min_p=0.0):
        super().__init__(model, tokenizer, noise_schedule, t_eps=t_eps)
        self.sampling_step = self.DenoisingStep(model, noise_schedule, tokenizer.mask_token_id, min_p=min_p)
        if compile_step:
            self.sampling_step = torch.compile(self.sampling_step)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):
        z_t = self.noise_schedule.sample_prior((num_samples, max_length)).to(device, non_blocking=True)

        ts = torch.linspace(self.t_eps, 1 - self.t_eps, num_denoising_steps + 1, device=device).unsqueeze(-1)

        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress):
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)], i=i, eps=self.t_eps).clone()

        return z_t


class AutoregressiveSampler(Sampler):
    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, compile_step=True):
        super().__init__(model, tokenizer, noise_schedule)
        if compile_step:
            self.model = torch.compile(model)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):
        bos_token_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id

        input_ids = torch.full((num_samples, max_length), eos_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((num_samples, max_length), dtype=torch.long, device=device)
        input_ids[:, 0] = bos_token_id
        attention_mask[:, 0] = 1

        done = torch.zeros(num_samples, device=device)
        for i in tqdm.trange(1, max_length, desc="Generating samples", disable=not show_progress):
            logits = self.model(input_ids, use_cache=False).logits[:, i-1]
            probs = logits.softmax(-1)
            next_x = (1 - done) * sample_categorical(probs) + done * self.tokenizer.pad_token_id
            input_ids[:, i] = next_x.to(input_ids.dtype)
            done += (1 - done) * (next_x == eos_token_id).to(done.dtype)
            if (done == 1).all():
                break

        return input_ids


def get_sampler(config, model, tokenizer, noise_schedule: NoiseSchedule, compile_step=False, min_p=0.0, new_config=None):
    if config.model.type == "diffusion":
        if config.model.diffusion_process == "gidd":
            return GiddSampler(model, tokenizer, noise_schedule, t_eps=config.model.t_eps, compile_step=compile_step, min_p=min_p)
        elif config.model.diffusion_process == "mdlm":
            return MDLMSampler(model, tokenizer, noise_schedule, t_eps=config.model.t_eps, compile_step=compile_step, min_p=min_p)
        elif config.model.diffusion_process == "hdlm":
            return HDLMSampler(model, tokenizer, noise_schedule, t_eps=config.model.t_eps, compile_step=compile_step, min_p=min_p, 
                               config=new_config if new_config is not None else config, cluster_dict=config.model.cluster_dict_path, cluster_size=config.model.cluster_size)
        else:
            raise ValueError(f"Unsupported forward process: {config.model.diffusion_process}")
    elif config.model.type == "autoregressive":
        return AutoregressiveSampler(model, tokenizer, noise_schedule, compile_step=True)
    else:
        raise ValueError(f"Unsupported model type: {config.model.type}")