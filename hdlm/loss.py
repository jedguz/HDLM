from abc import abstractmethod, ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(torch.nn.Module, ABC):
    def __init__(self, config, tokenizer, noise_schedule):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.vocab_size = len(tokenizer)

    @abstractmethod
    def loss(self, logits, input_ids, attention_mask, z_t, t, force_transitting=False, model=None):
        raise NotImplementedError

    def forward(self, logits, input_ids, attention_mask, z_t, t, reduction="tokenmean", force_transitting=False, model=None):
        if model is None:
            loss, elbo, metrics = self.loss(logits, input_ids, attention_mask, z_t, t, force_transitting)
        else:
            loss, elbo, metrics = self.loss(logits, input_ids, attention_mask, z_t, t, force_transitting, model)

        if reduction == "tokenmean":
            loss = (loss * attention_mask).sum() / attention_mask.sum()
            # num_tokens = attention_mask.numel()
            # loss = loss.sum() / num_tokens
        else:  # reduction == "none"
            pass

        return loss, elbo, metrics


class GiddLoss(Loss):
    def __init__(self, config, tokenizer, noise_schedule):
        super().__init__(config, tokenizer, noise_schedule)
        self.mask_id = tokenizer.mask_token_id
        self.loss_weighting = config.loss.loss_weighting
        self.min_loss_weight = config.loss.min_loss_weight
        self.max_loss_weight = config.loss.max_loss_weight
        assert self.max_loss_weight > 0, "max_loss_weight must be positive"

    def get_weights(self, t, z_t, input_ids):
        orig_dtype = t.dtype
        t = t.unsqueeze(-1).to(torch.float64)
        t1m = (1 - t)

        gamma = self.noise_schedule.log_gamma.exp()
        t_gamma = t.pow(gamma)
        t_gamma_prime = gamma * t.pow(gamma - 1)
        t1m_gamma = t1m.pow(gamma)
        t1m_gamma_prime = -t1m.pow(gamma - 1)
        B = self.noise_schedule.log_B.exp()

        c_t = t_gamma.sqrt() * t1m_gamma.sqrt() * B
        c_t_prime = (gamma / 2) * (1 - 2 * t) / (t * t1m) * c_t

        C_t = t_gamma + t1m_gamma + (self.vocab_size - 2) * c_t
        C_t_prime = t_gamma_prime + t1m_gamma_prime + (self.vocab_size - 2) * c_t_prime

        alpha_hat = t1m_gamma - c_t
        alpha_hat_prime = t1m_gamma_prime - c_t_prime

        is_mask = (z_t == self.mask_id).float()
        pi_hat = t_gamma * is_mask + c_t * (1 - is_mask)
        pi_hat_prime = t_gamma_prime * is_mask + c_t_prime * (1 - is_mask)

        alpha = alpha_hat / C_t
        pi_beta = pi_hat / C_t
        alpha_ratio = alpha_hat_prime / alpha_hat - C_t_prime / C_t
        omega_t = (pi_hat_prime - alpha_hat_prime / alpha_hat * pi_hat) / C_t

        is_x = (z_t == input_ids).float()
        # elbo_weights = omega_zt / q(zt | x)
        elbo_weights = (1 - is_x) * (omega_t / pi_beta) + is_x * (omega_t / (alpha + pi_beta))

        loss_weights = elbo_weights.clone()
        if self.loss_weighting == "clip":
            loss_weights.clip_(self.min_loss_weight, self.max_loss_weight)
        elif self.loss_weighting == "dynamic":
            log_snr = -(alpha / (1 - alpha)).log().clip(-20, 20)
            x_scale = B * torch.exp(gamma / 2 * log_snr)
            loss_weights = (1 - is_x) * ((1 - is_mask) + 2 * is_mask) + is_x * x_scale
            loss_weights.clip_(self.min_loss_weight, self.max_loss_weight)

        return alpha_ratio.to(orig_dtype), elbo_weights.to(orig_dtype), loss_weights.to(orig_dtype)

    def loss(self, logits, input_ids, attention_mask, z_t, t, *args, **kwargs):
        dtype = logits.dtype
        alpha_ratio, elbo_weights, ws = self.get_weights(t, z_t, input_ids)

        logits[..., self.mask_id] = torch.finfo(dtype).min

        x = F.one_hot(input_ids, logits.shape[-1]).to(dtype)
        x_hat = logits.softmax(-1).to(dtype)  # prevent automatic upcasting
        log_q_t = self.noise_schedule.probs_at_t(x, t).log_().clip_(min=-1e6)
        log_p_t = self.noise_schedule.probs_at_t(x_hat, t).log_().clip_(min=-1e6)

        kl_loss = F.kl_div(log_p_t, log_q_t, reduction="none", log_target=True).sum(-1)

        log_q_zt = log_q_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        log_p_zt = log_p_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        log_ratio = log_q_zt - log_p_zt

        correction = -log_ratio + log_ratio.exp()
        elbo = elbo_weights * (kl_loss + correction) + alpha_ratio

        loss = ws * (kl_loss + correction)

        metrics = {
            "kl_loss": (ws * kl_loss.detach() * attention_mask).sum() / (ws * attention_mask).sum(),
            "log_ratio": (ws * log_ratio.detach() * attention_mask).sum() / (ws * attention_mask).sum(),
            "ratio_corr": (ws * correction.detach() * attention_mask).sum() / (ws * attention_mask).sum(),
            "elbo": (elbo.detach() * attention_mask).sum() / attention_mask.sum(),
        }

        return loss, elbo, metrics
    

class HDLMLoss(Loss):
    def __init__(self, config, tokenizer, noise_schedule):
        super().__init__(config, tokenizer, noise_schedule)
        self.mask_id = tokenizer.mask_token_id  # 50258
        self.loss_weighting = config.loss.loss_weighting
        self.min_loss_weight = config.loss.min_loss_weight
        self.max_loss_weight = config.loss.max_loss_weight
        assert self.max_loss_weight > 0, "max_loss_weight must be positive"
        cluster_dict = config.model.cluster_dict_path
        cluster_size = config.model.cluster_size
        self.register_buffer("cluster_dict", torch.load(cluster_dict))
        self.vocab_size = len(tokenizer)  # 50259
        self.cluster_size = cluster_size
        # assert self.cluster_dict.shape == (self.vocab_size)
        if self.cluster_dict.max() == self.cluster_size - 1:
            self.cluster_dict += self.vocab_size
        assert self.cluster_dict.max() == self.vocab_size + self.cluster_size - 1 and self.cluster_dict.min() == self.vocab_size
    
        self.mask_only = config.loss.get('mask_only', True)
        self.cluster_loss_weight = config.loss.get('cluster_loss_weight', 1.0)
        self.token_loss_weight = config.loss.get('token_loss_weight', 1.0)
        self.auxiliary_loss_weight = config.loss.get('auxiliary_loss_weight', 0.0)
        self.simplified = config.loss.get('simplified', False)
        self.register_buffer("xi", torch.tensor(float(1 - self.noise_schedule.p_perturb)))
        if self.xi < 1:
            assert self.simplified, "simplified must be True when xi < 1"
        self.force_transitting_within = config.loss.get('force_transitting_within', True)  # set one of them to True during evaluation
        self.force_transitting_between = config.loss.get('force_transitting_between', False)  # set one of them to True during evaluation
        self.force_transitting = self.force_transitting_between or self.force_transitting_within
        assert not (self.force_transitting_within and self.force_transitting_between), "force_transitting_within and force_transitting_between cannot be both True"
        self.hard_training = config.loss.get("hard_training", False)
        self.top_k = config.loss.get("top_k", 0)
        self.original_mdlm = config.loss.get("original_mdlm", False)
    
    @torch.no_grad()
    def get_weights(self, t):
        dtype = t.dtype
        alpha_ratio = (-self.noise_schedule.gamma / (1 - t)).to(dtype)
        
        alpha_t, beta_pi_clusters, beta_pi_mask = self.noise_schedule.get_alpha_betapi(t)
        gamma = self.noise_schedule.gamma
        n_alpha_t_prime = (gamma * (1 - t).pow(gamma - 1)) if gamma != 1 else 1
        beta_pi_mask_prime = (gamma * (1 - (1 - t).pow(gamma - 1))) if gamma != 1 else (-(1 - t).log())
        weights_mask = (beta_pi_mask_prime / (beta_pi_mask[:, self.mask_id]).clip_(min=1e-4)).to(dtype)
        weights_clusters = (n_alpha_t_prime / (beta_pi_clusters[:, -1]).clip_(min=1e-4)).to(dtype)

        loss_weights_mask, loss_weights_clusters = weights_mask.clone(), weights_clusters.clone()
        if self.loss_weighting == "clip" or self.loss_weighting == "dynamic":
            loss_weights_mask.clip_(self.min_loss_weight, self.max_loss_weight)
            loss_weights_clusters.clip_(self.min_loss_weight, self.max_loss_weight)
        elif self.loss_weighting == "empirical":
            loss_weights_mask = t.pow(2) - t + 3
            loss_weights_clusters = 3 - 2 * t

        return alpha_ratio.unsqueeze_(1), weights_mask.unsqueeze_(1), weights_clusters.unsqueeze_(1), loss_weights_mask.unsqueeze_(1), loss_weights_clusters.unsqueeze_(1)
    

    def loss(self, logits, input_ids, attention_mask, z_t, t, force_transitting=False):
        logits, logits_clusters = logits    
        dtype = logits.dtype
        logits[..., self.mask_id] = torch.finfo(dtype).min
        logits[..., self.vocab_size:] = torch.finfo(dtype).min
        logits_clusters[..., :self.vocab_size] = torch.finfo(dtype).min
        logits_clusters[..., self.vocab_size + self.cluster_size:] = torch.finfo(dtype).min

        logits_normalized = logits.clone()

        with torch.no_grad():
            mask_cluster_ids = (self.vocab_size <= z_t) & (z_t < self.vocab_size + self.cluster_size)
        if self.force_transitting:
            with torch.no_grad():
                in_cluster_mask = (self.cluster_dict.unsqueeze(0) == z_t[mask_cluster_ids].unsqueeze(1))
                in_cluster_mask = torch.cat([in_cluster_mask, torch.zeros(in_cluster_mask.shape[0], logits.shape[-1] - in_cluster_mask.shape[1], dtype=in_cluster_mask.dtype, device=in_cluster_mask.device)], dim=-1)
                xi = torch.tensor(self.xi if self.force_transitting_between else 1., dtype=float, device=z_t.device).to(dtype)
            probs = logits[mask_cluster_ids].softmax(dim=-1).to(dtype)
            in_cluster_adjustment = (xi / (probs * in_cluster_mask).sum(dim=-1, keepdim=True).clip(min=1e-30)).to(dtype)
            out_cluster_adjustment = ((1. - xi) / (probs * (~in_cluster_mask)).sum(dim=-1, keepdim=True).clip(min=1e-30)).to(dtype)
            probs = torch.where(in_cluster_mask, probs * in_cluster_adjustment, probs * out_cluster_adjustment).clip_(min=1e-30)
            logits_normalized[mask_cluster_ids] = torch.log(probs).clip_(min=-1e6).to(dtype)

        if self.simplified:  # MDLM-like type of loss， also supports approximate elbo for xi < 1
            assert not self.force_transitting
            with torch.no_grad():
                alpha_ratio, weights_mask, weights_clusters, loss_weights_mask, loss_weights_clusters = self.get_weights(t)
                
                mask_ids = (z_t == self.mask_id)
                in_cluster_mask = (self.cluster_dict[input_ids] == z_t)
                out_cluster_mask = ~in_cluster_mask & mask_cluster_ids
                
            x_hat = logits.softmax(-1).to(dtype)  # prevent automatic upcasting
            # x_hat = x_hat / x_hat.sum(dim=-1, keepdim=True).to(dtype)
            probs_clusters = torch.einsum('b s v, v c -> b s c', x_hat[..., :self.noise_schedule.cluster_matrix.shape[0]], self.noise_schedule.cluster_matrix)
            probs_clusters = torch.gather(probs_clusters, dim=-1, index=(self.cluster_dict[input_ids]-self.vocab_size).unsqueeze(-1)).squeeze(-1)
            probs_clusters = -torch.log((probs_clusters + (1 - self.xi) * (1 - probs_clusters) / (self.xi * (self.cluster_size - 1))).clip_(min=1e-30)).to(dtype)
            loss_clusters = self.xi * probs_clusters
            loss_clusters[~mask_ids] = 0
            
            loss_tokens = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
            loss_in = (loss_tokens - probs_clusters) * in_cluster_mask
            loss_out = loss_tokens * out_cluster_mask
            loss_tokens = loss_in + loss_out
            loss_tokens[~mask_cluster_ids] = 0

            loss = loss_clusters * loss_weights_mask + loss_tokens * loss_weights_clusters       
            elbo = loss_clusters * weights_mask + loss_tokens * weights_clusters
                       
            if self.auxiliary_loss_weight > 0:
                loss_auxiliary = F.cross_entropy(logits_clusters.transpose(1, 2), self.cluster_dict[input_ids], reduction="none")
                if self.mask_only:
                    loss_auxiliary[~mask_ids] = 0
                loss = loss + loss_auxiliary * self.auxiliary_loss_weight
            
        else: # GIDD-like type of loss, strict elbo weights
            with torch.no_grad():
                alpha_ratio, weights_mask, weights_clusters, loss_weights_mask, loss_weights_clusters = self.get_weights(t)
                mask_ids = (z_t == self.mask_id)
                
            x_hat = logits.softmax(-1).to(dtype)  # prevent automatic upcasting
            # x_hat = x_hat / x_hat.sum(dim=-1, keepdim=True).to(dtype)
            probs_clusters = torch.einsum('b s v, v c -> b s c', x_hat[..., :self.noise_schedule.cluster_matrix.shape[0]], self.noise_schedule.cluster_matrix)
            log_probs_clusters = torch.log(probs_clusters.clip_(min=1e-30)).to(dtype)
            loss_clusters_elbo = F.nll_loss(log_probs_clusters.transpose(1, 2), self.cluster_dict[input_ids]-self.vocab_size, reduction="none")
            loss_clusters = loss_clusters_elbo if not self.hard_training else F.cross_entropy(logits.transpose(1, 2),input_ids, reduction="none")
            loss_tokens = F.cross_entropy(logits_normalized.transpose(1, 2), input_ids, reduction="none")
            loss_clusters[~mask_ids] = 0
            loss_clusters_elbo[~mask_ids] = 0
            loss_tokens[~mask_cluster_ids] = 0

            loss = loss_clusters * loss_weights_mask + loss_tokens * loss_weights_clusters        
            elbo = loss_clusters_elbo * weights_mask + loss_tokens * weights_clusters
            
            if self.auxiliary_loss_weight > 0:
                loss_auxiliary = F.cross_entropy(logits_clusters.transpose(1, 2), self.cluster_dict[input_ids], reduction="none")
                if self.mask_only:
                    loss_auxiliary[~mask_ids] = 0
                loss = loss + loss_auxiliary * self.auxiliary_loss_weight

        metrics = {
            "loss_clusters": (loss_clusters.detach() * mask_ids * attention_mask).sum() / (attention_mask).sum(),
            "loss_tokens": (loss_tokens.detach() * mask_cluster_ids * attention_mask).sum() / (attention_mask).sum(),
            "loss": (loss.detach() * attention_mask).sum() / (attention_mask).sum(),
            "elbo": (elbo.detach() * attention_mask).sum() / attention_mask.sum(),
        }
        return loss, elbo, metrics


class MDLMLoss(Loss):
    def __init__(self, config, tokenizer, noise_schedule):
        super().__init__(config, tokenizer, noise_schedule)
        self.mask_id = tokenizer.mask_token_id
        self.neg_infty = -1e6

    def get_sigmas(self, t, eps=1e-4):
        dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
        sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
        return dsigma, sigma

    def loss(self, logits, input_ids, attention_mask, z_t, t, *args, **kwargs):
        dsigma, sigma_t = self.get_sigmas(t)

        logits[..., self.mask_id] = self.neg_infty
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        mask_ids = (z_t == self.mask_id)
        logits[~mask_ids] = self.neg_infty
        logits = torch.where(~mask_ids.unsqueeze(-1).expand_as(logits), logits.scatter(-1, z_t.unsqueeze(-1), 0), logits)

        rec_loss = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")

        weights = dsigma.unsqueeze(-1) / torch.expm1(sigma_t).unsqueeze(-1)
        weights = weights * mask_ids.to(weights.dtype)

        elbo = weights * rec_loss

        metrics = {
            "rec_loss": (weights * rec_loss.detach() * attention_mask).sum() / attention_mask.sum(),
            "elbo": (elbo.detach() * attention_mask).sum() / attention_mask.sum(),
        }

        return elbo, elbo, metrics
    

def get_loss(config, tokenizer, noise_schedule):
    if config.loss.loss_type == "gidd":
        return GiddLoss(config, tokenizer, noise_schedule)
    elif config.loss.loss_type == "mdlm":
        return MDLMLoss(config, tokenizer, noise_schedule)
    elif config.loss.loss_type == "hdlm":
        return HDLMLoss(config, tokenizer, noise_schedule)
    elif config.loss.loss_type == "ar":
        return nn.CrossEntropyLoss(reduction="none")
    else:
        raise ValueError(f"Unknown loss_type: {config.loss.loss_type}")