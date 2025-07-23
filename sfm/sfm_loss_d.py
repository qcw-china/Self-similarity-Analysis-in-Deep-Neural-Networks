import sys

import numpy as np
import torch


@torch.no_grad()
def approximate_threshold_box_cover(adj: torch.Tensor, xz=1.):
    adj.fill_diagonal_(True)
    p = adj.numel() / torch.sum(adj)
    Nl = torch.log(p) / np.log(adj.shape[0])
    Nl = torch.pow(Nl, 1 / xz)
    Nl = 1 + (adj.shape[0] - 1) * Nl
    return Nl


@torch.no_grad()
def linkage_function(adj_norm: torch.Tensor, shd, a=0, b=1):
    rate1 = (adj_norm - a) / (shd - a)
    rate2 = (adj_norm - shd) / (b - shd)
    pi_2 = torch.pi / 2
    adj_norm = torch.sin(pi_2 * rate1) * (adj_norm <= shd).int() + torch.cos(pi_2 * rate2) * (
            adj_norm > shd).int()
    return adj_norm


def pred_fun(a, b):
    def _(x):
        return (-a / b) * x + a

    return _


def compute_mse_sim_loss(x, sample_num=10):
    n, d = x.shape
    device = x.device

    dist = torch.cdist(x, x, compute_mode='use_mm_for_euclid_dist').clone() / np.sqrt(d)

    dist = (1-torch.eye(n, device=device))*dist

    # return torch.norm(dist) / n

    TV = torch.max(dist).item()
    sample = torch.rand(sample_num, device=device)
    # sample = torch.linspace(0.01, 1.-0.01, steps=sample_num, device=device)
    scales = TV * sample

    links = torch.cat(
        [torch.sum(dist * linkage_function(dist, s, a=0, b=TV)).unsqueeze(0) for s in
         scales])
    # Nl = torch.tensor([threshold_box_cover((dist <= s).cpu().numpy(), id=i) for s in scales], device='cuda')
    Nl = torch.tensor([approximate_threshold_box_cover(dist <= s) for s in scales], device='cuda')

    Nl = Nl + links - links.detach()

    logscales = torch.log(1 + scales)
    logNl = torch.log(Nl) / np.log(n)
    
    max = 0.01
    pf = pred_fun(1, np.log(1 + max))
    pf_Nl = pf(logscales)
    loss = torch.nn.functional.mse_loss(pf_Nl, logNl)
    # loss = -torch.log(loss)
    return loss


if __name__ == '__main__':
    x = torch.randn(32, 128)
    compute_mse_sim_loss(x, 10)
