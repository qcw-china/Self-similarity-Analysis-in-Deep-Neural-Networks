import sys

import numpy as np
import torch


@torch.no_grad()
def approximate_threshold_box_cover(dist: torch.Tensor, scales: torch.Tensor, xz=1.):
    B, N, _ = dist.shape
    _, S = scales.shape
    dist = dist.unsqueeze(1)
    scales = scales.view(B, S, 1, 1)
    D = (dist <= scales).int()

    p = N**2 / torch.sum(D, dim=(2,3))
    Nl = torch.log(p) / np.log(N)
    Nl = torch.pow(Nl, 1 / xz)
    Nl = 1 + (N - 1) * Nl
    return Nl


@torch.no_grad()
def linkage_function(dist: torch.Tensor, scales, bound: torch.Tensor, upper: torch.Tensor):
    B, N, _ = dist.shape
    _, S = scales.shape
    dist = dist.view(B, 1, N, N)
    upper = upper.view(B, 1, 1, 1)
    bound = bound.view(B, 1, 1, 1)
    scales = scales.view(B, S, 1, 1)
    rate1 = (dist - bound) / (scales - bound)
    rate2 = (dist - scales) / (upper - scales)
    pi_2 = torch.pi / 2
    dist = torch.sin(pi_2 * rate1) * (dist <= scales).int() + torch.cos(pi_2 * rate2) * (
            dist > scales).int()
    return dist


def pred_fun(a, b):
    def _(x):
        return (-a / b) * x + a

    return _


def compute_mse_sim_loss(x, sample_num=10, up_r=1.):
    b, n, d = x.shape
    device = x.device

    dist = torch.cdist(x, x, compute_mode='use_mm_for_euclid_dist') / np.sqrt(d)
    dist = torch.relu(dist * (1 - torch.eye(n, device=device)).view(1, n, n).repeat(b, 1, 1))

    TV = torch.max(dist.view(b, n * n), dim=1, keepdim=True)[0].detach()
    TZ = torch.min((dist + 99999 * torch.eye(n, device=device).unsqueeze(0).repeat(b, 1, 1))
                   .view(b, n * n), dim=1, keepdim=True)[0].detach()

    scales = (TV-TZ) * torch.rand(1, sample_num, device=device) + TZ
    links = linkage_function(dist, scales, bound=torch.zeros_like(TV), upper=TV*1.1)
    links = torch.mean(dist.unsqueeze(1) * links, dim=(2, 3))

    # Nl = torch.tensor([threshold_box_cover((dist <= s).cpu().numpy(), id=i) for s in scales], device='cuda')
    Nl = approximate_threshold_box_cover(dist, scales)

    Nl = Nl + links - links.detach()

    logscales = torch.log(1 + scales)
    logNl = torch.log(Nl) / np.log(n)

    pf = pred_fun(up_r, torch.log(1 + TV))
    pf_Nl = pf(logscales)
    loss = torch.nn.functional.mse_loss(pf_Nl, logNl)
    # loss = -torch.log(loss)
    return loss


if __name__ == '__main__':
    x = torch.randn(4, 8, 128).cuda()*0.1
    x = torch.nn.Parameter(x)
    opt = torch.optim.SGD([x],lr=100)

    for i in range(1000):
        loss = compute_mse_sim_loss(x, 10, up_r=1.3)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
