import random
import sys
import time

import numpy as np
import torch


def linkage_function(dist: torch.Tensor, l, fac, k=1.):
    # num = dist.shape[0]
    dist = dist.unsqueeze(0)
    l = l.unsqueeze(1)
    link = torch.sigmoid((l - dist) * k)
    link_ = (link > 0.5).float()
    link_ = torch.mean(link_, dim=1)
    link = torch.mean(link - 0.5, dim=1)
    # index = torch.topk(-torch.abs(link-0.5), k=link.shape[1]//10, dim=1)[1]
    # link = torch.gather(link, index=index, dim=1)
    # link = torch.sum(link, dim=1)
    # link = torch.logsumexp(link, dim=1)
    link = link_ + (link - link.detach()) * fac
    return link


def pred_fun(a, b, n=1):
    def _(x):
        # 计算 y 的值
        return ((-x + a) / (b - a) + 1) * n

    return _


def get_upper_triangle_indices(n, device):
    # 生成索引矩阵
    row_idx, col_idx = torch.triu_indices(n, n, offset=1, device=device)
    return row_idx, col_idx


def random_permutation(n):
    # 生成0到n的随机排列
    return list(np.random.permutation(n))


def rotate_right(nums, k):
    """
    将列表元素右移k位，循环超出部分到左侧
    :param nums: 输入列表
    :param k: 右移位数
    :return: 旋转后的新列表
    """
    if not nums:  # 处理空列表
        return []

    n = len(nums)
    k = k % n  # 计算有效位移
    return nums[-k:] + nums[:-k]  # 切片重组


def compute_mse_sim_loss(x, sample_num=256, target_rate=1.):
    n, d = x.shape
    device = x.device
    # xm = torch.mean(x, dim=0, keepdim=True)
    # dist = torch.norm(x)
    # return dist

    i1, i2 = get_upper_triangle_indices(n, device)
    dist = torch.norm(x[i1] - x[i2], dim=1, p=2) / np.sqrt(d)
    # print('dist长度', dist.shape[0])

    TV = torch.max(dist).item()
    TZ = torch.min(dist).item()
    # p = 1 - TZ

    pf = pred_fun(np.log(1 + TZ), np.log(1 + TV), n=1)

    # print('小大', TZ, TV)
    sample = torch.linspace(start=0., end=1., steps=sample_num, device=device)
    l = (1 + TZ) * torch.pow((1 + TV) / (1 + TZ), sample) - 1

    fac = n
    # loss1
    Nl = linkage_function(dist, l, fac=fac, k=1)
    NlNl = n - (n - 1) * torch.log(1 + (n - 1) * Nl) / np.log(n)

    logl = torch.log(1 + l)
    pf_logNl = pf(logl)

    logNl = torch.log(NlNl) / np.log(n)

    S = torch.sum(torch.abs(pf_logNl - logNl)) * 2 / sample_num
    # S = torch.sum(torch.relu(pf_logNl - logNl)) * 2 / sample_num

    loss = 0.5 * (S - target_rate) ** 2
    # loss = -torch.log(S)

    now_rate = S.item()

    return loss, TZ, TV, now_rate


if __name__ == '__main__':
    seed = 652
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    n, d = 384, 3
    x = torch.randn(n, d, requires_grad=True)

    # from deksa.stu5 import drawdraw
    # drawdraw(x, show=True, view_01=False, norm=False, stright_dist=False, fre=False, a=True)

    # loss = compute_mse_sim_loss(x, l_min=0.82351, l_max=3.1817080, rate=0.7)
    # print(loss)
    # sys.exit()

    x = torch.nn.Parameter(x)
    opt = torch.optim.SGD([x], lr=10)
    epoch = 10

    loss_record = []

    for i in range(epoch):
        loss = compute_mse_sim_loss(x)
        # lossb = loss[1] + loss[0]
        lossb = loss[0]
        # print(loss)
        # sys.exit()
        opt.zero_grad()
        lossb.backward()

        # print(x.grad)
        # print(torch.std_mean(x.grad))
        # sys.exit()
        opt.step()
        # loss_record.append(round(loss.item(), 2))
        print(f'loss:{loss}')
        print('------------')

    from deksa.stu5 import drawdraw

    drawdraw(x, show=True, view_01=False, norm=False, stright_dist=False, fre=False, a=False)
    # a = time.time()
    # i1,i2 = get_upper_triangle_indices(n)
    # dist = torch.norm(x[i1]-x[i2], dim=1)
    #
    # # dist = torch.cdist(x, x)
    # print(time.time()-a)
