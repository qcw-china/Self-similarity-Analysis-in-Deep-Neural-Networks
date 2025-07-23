import random
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt


def find_max_and_min_tensor(matrix):
    if matrix.size(0) != matrix.size(1):
        raise ValueError("Input tensor must be square (n x n)")
    n = matrix.size(0)
    upper_triangle_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    upper_triangle_elements = matrix[upper_triangle_mask]
    max_value = upper_triangle_elements.max()
    min_value = upper_triangle_elements.min()
    return min_value.item(), max_value.item()


def box_edge(fv: np.ndarray, direct=False):
    n, d = fv.shape
    ind = np.triu_indices(n, k=1)

    # a = time.time()
    if direct:
        E = fv[ind]
    else:
        E = np.linalg.norm(fv[ind[0]] - fv[ind[1]], axis=1, ord=2) / np.sqrt(d)
    aq = E.argsort()

    # 记录方阵
    R = np.zeros((n, n), dtype=int)

    # 类别向量
    C = np.arange(n)
    o=0
    dex = (np.empty(n-1, dtype=int), np.empty(n-1, dtype=int))
    p = n
    # b = time.time()
    for idx, _ in enumerate(aq):
        w = R[C == C[ind[0][_]], :][:, C == C[ind[1][_]]]
        if len(np.nonzero(w)[0]) + 1 == w.size:
            p = p - 1
            mm = C[ind[1][_]]
            C[C == mm] = C[ind[0][_]]
            C[C > mm] -= 1
            dex[0][o] = ind[0][_]
            dex[1][o] = ind[1][_]
            o = o + 1
        R[ind[0][_], ind[1][_]], R[ind[1][_], ind[0][_]] = True, True
    return dex


def get_loss(scales, N_l, poly_params):
    N_l_pred = np.polyval(poly_params, scales)

    # 计算残差
    residuals = N_l - N_l_pred

    # 残差的平方和
    RSS = np.sum(residuals ** 2)

    # 样本数和参数数目
    n = len(N_l)
    p = len(poly_params)

    # 拟合标准误差
    std_error = np.sqrt(RSS / (n - p))

    return std_error


def draw(params, scales, N_l, fun, show=True, xlim=None, ylim=None, prinT=True):
    if prinT:
        print(f'Fitted parameters: {str(params)}')

    # 创建绘图
    plt.figure(figsize=(10, 6))

    # 绘制实际数据点，设置透明度和边框
    plt.scatter(scales, N_l, label='Data', color='blue', s=30, alpha=0.6, edgecolor='black')

    # 绘制拟合曲线
    plt.plot(scales, fun(scales), color='red', linewidth=2, label=f'Fit: {str(params)}')

    # 设置横纵坐标范围
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim([min(scales), max(scales)])

    if ylim is not None:
        plt.ylim(ylim)
    else:
        plt.ylim([min(N_l), max(N_l) * 1.1])

    plt.xlabel(r'$\log(1 + l_{B})$', fontsize=14)
    plt.ylabel(r'$\log\left(\frac{N_{B}}{n}\right)$', fontsize=14)

    plt.title('Box-covering Method', fontsize=16)
    plt.legend(fontsize=12)

    # 设置坐标轴的线性刻度
    plt.xscale('linear')
    plt.yscale('linear')

    # 显示美观的网格
    plt.grid(True, linestyle='--', alpha=0.5)

    # 显示图形
    if show:
        plt.show()

    # 输出估算的斜率
    if prinT:
        print(f'Estimated slope: {-params[0]:.2f}')


@torch.no_grad()
def drawdraw(x, output_dir, show=False, view_01=True, norm=True, direct_dist=False):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    if direct_dist:
        dist = x
    else:
        dist = torch.cdist(x, x, p=2).clone() / np.sqrt(x.shape[1])

    num = x.shape[0]
    mi, ma = find_max_and_min_tensor(dist)
    print(f'所有距离的最小值和最大值：{mi, ma}')

    # 记录开始时间
    scales_index = box_edge(x.cpu().numpy(), direct=direct_dist)
    scales = dist[scales_index]

    N_l = torch.arange(num - 1, 0, -1)
    scales, N_l = scales.cpu().numpy(), N_l.numpy()

    scales = np.log(1 + scales)
    N_l = np.log(N_l)

    cut_index = 0
    print(flush=True, end='')
    print(f'cut_index:{cut_index}')

    if norm:
        scales = scales / max(scales)
        N_l = N_l / np.log(num)

    scales, N_l = scales[cut_index:], N_l[cut_index:]

    params = np.polyfit(scales, N_l, 1)
    _func = np.poly1d(params)

    loss = get_loss(scales, N_l, params)
    params = (round(params[0], 2), round(params[1], 2), round(loss, 4))
    draw(params, scales, N_l, _func,
         show=show, xlim=(0, 1) if view_01 else None, ylim=(0, 1) if view_01 else None, prinT=False)

    output_path = os.path.join(output_dir, "NL_L.png")

    plt.savefig(output_path, dpi=300)
    print(f"图片已保存到: {output_path}")


if __name__ == '__main__':

    seed = 652
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # p = "E:\DL_save/one/pretrained.pth"
    x = torch.randn(256, 16).cuda()
    dist = torch.cdist(x, x, p=2).clone() / np.sqrt(x.shape[1])
    drawdraw(dist, 'D:\Pycharm_Projects\pythonProject1', show=False, view_01=True, norm=True, direct_dist=True)
