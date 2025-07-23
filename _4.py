import random
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import timm
import torch
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from pic.imagenet_dataset import imagenet_data_train


@torch.no_grad()
def plot_hierarchical_clusters(distance_matrix, n_thresholds=3, figsize=(15, 8), save_path=None, dpi=300):
    """
    最终修复版层次聚类可视化

    参数：
        distance_matrix: 对称距离矩阵 (numpy数组)
        n_thresholds: 自动生成的阈值数量 (默认3，包含0阈值)
        figsize: 画布尺寸 (默认20x12)
    """
    # 输入验证
    if not isinstance(distance_matrix, np.ndarray):
        raise ValueError("输入必须是numpy数组")
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("必须为方阵")
    if not np.allclose(distance_matrix, distance_matrix.T):
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        # raise ValueError("矩阵必须对称")
    if not isinstance(n_thresholds, int) or n_thresholds < 1:
        raise ValueError("n_thresholds必须是正整数")

    # 生成阈值列表（强制包含0阈值）
    triu_values = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    # 修改后的阈值生成逻辑（使用平方函数实现非线性分布）
    x = np.linspace(0, 1, n_thresholds - 1)  # 生成0-1的均匀序列
    power_law = x ** 2  # 使用平方函数增强低值密度
    percentiles = power_law * 60 + 10  # 映射到10%~70%范围
    auto_thresholds = np.percentile(triu_values, percentiles)

    all_thresholds = np.insert(auto_thresholds, 0, 0)  # 首位置插入0阈值

    # MDS降维
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    coordinates = mds.fit_transform(distance_matrix)

    # 创建动态子图布局
    total_plots = len(all_thresholds)
    cols = min(3, total_plots)  # 每行最多3个子图
    rows = (total_plots + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    if total_plots == 1:
        axs = np.array([[axs]])
    elif rows == 1:
        axs = axs.reshape(1, -1)
    else:
        axs = axs.reshape(rows, cols)

    # 遍历所有阈值进行可视化
    for i, (threshold, ax) in enumerate(zip(all_thresholds, axs.flat)):
        # 生成标签
        if threshold == 0:
            # 每个点独立成簇
            labels = np.arange(distance_matrix.shape[0])
        else:
            cluster = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric='precomputed',
                linkage='average'
            )
            labels = cluster.fit_predict(distance_matrix)

        # 计算聚类结果
        unique_labels, counts = np.unique(labels, return_counts=True)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))  # 统一颜色生成

        # 统一可视化逻辑
        if threshold == 0:
            # 特殊处理：显示所有原始点
            sizes = np.full(len(labels), 70)  # 固定尺寸
            ax.scatter(
                coordinates[:, 0], coordinates[:, 1],
                c=colors[labels],  # 直接使用标签索引颜色
                s=sizes,
                alpha=0.8,
                edgecolors='w',
                linewidth=0.5
            )
        else:
            # 显示簇中心
            centers = np.array([coordinates[labels == label].mean(0) for label in unique_labels])
            sizes = np.sqrt(counts) * 50 + 20  # 动态尺寸
            ax.scatter(
                centers[:, 0], centers[:, 1],
                c=colors,
                s=sizes,
                alpha=0.8
            )

        # 统一标注样式
        title = f'Threshold: {threshold:.3f}\nClusters: {len(unique_labels)}'
        if threshold == 0:
            # title = f"Baseline\nClusters: {len(unique_labels)}"
            title = f"Number of neurons: {len(unique_labels)}"
        ax.set_title(title, fontsize=12, pad=12)
        ax.grid(alpha=0.3)
        ax.set_xlabel('MDS 1', fontsize=9)
        if i % cols == 0:
            ax.set_ylabel('MDS 2', fontsize=9)

    # 隐藏多余的子图
    for j in range(total_plots, rows * cols):
        axs.flat[j].set_visible(False)

    plt.tight_layout(pad=3.0)
    if save_path is not None:
        import os
        from matplotlib import rcParams

        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # 根据扩展名选择保存格式
        file_ext = os.path.splitext(save_path)[1].lower()

        # 矢量图格式设置
        if file_ext in ['.pdf', '.svg', '.eps']:
            plt.savefig(
                save_path,
                format=file_ext[1:],  # 自动提取格式名（如 'pdf'）
                bbox_inches='tight',
                transparent=False,
                metadata={
                    'Creator': 'Hierarchical Clustering Tool',
                    'Producer': 'Matplotlib'
                }
            )
        # 位图格式（如PNG）
        else:
            plt.savefig(
                save_path,
                dpi=dpi,
                bbox_inches='tight',
                transparent=False,
                pil_kwargs={'compress_level': 0} if file_ext == '.png' else None
            )

        print(f"图片已保存至：{os.path.abspath(save_path)}")

    plt.show()


def plot_mds_simplified(distance_matrix1,
                        distance_matrix2,
                        fig_size=(3.5, 3.5),
                        save_path=None,
                        label1="Matrix1",
                        label2="Matrix2"):
    """
    简化版功能：
    - 隐藏坐标轴标签和标题
    - 图例自动适应图片尺寸
    - 其他参数保持默认
    """

    # MDS降维
    mds = MDS(n_components=2, dissimilarity='precomputed',
              random_state=42, metric=True, normalized_stress=False)
    coords1 = mds.fit_transform(distance_matrix1)
    coords2 = mds.fit_transform(distance_matrix2)

    # 创建画布
    plt.figure(figsize=fig_size)

    # 绘制散点图（保持默认样式）
    plt.scatter(coords1[:, 0], coords1[:, 1],
                marker='^', edgecolors='#4169E1',
                facecolor='none', label=label1)
    plt.scatter(coords2[:, 0], coords2[:, 1],
                marker='s', edgecolors='#8B0000',
                facecolor='none', label=label2)

    # 自动调整图例
    plt.legend(loc='best',  # 自动选择最佳位置
               frameon=True,  # 保留边框
               handletextpad=0.3,  # 文本与符号间距
               borderaxespad=0.2,  # 边框内边距
               framealpha=0.9,  # 半透明背景
               fontsize=10.5)  # 设置图例字体大小

    # 设置坐标轴刻度字体大小
    plt.tick_params(axis='both', which='major', labelsize=10.5)

    if save_path:
        import os
        # 提取文件夹路径并创建目录
        dir_path = os.path.dirname(save_path)
        if dir_path:  # 避免空路径（如当前目录）
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    # plt.show()


def plot_mds_simplified_unique(distance_matrix,
                        fig_size=(1.5, 1.5),
                        save_path=None,
                        label1="Matrix1",
                        label2="Matrix2"):
    """
    修改后功能：
    - 输入单个距离矩阵，前50%为label1，后50%为label2
    - 保持原有可视化样式和参数
    """

    # MDS降维
    mds = MDS(n_components=2, dissimilarity='precomputed',
              random_state=42, metric=True, normalized_stress=False)
    coords = mds.fit_transform(distance_matrix)

    # 分割数据点
    split_idx = len(coords) // 2
    coords1 = coords[:split_idx]
    coords2 = coords[split_idx:]

    # 创建画布
    plt.figure(figsize=fig_size)

    # 绘制散点图（保持默认样式）
    plt.scatter(coords1[:, 0], coords1[:, 1],
                marker='^', edgecolors='#4169E1',
                facecolor='none', label=label1)
    plt.scatter(coords2[:, 0], coords2[:, 1],
                marker='s', edgecolors='#8B0000',
                facecolor='none', label=label2)

    # 自动调整图例
    # plt.legend(loc='best',
    #            frameon=True,
    #            handletextpad=0.3,
    #            borderaxespad=0.2,
    #            framealpha=0.9,
    #            fontsize=10.5)

    # 设置坐标轴刻度字体大小
    plt.tick_params(axis='both', which='major', labelsize=10.5)

    # 保存逻辑
    if save_path:
        import os
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)



# 测试用例

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    seed = 0
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)

    param_groups = {
        'vit': 'vit_small_patch16_224',  # 多版本
        'resnet': 'resnet34',  # 多版本
        'resmlp': 'resmlp_12_224',
        'poolformer': 'poolformer_s12',
        'vgg': 'vgg11',
        'pvt': 'pvt_v2_b1',
        # 'mlp': 'mlp_512',
        'mixer': 'mixer_b16_224'
    }
    param_choice = 'resmlp'

    model = timm.create_model(
        param_groups[param_choice],
        pretrained=False,
    )

    model: torch.nn.Module
    for n, m in model.named_modules():
        print(n)

    data = imagenet_data_train()
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=512,
        shuffle=True,  # 启用随机打乱
    )
    features = {}


    def get_feature(name):
        def hook(model, input, output):
            features[name] = output.clone().detach()
        return hook


    for n, m in model.named_modules():
        nl = n.split('.')
        if len(nl) == 1 and 'layer' in nl[0] and param_choice in ['resnet', ]:
            print(f'注册:{n}')
            m.register_forward_hook(get_feature(n))
            features[n] = None

        if len(nl) == 2 and nl[0] == 'stages' and param_choice in ['poolformer', 'pvt']:
            print(f'注册:{n}')
            m.register_forward_hook(get_feature(n))
            features[n] = None

        if len(nl) == 2 and nl[0] == 'blocks' and param_choice in ['resmlp', 'vit', 'mixer']:
            print(f'注册:{n}')
            m.register_forward_hook(get_feature(n))
            features[n] = None

        if len(nl) == 4 and 'blocks' in nl[2] and param_choice in []:
            print(f'注册:{n}')
            m.register_forward_hook(get_feature(n))
            features[n] = None

        if len(nl) == 2 and nl[1] in ['1', '4', '9', '14', '19'] and param_choice in ['vgg']:
            print(f'注册:{n}')
            m.register_forward_hook(get_feature(n))
            features[n] = None

    device = torch.device('cuda')
    model.to(device)


    for k in features.keys():
        # k = list(features.keys())[0]
        wds = k.split('.')[-1]
        if wds not in ['0', '4', '8', '11']:
            continue

        F = []
        dl = []
        sample_index = None
        for pretrained in [False, True]:
            # Load model directly
            if pretrained:
                from safetensors import safe_open

                # 从safetensors加载权重
                weights_path = fr"D:\paper2\save_data\模型预训练参数文件\{param_groups[param_choice]}.safetensors"
                with safe_open(weights_path, framework="pt") as f:
                    state_dict = {key: f.get_tensor(key) for key in f.keys()}
                model.load_state_dict(state_dict)

            for x, l in data_loader:
                x = x.to(device, non_blocking=True)
                model(x)
                f = features[k]
                if param_choice in ['poolformer', 'resnet', 'vgg', 'pvt']:
                    f = torch.mean(f, dim=(2, 3)).transpose(0, 1)
                if param_choice in ['vit', 'resmlp', 'mixer']:
                    # f = f[:, random.randint(0, f.shape[1]-1)].transpose(0, 1)
                    f = torch.mean(f, dim=1).transpose(0, 1)
                if f.shape[0] > 128:
                    if sample_index is None:
                        sample_index = random.sample(range(f.shape[0]), k=128)
                    f = f[sample_index]

                F.append(f)
                # dist = torch.cdist(f, f) / np.sqrt(f.shape[1])
                # dist = dist.cpu().numpy()
                # dl.append(dist)
                # plot_hierarchical_clusters(dist, 1, figsize=(5, 4), save_path=None)
                break
        f = torch.cat(F, dim=0)
        dist = torch.cdist(f, f) / np.sqrt(f.shape[1])
        dist = (dist + dist.T) / 2
        dist = dist.cpu().numpy()

        print(k)
        # plot_mds_simplified(dl[0],
        #                     dl[1],
        #                     save_path=f'file/picture2/change/{param_groups[param_choice]}/{param_groups[param_choice]}_{k}.pdf',
        #                     label1="untrained",
        #                     label2="trained")
        plot_mds_simplified_unique(dist,
                                   save_path=f'file/picture2/change/{param_groups[param_choice]}/{param_groups[param_choice]}_{k}.pdf',
                                   label1="untrained",
                                   label2="trained")
        # sys.exit()
