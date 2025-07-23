import os
# 设置环境变量以避免OpenMP重复初始化错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import numpy as np
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from safetensors import safe_open
import argparse
from tqdm import tqdm
import powerlaw
from scipy import stats
from sklearn.decomposition import PCA
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置模型下载缓存目录
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_models')
os.environ['TORCH_HOME'] = cache_dir

def load_model_from_safetensors(model_name, model_type, pretrained=True):
    """从本地 safetensors 文件加载预训练模型"""
    # 创建模型实例（不带预训练权重）
    model = timm.create_model(model_name, pretrained=False)
    
    if pretrained:
        # 构建 safetensors 文件路径
        weights_path = os.path.join(cache_dir, f"{model_name}.safetensors")
        
        if os.path.exists(weights_path):
            print(f"从本地加载预训练权重: {weights_path}")
            try:
                # 从 safetensors 加载权重
                with safe_open(weights_path, framework="pt") as f:
                    state_dict = {key: f.get_tensor(key) for key in f.keys()}
                
                # 处理分类头不匹配的问题
                if 'head.weight' in state_dict and state_dict['head.weight'].shape[0] != model.head.weight.shape[0]:
                    print(f"分类头大小不匹配，跳过加载分类头参数")
                    # 删除分类头参数
                    state_dict.pop('head.weight', None)
                    state_dict.pop('head.bias', None)
                
                # 加载权重到模型
                model.load_state_dict(state_dict, strict=False)
                print(f"成功加载预训练权重")
            except Exception as e:
                print(f"加载预训练权重失败: {str(e)}")
        else:
            print(f"预训练权重文件不存在: {weights_path}")
            print(f"请先运行 python download_models.py 下载预训练模型")
    
    return model

def prepare_data(dataset_name='CIFAR10', batch_size=512):
    """准备数据加载器"""
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize(256),  # 先调整到稍大的尺寸
        transforms.CenterCrop(224),  # 然后中心裁剪到统一大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 根据数据集名称加载不同数据集
    if dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(
            root='data-2',
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_name == 'CIFAR100':
        dataset = datasets.CIFAR100(
            root='data-1',
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_name == 'IMNETTE':
        data_dir = Path('data-3') / 'imagenette2-320' / 'train'
        if not data_dir.exists():
            raise FileNotFoundError(f"IMNETTE数据集未找到: {data_dir}")
        dataset = datasets.ImageFolder(
            root=str(data_dir),
            transform=transform
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}，可用选项: CIFAR10, CIFAR100, IMNETTE")
    
    print(f"已加载 {dataset_name} 数据集，共 {len(dataset)} 个样本")
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    return data_loader

def extract_features(model, data_loader, device, model_type, max_batches=None):
    """提取模型中间层特征"""
    features = {}
    
    # 定义钩子函数
    def get_feature(name):
        def hook(_, __, output):
            if name not in features:
                features[name] = []
            # 确保输出是CPU张量并克隆
            if isinstance(output, torch.Tensor):
                features[name].append(output.clone().detach().cpu())
            else:
                # 处理输出是元组的情况（如某些注意力机制层）
                features[name].append(output[0].clone().detach().cpu() if isinstance(output, tuple) else output.clone().detach().cpu())
        return hook
    
    # 注册钩子
    hooks = []
    if model_type == 'vit':
        for n, m in model.named_modules():
            nl = n.split('.')
            if len(nl) == 2 and 'blocks' in nl[0]:
                hooks.append(m.register_forward_hook(get_feature(n)))
    elif model_type == 'resnet':
        for n, m in model.named_modules():
            nl = n.split('.')
            if len(nl) == 1 and 'layer' in nl[0]:
                hooks.append(m.register_forward_hook(get_feature(n)))
    elif model_type == 'resmlp':
        for n, m in model.named_modules():
            nl = n.split('.')
            if len(nl) == 2 and 'blocks' in nl[0]:
                hooks.append(m.register_forward_hook(get_feature(n)))
    elif model_type == 'poolformer':
        for i in range(0, 4):
            for idx, block in enumerate(model.stages[i].blocks):
                hooks.append(block.register_forward_hook(get_feature(f'stages.{i}.blocks.{idx}')))
    elif model_type == 'vgg':
        for n, m in model.named_modules():
            nl = n.split('.')
            if len(nl) == 2 and nl[1] in ['1', '4', '9', '14', '19']:
                hooks.append(m.register_forward_hook(get_feature(n)))
    elif model_type == 'mixer':
        for n, m in model.named_modules():
            nl = n.split('.')
            if len(nl) == 2 and 'blocks' in nl[0]:
                hooks.append(m.register_forward_hook(get_feature(n)))
    elif model_type == 'pvt_v2':
        # 为PVT-v2模型的stages注册钩子
        if hasattr(model, 'stages'):
            print("找到pvt_v2的stages结构")
            # 每个stage整体添加一个钩子
            for stage_idx in range(4):  # PVT-v2通常有4个stage
                if hasattr(model.stages, str(stage_idx)):
                    stage = getattr(model.stages, str(stage_idx))
                    hooks.append(stage.register_forward_hook(get_feature(f'stage_{stage_idx}')))
            
            # 为每个stage的最后一个block添加钩子，提取每个阶段的最终特征
            for stage_idx in range(4):
                if hasattr(model.stages, str(stage_idx)) and hasattr(getattr(model.stages, str(stage_idx)), 'blocks'):
                    stage = getattr(model.stages, str(stage_idx))
                    if len(stage.blocks) > 0:
                        last_block_idx = len(stage.blocks) - 1
                        hooks.append(stage.blocks[last_block_idx].register_forward_hook(
                            get_feature(f'stage_{stage_idx}_final_block')))
        else:
            # 如果不是上面的结构，尝试通用方法
            for name, module in model.named_modules():
                if 'stages' in name and 'blocks' in name and name.count('.') <= 2:
                    # 找到stages.X.blocks这种模式的模块
                    hooks.append(module.register_forward_hook(get_feature(name)))
    
    # 提取特征
    model.eval()
    try:
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(tqdm(data_loader, desc="提取特征")):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                x = x.to(device)
                model(x)
                del x
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    finally:
        for hook in hooks:
            hook.remove()
    
    # 合并所有批次的特征
    for name in features:
        features[name] = torch.cat(features[name], dim=0)
    
    print(f"提取的特征层: {list(features.keys())}")
    for name, tensor in features.items():
        print(f"层 {name} 特征形状: {tensor.shape}")
    
    return features

def process_features(features):
    """处理提取的特征以便进行分析"""
    processed_features = {}
    
    for name, feature in features.items():
        # 根据特征的实际形状进行处理
        try:
            if len(feature.shape) == 4:  # (B, C, H, W) 卷积特征形式
                B, C, H, W = feature.shape
                f = feature.reshape(B, C, -1)  # (B, C, H*W)
                f = f.permute(0, 2, 1)  # (B, H*W, C)
                f = f.reshape(B, -1)  # (B, H*W*C)
            elif len(feature.shape) == 3:  # (B, L, C) Transformer特征形式
                B, L, C = feature.shape
                f = feature.reshape(B, -1)  # (B, L*C)
            elif len(feature.shape) == 2:  # (B, D) 已展平特征
                f = feature
            else:
                B = feature.shape[0]  # 批次大小总是第一维
                f = feature.reshape(B, -1)
                
            processed_features[name] = f
        except Exception as e:
            print(f"  处理特征 {name} 时出错: {str(e)}")
            continue
    
    return processed_features

def calculate_sparsity(features):
    """
    计算特征的稀疏性: s_l = ||F^(l)||_0 / (N * d_l)
    
    Args:
        features: 特征字典，键为层名称，值为特征张量
        
    Returns:
        sparsity_dict: 稀疏性字典，键为层名称，值为稀疏性度量
    """
    sparsity_dict = {}
    
    for name, feature in features.items():
        # 计算非零元素比例
        non_zero = torch.count_nonzero(feature)
        total_elements = feature.numel()
        sparsity = float(non_zero) / total_elements
        sparsity_dict[name] = sparsity
    
    return sparsity_dict

def fit_powerlaw(data, name, save_dir=None):
    """
    使用powerlaw包拟合数据到幂律分布
    
    Args:
        data: 要拟合的数据
        name: 数据的名称（用于绘图标签）
        save_dir: 保存图表的目录
        
    Returns:
        gamma: 幂律指数
        ks_stat: KS统计量
        p_value: KS检验p值
    """
    # 确保数据是正的
    data_abs = np.abs(data.flatten())
    # 移除零和极小值
    data_abs = data_abs[data_abs > 1e-10]
    
    if len(data_abs) < 50:
        print(f"警告: {name} 的有效数据点太少，无法进行可靠的幂律拟合。")
        return None, None, None
    
    try:
        # 拟合幂律分布
        fit = powerlaw.Fit(data_abs, verbose=False)
        
        # 获取拟合参数
        gamma = fit.alpha
        
        # 执行KS检验，比较拟合的分布和实际数据
        # 使用powerlaw包的内置方法获取KS统计量和p值，这比kstest更适合幂律分布
        D = fit.power_law.KS()  # KS统计量
        
        # 获取p值 - 这里我们使用powerlaw比较不同分布的方法
        # 比较幂律分布和指数分布，p值大于0.1表示幂律分布拟合更好
        R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
        
        # 如果需要保存图表
        if save_dir:
            plt.figure(figsize=(10, 6))
            fit.plot_pdf(linewidth=2, label=f'Data PDF')
            fit.power_law.plot_pdf(linestyle='--', color='r', label=f'Power-law Fit (gamma={gamma:.3f})')
            
            # 格式化p值，避免显示为0.0000
            p_display = f"<0.0001" if p < 0.0001 else f"{p:.4f}"
            plt.title(f'{name} Feature Value Distribution & Power-law Fit\nKS Statistic={D:.4f}, p-value={p_display}')
            plt.xlabel('Feature Value |λ|')
            plt.ylabel('Probability Density P(|λ|)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, f"{name}_powerlaw_fit.png"), dpi=300)
            plt.close()
        
        return gamma, D, p
    except Exception as e:
        print(f"拟合 {name} 的幂律分布失败: {str(e)}")
        return None, None, None

def analyze_feature_distribution(features, save_dir=None):
    """
    分析特征分布，计算稀疏性和拟合幂律分布
    
    Args:
        features: 特征字典，键为层名称，值为特征张量
        save_dir: 保存结果的目录
        
    Returns:
        results: 分析结果字典
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 计算稀疏性
    sparsity_dict = calculate_sparsity(features)
    
    # 排序层名称以便于分析
    def get_layer_number(name):
        parts = name.split('.')
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return float('inf')
        return float('inf')
    
    layer_names = sorted(features.keys(), key=get_layer_number)
    
    # 拟合幂律分布
    power_law_results = {}
    valid_gammas = []
    
    for name in tqdm(layer_names, desc="拟合幂律分布"):
        feature = features[name]
        
        # 获取特征值（使用PCA以减少计算量）
        if feature.shape[1] > 1000:
            # 如果特征维度太大，先用PCA降维
            pca = PCA(n_components=min(1000, feature.shape[0]//2))
            feature_reduced = pca.fit_transform(feature.numpy())
            eigenvalues = np.linalg.eigvalsh(np.cov(feature_reduced.T))
        else:
            # 直接计算协方差矩阵的特征值
            eigenvalues = np.linalg.eigvalsh(np.cov(feature.numpy().T))
        
        # 拟合幂律分布
        gamma, ks_stat, p_value = fit_powerlaw(eigenvalues, name, save_dir)
        
        if gamma is not None:
            power_law_results[name] = {
                'gamma': gamma,
                'ks_statistic': ks_stat,
                'p_value': p_value
            }
            valid_gammas.append(gamma)
    
    # 计算γ的统计信息
    gamma_mean = np.mean(valid_gammas) if valid_gammas else None
    gamma_std = np.std(valid_gammas) if valid_gammas else None
    
    # 判断是否存在统计自相似性
    statistical_self_similarity = gamma_std < 0.2 if gamma_std is not None else False
    
    # 生成结果
    results = {
        'sparsity': sparsity_dict,
        'power_law': power_law_results,
        'gamma_mean': gamma_mean,
        'gamma_std': gamma_std,
        'statistical_self_similarity': statistical_self_similarity
    }
    
    # 保存结果
    if save_dir:
        # 保存稀疏性结果
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(layer_names)), [sparsity_dict[name] for name in layer_names])
        plt.xlabel('Layer Index')
        plt.ylabel('Sparsity (Ratio of Non-zero Elements)')
        plt.title('Feature Sparsity Analysis Across Layers')
        plt.xticks(range(len(layer_names)), [name.split('.')[-1] if '.' in name else name for name in layer_names], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "feature_sparsity.png"), dpi=300)
        plt.close()
        
        # 保存γ值分布
        if valid_gammas:
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(valid_gammas)), valid_gammas, marker='o', linestyle='-')
            if gamma_mean is not None:
                plt.axhline(y=float(gamma_mean), color='r', linestyle='--', label=f'Mean: {gamma_mean:.3f}')
                if gamma_std is not None:
                    plt.fill_between(
                        range(len(valid_gammas)), 
                        [gamma_mean - gamma_std] * len(valid_gammas), 
                        [gamma_mean + gamma_std] * len(valid_gammas), 
                        alpha=0.2, color='r', label=f'Std Dev: {gamma_std:.3f}'
                    )
            plt.xlabel('Layer Index')
            plt.ylabel('Power-law Exponent gamma')
            if gamma_std is not None:
                std_text = f"{gamma_std:.3f}"
            else:
                std_text = "N/A"
            plt.title(f'Power-law Exponents Across Layers (gamma Std Dev: {std_text}, Statistical Self-Similarity: {"Present" if statistical_self_similarity else "Absent"})')
            plt.xticks(range(len(valid_gammas)), [name.split('.')[-1] if '.' in name else name for name in layer_names[:len(valid_gammas)]], rotation=90)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "power_law_exponents.png"), dpi=300)
            plt.close()
        
        # 保存原始数据
        np.savez(
            os.path.join(save_dir, "feature_distribution_results.npz"),
            sparsity=np.array(list(sparsity_dict.values())),
            power_law_gammas=np.array([results['power_law'][name]['gamma'] for name in results['power_law'] if name in results['power_law']]) if valid_gammas else np.array([]),
            gamma_mean=np.array([gamma_mean]) if gamma_mean is not None else np.array([0.0]),
            gamma_std=np.array([gamma_std]) if gamma_std is not None else np.array([0.0]),
            statistical_self_similarity=np.array([statistical_self_similarity])
        )
        
        # 保存文本报告
        with open(os.path.join(save_dir, "feature_distribution_report.txt"), 'w') as f:
            f.write("Feature Distribution Analysis Report\n")
            f.write("=================================\n\n")
            
            f.write("Sparsity Analysis:\n")
            for name in layer_names:
                f.write(f"  {name}: {sparsity_dict[name]:.4f}\n")
            f.write("\n")
            
            f.write("Power-law Distribution Fitting:\n")
            for name in layer_names:
                if name in power_law_results:
                    gamma_val = power_law_results[name]['gamma']
                    ks_stat = power_law_results[name]['ks_statistic']
                    p_val = power_law_results[name]['p_value']
                    # 格式化p值，避免显示为0.0000
                    if p_val < 0.0001:
                        p_val_str = "<0.0001"
                    else:
                        p_val_str = f"{p_val:.4f}"
                    f.write(f"  {name}: gamma={gamma_val:.4f}, KS Statistic={ks_stat:.4f}, p-value={p_val_str}\n")
                else:
                    f.write(f"  {name}: Fitting failed\n")
            f.write("\n")
            
            f.write(f"gamma Mean: {gamma_mean:.4f}\n")
            f.write(f"gamma Std Dev: {gamma_std:.4f}\n")
            f.write(f"Conclusion: Statistical self-similarity is {'present' if statistical_self_similarity else 'absent'} (gamma Std Dev is {'<' if statistical_self_similarity else '>='} 0.2)\n")
    
    return results

def save_comparison_summary(model_name, dataset_name, model_status, gamma_mean, gamma_std, is_self_similar):
    """
    Save model comparison results to a summary file
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        model_status: 'pretrained' or 'untrained'
        gamma_mean: Mean value of gamma
        gamma_std: Standard deviation of gamma
        is_self_similar: Whether statistical self-similarity is present
    """
    summary_file = "model_gamma_comparison.csv"
    
    # Create header if file doesn't exist
    if not os.path.exists(summary_file):
        with open(summary_file, 'w') as f:
            f.write("Model,Dataset,Status,GammaMean,GammaStdDev,SelfSimilarity\n")
    
    # Append results
    with open(summary_file, 'a') as f:
        self_sim_text = "Yes" if is_self_similar else "No"
        gamma_mean_str = f"{gamma_mean:.4f}" if gamma_mean is not None else "N/A"
        gamma_std_str = f"{gamma_std:.4f}" if gamma_std is not None else "N/A"
        f.write(f"{model_name},{dataset_name},{model_status},{gamma_mean_str},{gamma_std_str},{self_sim_text}\n")
    
    print(f"Comparison data added to {summary_file}")

def main():
    """主函数"""
    # 模型配置
    param_groups = {
        'vit': 'vit_small_patch16_224',
        'resnet': 'resnet34',
        'resmlp': 'resmlp_12_224',
        'poolformer': 'poolformer_s12',
        'vgg': 'vgg11',
        'mixer': 'mixer_b16_224',
        'pvt_v2': 'pvt_v2_b3'
    }
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='分析神经网络特征的分布和统计尺度不变性')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'IMNETTE'],
                        default='CIFAR10', help='数据集名称')
    parser.add_argument('--model', type=str, choices=list(param_groups.keys()),
                        default='resnet', help='模型类型')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--max_batches', type=int, default=10,
                        help='处理的最大批次数')
    parser.add_argument('--pretrained', action='store_true',
                        help='是否使用预训练模型')
    
    args = parser.parse_args()
    
  # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 选择模型
    param_choice = args.model
    model_name = param_groups[param_choice]
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    model_status = "pretrained" if args.pretrained else "untrained"
    save_dir = Path(f"feature_distribution_{args.dataset}")
    save_dir.mkdir(exist_ok=True)
    
    # 准备数据加载器
    print(f"\nPreparing dataset: {args.dataset}...")
    data_loader = prepare_data(dataset_name=args.dataset, batch_size=args.batch_size)
    
    # 加载模型并提取特征
    print(f"\nLoading model: {model_name}, status: {model_status}")
    model = load_model_from_safetensors(model_name, param_choice, pretrained=args.pretrained)
    model.to(device)
    
    print(f"\nExtracting features...")
    features = extract_features(model, data_loader, device, param_choice, max_batches=args.max_batches)
    
    # 释放模型内存
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # 处理特征
    print("\nProcessing features...")
    processed_features = process_features(features)
    
    # 分析特征分布
    print("\nAnalyzing feature distributions...")
    results = analyze_feature_distribution(processed_features, save_dir)
    
    # 输出结果总结
    print("\nAnalysis completed!")
    if results['gamma_mean'] is not None:
        print(f"Power-law exponent gamma mean: {results['gamma_mean']:.4f}")
    if results['gamma_std'] is not None:
        print(f"Power-law exponent gamma std dev: {results['gamma_std']:.4f}")
    print(f"Conclusion: Statistical self-similarity is {'present' if results['statistical_self_similarity'] else 'absent'} (gamma std dev is {'<' if results['statistical_self_similarity'] else '>='} 0.2)")
    print(f"\nResults saved to: {save_dir}")
    
    # 保存到比较汇总文件
    save_comparison_summary(
        model_name=param_choice,
        dataset_name=args.dataset,
        model_status=model_status,
        gamma_mean=results['gamma_mean'],
        gamma_std=results['gamma_std'],
        is_self_similar=results['statistical_self_similarity']
    )

if __name__ == "__main__":
    main()

# 使用示例:
# python analyze_feature_distribution.py --dataset CIFAR100 --batch_size 64 --max_batches 10 --pretrained --model vit 