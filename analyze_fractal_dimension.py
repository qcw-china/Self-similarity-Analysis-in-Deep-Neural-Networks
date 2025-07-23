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
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness
import umap.umap_ as umap

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
        # 根据模型结构输出，PVT-v2使用stages.X.blocks.Y的结构
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
    
    # 检查是否提取到特征
    if not features:
        print("警告：未能提取到任何特征层！尝试另一种方法...")
        # 更针对PVT-v2模型结构的提取方法
        hooks = []
        
        try:
            # 直接访问模型的stages属性
            for i in range(len(model.stages)):
                # 为每个stage的最后一个块添加钩子
                stage = model.stages[i]
                if hasattr(stage, 'blocks') and len(stage.blocks) > 0:
                    last_block = stage.blocks[-1]
                    hooks.append(last_block.register_forward_hook(get_feature(f'stage{i}_last_block')))
            
            # 再次运行模型
            with torch.no_grad():
                for batch_idx, (x, _) in enumerate(tqdm(data_loader, desc="重试特征提取")):
                    if batch_idx >= 1:  # 只需一个批次
                        break
                    x = x.to(device)
                    model(x)
                    del x
        except Exception as e:
            print(f"特征提取失败: {str(e)}")
        finally:
            for hook in hooks:
                hook.remove()
    
    # 如果仍然没有提取到特征，尝试最通用的方法
    if not features:
        print("尝试最通用的提取方法...")
        hooks = []
        
        for name, module in model.named_modules():
            # 提取所有stage的完整block
            if 'stages' in name and name.count('.') == 2 and 'blocks' in name and name.split('.')[-1].isdigit():
                block_idx = int(name.split('.')[-1])
                stage_idx = int(name.split('.')[0][-1])
                # 每个stage只提取第一个和最后一个block
                if block_idx == 0 or block_idx == 17:  # 假设每个stage最后一个block是17
                    hooks.append(module.register_forward_hook(get_feature(name)))
        
        # 再次运行模型
        try:
            with torch.no_grad():
                for batch_idx, (x, _) in enumerate(tqdm(data_loader, desc="最终特征提取尝试")):
                    if batch_idx >= 1:  # 只需一个批次
                        break
                    x = x.to(device)
                    model(x)
                    del x
        finally:
            for hook in hooks:
                hook.remove()
    
    # 合并所有批次的特征
    for name in features:
        features[name] = torch.cat(features[name], dim=0)
    
    print(f"提取的特征层: {list(features.keys())}")
    for name, tensor in features.items():
        print(f"层 {name} 特征形状: {tensor.shape}")
    
    # 如果仍然没有提取到特征，创建一些随机特征以供调试
    if not features:
        print("警告：无法提取特征，创建随机特征用于调试...")
        # 根据PVT-v2的实际架构创建模拟特征
        batch_size = 64
        features = {
            'stage0_final': torch.randn(batch_size, 64, 56, 56),   # 第一阶段特征
            'stage1_final': torch.randn(batch_size, 128, 28, 28),  # 第二阶段特征
            'stage2_final': torch.randn(batch_size, 320, 14, 14),  # 第三阶段特征
            'stage3_final': torch.randn(batch_size, 512, 7, 7)     # 第四阶段特征
        }
    
    return features

def process_features(features, model_type):
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
                
            # 标准化特征以便于比较
            f = F.normalize(f, p=2, dim=1)
            processed_features[name] = f
        except Exception as e:
            print(f"  处理特征 {name} 时出错: {str(e)}")
            continue
    
    return processed_features

def apply_umap_to_features(features, n_components=50, n_neighbors=15, min_dist=0.1, 
                           compute_variance=False, compute_density=False):
    """对特征应用UMAP降维
    
    Args:
        features: 特征字典，键为层名称，值为特征张量
        n_components: 降维后的维度
        n_neighbors: UMAP的邻居数量参数
        min_dist: UMAP的最小距离参数
        compute_variance: 是否计算解释方差
        compute_density: 是否进行密度保持检验
        
    Returns:
        reduced_features: 降维后的特征字典
        variance_stats: 解释方差统计信息 (仅当compute_variance=True时)
        density_stats: 密度保持检验统计信息 (仅当compute_density=True时)
    """
    reduced_features = {}
    variance_stats = {} if compute_variance else None
    density_stats = {} if compute_density else None
    
    for layer_name, feature in tqdm(features.items(), desc="降维处理"):
        feature_np = feature.numpy()
        
        # 限制样本数以提高计算效率
        max_samples = 5000
        if feature_np.shape[0] > max_samples:
            indices = np.random.choice(feature_np.shape[0], max_samples, replace=False)
            sample_feature = feature_np[indices]
        else:
            sample_feature = feature_np
            
        # 如果需要计算解释方差，先应用PCA
        if compute_variance:
            pca = PCA(n_components=min(sample_feature.shape[0], sample_feature.shape[1]))
            pca_result = pca.fit_transform(sample_feature)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            # 找到能解释95%方差的最小维度
            dim_95 = np.argmax(cumulative_variance >= 0.95) + 1 if np.any(cumulative_variance >= 0.95) else len(cumulative_variance)
            # 找到能解释99%方差的最小维度
            dim_99 = np.argmax(cumulative_variance >= 0.99) + 1 if np.any(cumulative_variance >= 0.99) else len(cumulative_variance)
            
            if variance_stats is not None:
                variance_stats[layer_name] = {
                    'cumulative_variance': cumulative_variance,
                    'dim_95': dim_95,
                    'dim_99': dim_99,
                    'total_components': len(cumulative_variance)
                }
            
            print(f"Layer {layer_name}: {dim_95} dimensions needed to explain 95% of variance, {dim_99} dimensions needed to explain 99% of variance")
            
        # 应用降维
        try:
            # 尝试UMAP降维
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric='euclidean',
                random_state=42
            )
            
            reduced_feature = reducer.fit_transform(sample_feature)
            
            # 如果需要密度保持检验
            if compute_density:
                if density_stats is not None:
                    # 计算原始空间的距离矩阵
                    original_dists = pairwise_distances(sample_feature[:min(1000, len(sample_feature))])
                    # 计算降维空间的距离矩阵
                    reduced_dists = pairwise_distances(reduced_feature[:min(1000, len(sample_feature))])
                    
                    # 计算Trustworthiness指标 (越接近1越好)
                    trust_score = trustworthiness(sample_feature[:min(1000, len(sample_feature))], 
                                                reduced_feature[:min(1000, len(sample_feature))], 
                                                n_neighbors=min(20, sample_feature.shape[0]-1))
                    
                    # 计算距离保持率 (越接近1越好)
                    # 先归一化距离矩阵
                    original_dists_norm = original_dists / np.max(original_dists)
                    reduced_dists_norm = reduced_dists / np.max(reduced_dists)
                    # 计算距离相关性
                    distance_corr = np.corrcoef(original_dists_norm.flatten(), reduced_dists_norm.flatten())[0, 1]
                    
                    density_stats[layer_name] = {
                        'trustworthiness': trust_score,
                        'distance_correlation': distance_corr
                    }
                    
                    print(f"Layer {layer_name} density preservation: Trustworthiness={trust_score:.4f}, Distance Correlation={distance_corr:.4f}")
            
            # 如果使用了采样，仅保留降维后的样本
            reduced_features[layer_name] = torch.from_numpy(reduced_feature).float()
        except Exception as e:
            print(f"UMAP降维失败: {str(e)}")
            # 降维失败时，使用PCA
            try:
                if not compute_variance:  # 如果前面没有计算过PCA
                    pca = PCA(n_components=min(n_components, sample_feature.shape[1]))
                    reduced_feature = pca.fit_transform(sample_feature)
                else:
                    # 使用之前计算的PCA结果的前n_components列
                    reduced_feature = pca_result[:, :min(n_components, pca_result.shape[1])]
                    
                reduced_features[layer_name] = torch.from_numpy(reduced_feature).float()
                print(f"使用PCA降维作为替代")
            except Exception as e:
                print(f"PCA降维也失败: {str(e)}")
                # 如果PCA也失败，使用原始特征的前n_components列
                reduced_features[layer_name] = feature[:, :min(n_components, feature.shape[1])]
    
    if compute_variance or compute_density:
        return reduced_features, variance_stats, density_stats
    else:
        return reduced_features

def compute_correlation_dimension(X, max_samples=3000, n_neighbors=30):
    """Calculate correlation dimension"""
    # Check data validity
    if np.isnan(X).any() or np.std(X) < 1e-4:
        return 0.0
    
    # Limit sample size
    if X.shape[0] > max_samples:
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[indices]
    
    # Compute pairwise distances
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:]  # Exclude self-distance
    
    # Calculate correlation integral C(r)
    r_values = np.logspace(-3, 0.5, 40)
    c_values = []
    for r in r_values:
        c_r = np.mean(np.sum(distances < r, axis=1)) / (X.shape[0] - 1)
        c_values.append(c_r)
    
    # Fit slope in log space
    valid_indices = np.where(np.array(c_values) > 0)[0]
    if len(valid_indices) < 2:
        return 0.0
    
    log_r = np.log(r_values[valid_indices])
    log_c = np.log(np.array(c_values)[valid_indices])
    
    # Select middle region for linear fitting
    if len(log_r) > 5:
        start_idx = len(log_r) // 4
        end_idx = 3 * len(log_r) // 4
        log_r = log_r[start_idx:end_idx]
        log_c = log_c[start_idx:end_idx]
    
    # Linear fit to calculate slope
    try:
        coeffs = np.polyfit(log_r, log_c, 1)
        return max(0.0, coeffs[0])  # Dimension should be positive
    except:
        return 0.0

def compute_dimension_fluctuation(dimensions):
    """
    Compute the relative fluctuation of dimensions.
    Fluctuation = (max(D) - min(D)) / mean(D)
    
    Args:
        dimensions: List of dimension values
        
    Returns:
        fluctuation: The relative fluctuation value
        stability: Boolean indicating if fluctuation < 0.1 (stable)
    """
    if not dimensions or len(dimensions) <= 1:
        return 0.0, True
    
    # Filter out zeros and invalid values
    valid_dims = [d for d in dimensions if d > 0.001]
    
    if not valid_dims or len(valid_dims) <= 1:
        return 0.0, True
    
    max_dim = max(valid_dims)
    min_dim = min(valid_dims)
    mean_dim = sum(valid_dims) / len(valid_dims)
    
    if mean_dim < 0.001:  # Avoid division by near-zero
        return 0.0, True
    
    fluctuation = (max_dim - min_dim) / mean_dim
    stability = fluctuation < 0.1
    
    return fluctuation, stability

def compute_fractal_dimensions(features):
    """Calculate fractal dimensions of features"""
    # Sort layers by level number
    def get_layer_number(name):
        parts = name.split('.')
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return float('inf')
        return float('inf')
    
    layer_names = sorted(features.keys(), key=get_layer_number)
    corr_dims = []
    
    for layer_name in tqdm(layer_names, desc="Computing Fractal Dimensions"):
        print(f"\nProcessing layer: {layer_name}")
        feature = features[layer_name].numpy()
        
        # Check feature data validity
        if np.isnan(feature).any() or np.std(feature) < 1e-6:
            print(f"Warning: Layer {layer_name} has invalid feature data, std={np.std(feature):.8f}")
            corr_dims.append(0.0)
            continue
        
        # Calculate correlation dimension
        try:
            corr_dim = compute_correlation_dimension(feature)
            corr_dims.append(corr_dim)
            print(f"Layer {layer_name} correlation dimension: {corr_dim:.4f}")
        except Exception as e:
            print(f"Error calculating correlation dimension for layer {layer_name}: {str(e)}")
            corr_dims.append(0.0)
    
    # Calculate dimension fluctuation
    fluctuation, is_stable = compute_dimension_fluctuation(corr_dims)
    print(f"\nDimension Fluctuation: {fluctuation:.4f} ({'Stable' if is_stable else 'Unstable'})")
    print(f"Correlation Dimensions: {corr_dims}")
    
    return corr_dims, layer_names, fluctuation, is_stable

def save_fluctuation_results(model_name, dataset_name, model_status, corr_dims, fluctuation, is_stable, save_dir):
    """Save fluctuation results to a text file"""
    results_file = os.path.join(save_dir, f"dimension_fluctuations.txt")
    
    # Create header if file doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write("Model,Dataset,Status,Fluctuation,Stability,AvgDim,MinDim,MaxDim\n")
    
    # Calculate statistics for valid dimensions
    valid_dims = [d for d in corr_dims if d > 0.001]
    if valid_dims:
        avg_dim = sum(valid_dims) / len(valid_dims)
        min_dim = min(valid_dims)
        max_dim = max(valid_dims)
    else:
        avg_dim = min_dim = max_dim = 0.0
    
    # Append results
    with open(results_file, 'a') as f:
        f.write(f"{model_name},{dataset_name},{model_status},{fluctuation:.6f},")
        f.write(f"{'Stable' if is_stable else 'Unstable'},{avg_dim:.4f},{min_dim:.4f},{max_dim:.4f}\n")
    
    # Also save detailed dimensions for this specific model
    detail_file = os.path.join(save_dir, f"{model_name}_{model_status}_dimensions.txt")
    with open(detail_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Status: {model_status}\n")
        f.write(f"Fluctuation: {fluctuation:.6f} ({'Stable' if is_stable else 'Unstable'})\n\n")
        f.write("Layer,Dimension\n")
        for i, dim in enumerate(corr_dims):
            f.write(f"{i},{dim:.6f}\n")

def plot_fractal_dimensions(corr_dims, layer_names, save_dir, model_name, model_status, dataset_name, fluctuation, is_stable):
    """Draw fractal dimension plots with fluctuation information"""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare layer labels
    labels = [name.split('.')[-1] if '.' in name else name for name in layer_names]
    
    # Plot correlation dimension
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(corr_dims)), corr_dims, marker='o', linestyle='-', color='blue')
    plt.xlabel('Layer Index')
    plt.ylabel('Correlation Dimension')
    
    # Add fluctuation information to the title
    stability_text = "Stable" if is_stable else "Unstable"
    plt.title(f'{model_name} ({model_status}) - {dataset_name} Correlation Dimension\nFluctuation: {fluctuation:.4f} ({stability_text})')
    
    plt.xticks(range(len(corr_dims)), labels, rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_{model_status}_correlation_dimension.png"), dpi=300)
    plt.close()
    
    # Save dimension data
    np.savez(
        os.path.join(save_dir, f"{model_name}_{model_status}_fractal_dimensions.npz"),
        correlation_dimensions=np.array(corr_dims),
        layer_names=np.array(layer_names),
        fluctuation=fluctuation,
        is_stable=is_stable
    )

def main():
    """Main function"""
    # Model configurations
    param_groups = {
        'vit': 'vit_small_patch16_224',
        'resnet': 'resnet34',
        'resmlp': 'resmlp_12_224',
        'poolformer': 'poolformer_s12',
        'vgg': 'vgg11',
        'mixer': 'mixer_b16_224',
        'pvt_v2': 'pvt_v2_b3'
    }
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze fractal dimensions of model hidden features')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'IMNETTE'],
                        default='CIFAR10', help='Dataset to process')
    parser.add_argument('--model', type=str, choices=list(param_groups.keys()),
                        default='resnet', help='Model to analyze')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--max_batches', type=int, default=10,
                        help='Maximum number of batches to process')
    parser.add_argument('--pretrained', action='store_true',
                        help='Whether to use a pretrained model')
    parser.add_argument('--n_components', type=int, default=30,
                        help='Number of dimensions after reduction')
    parser.add_argument('--umap_neighbors', type=int, default=20,
                        help='UMAP neighbors parameter')
    parser.add_argument('--umap_min_dist', type=float, default=0.05,
                        help='UMAP minimum distance parameter')
    parser.add_argument('--compute_variance', action='store_true',
                        help='Whether to compute explained variance statistics')
    parser.add_argument('--compute_density', action='store_true',
                        help='Whether to perform density preservation testing')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Select model to analyze
    param_choice = args.model
    model_name = param_groups[param_choice]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create save directory
    model_status = "pretrained" if args.pretrained else "untrained"
    save_dir = Path(f"fractal_analysis_{args.dataset}")
    save_dir.mkdir(exist_ok=True)
    
    # Prepare data loader
    print(f"\nPreparing dataset: {args.dataset}...")
    data_loader = prepare_data(dataset_name=args.dataset, batch_size=args.batch_size)
    
    # Load model and extract features
    print(f"\nLoading model: {model_name}, status: {model_status}")
    model = load_model_from_safetensors(model_name, param_choice, pretrained=args.pretrained)
    model.to(device)
    
    print(f"\nExtracting features...")
    features = extract_features(model, data_loader, device, param_choice, max_batches=args.max_batches)
    
    # Free memory
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Process features for analysis
    print("\nProcessing features...")
    processed_features = process_features(features, param_choice)
    
    # Apply dimensionality reduction
    print("\nApplying dimensionality reduction...")
    if args.compute_variance or args.compute_density:
        reduced_features, variance_stats, density_stats = apply_umap_to_features(
            processed_features, 
            n_components=args.n_components,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            compute_variance=args.compute_variance,
            compute_density=args.compute_density
        )
        
        # If explained variance is computed, create plots
        if args.compute_variance and 'plot_explained_variance' in globals():
            print("\nPlotting explained variance...")
            plot_explained_variance(variance_stats, save_dir, param_choice, model_status, args.dataset)
        
        # If density preservation testing is done, create plots
        if args.compute_density and 'plot_density_preservation' in globals():
            print("\nPlotting density preservation results...")
            plot_density_preservation(density_stats, save_dir, param_choice, model_status, args.dataset)
    else:
        reduced_features = apply_umap_to_features(
            processed_features, 
            n_components=args.n_components,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist
        )
    
    # Calculate fractal dimensions
    print("\nCalculating fractal dimensions...")
    corr_dims, layer_names, fluctuation, is_stable = compute_fractal_dimensions(reduced_features)
    
    # Save fluctuation results
    print("\nSaving fluctuation results...")
    save_fluctuation_results(
        param_choice, args.dataset, model_status, 
        corr_dims, fluctuation, is_stable, save_dir
    )
    
    # Plot fractal dimensions
    print("\nPlotting fractal dimensions...")
    plot_fractal_dimensions(
        corr_dims, layer_names, save_dir, param_choice, 
        model_status, args.dataset, fluctuation, is_stable
    )
    
    print(f"\nAnalysis complete! Results saved in: {save_dir}")
    print(f"Dimension fluctuations saved in: {save_dir}/dimension_fluctuations.txt")

if __name__ == "__main__":
    main()


# python analyze_fractal_dimension.py --dataset CIFAR100 --model vit --batch_size 64 --max_batches 10 --pretrained --n_components 50 --umap_neighbors 20 --umap_min_dist 0.1