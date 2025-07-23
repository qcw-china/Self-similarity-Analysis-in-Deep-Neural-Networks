import os
import requests
from tqdm import tqdm
import argparse

def download_file(url, destination):
    """下载文件并显示进度条"""
    if os.path.exists(destination):
        print(f"文件已存在: {destination}")
        return
    
    print(f"下载: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # 确保目标目录存在
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    print(f"下载完成: {destination}")

def main():
    parser = argparse.ArgumentParser(description='下载预训练模型')
    parser.add_argument('--output_dir', type=str, default='./pretrained_models', help='模型保存目录')
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 模型配置
    models = {
        'vit_small_patch16_224': 'https://huggingface.co/timm/vit_small_patch16_224.dino/resolve/main/model.safetensors',
        'resnet34': 'https://huggingface.co/timm/resnet34.a1_in1k/resolve/main/model.safetensors',
        'resmlp_12_224': 'https://huggingface.co/timm/resmlp_12_224.fb_in1k/resolve/main/model.safetensors',
        'poolformer_s12': 'https://huggingface.co/timm/poolformer_s12.sail_in1k/resolve/main/model.safetensors',
        'vgg11': 'https://huggingface.co/timm/vgg11.tv_in1k/resolve/main/model.safetensors',
        'mixer_b16_224': 'https://huggingface.co/timm/mixer_b16_224.miil_in21k/resolve/main/model.safetensors',
        'pvt_v2_b3': 'https://huggingface.co/timm/pvt_v2_b3.in1k/resolve/main/model.safetensors'
    }
    
    # 使用镜像站点
    mirror_models = {
        'vit_small_patch16_224': 'https://hf-mirror.com/timm/vit_small_patch16_224.dino/resolve/main/model.safetensors',
        'resnet34': 'https://hf-mirror.com/timm/resnet34.a1_in1k/resolve/main/model.safetensors',
        'resmlp_12_224': 'https://hf-mirror.com/timm/resmlp_12_224.fb_in1k/resolve/main/model.safetensors',
        'poolformer_s12': 'https://hf-mirror.com/timm/poolformer_s12.sail_in1k/resolve/main/model.safetensors',
        'vgg11': 'https://hf-mirror.com/timm/vgg11.tv_in1k/resolve/main/model.safetensors',
        'mixer_b16_224': 'https://hf-mirror.com/timm/mixer_b16_224.miil_in21k/resolve/main/model.safetensors',
        'pvt_v2_b3': 'https://hf-mirror.com/timm/pvt_v2_b3.in1k/resolve/main/model.safetensors'
    }
    
    # 下载模型
    for model_name, url in mirror_models.items():
        destination = os.path.join(args.output_dir, f"{model_name}.safetensors")
        try:
            download_file(url, destination)
        except Exception as e:
            print(f"下载 {model_name} 失败: {str(e)}")
            print("尝试使用原始链接...")
            try:
                download_file(models[model_name], destination)
            except Exception as e:
                print(f"从原始链接下载 {model_name} 也失败: {str(e)}")

if __name__ == "__main__":
    main() 