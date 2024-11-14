import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def calculate_dataset_stats(data_dir):
    """计算数据集的均值和标准差
    Args:
        data_dir: 数据集目录路径
    Returns:
        mean: 各通道的均值
        std: 各通道的标准差
    """
    total_mean = np.zeros(3)
    total_std = np.zeros(3)
    total_images = 0
    
    # 遍历数据目录
    for class_dir in range(20):  # 只使用训练集(0-19类)计算统计信息
        class_path = os.path.join(data_dir, f"{class_dir:02d}")
        if os.path.exists(class_path):
            for img_name in os.listdir(class_path):
                if img_name.endswith('.png'):
                    img_path = os.path.join(class_path, img_name) 
                    try:
                        # 读取PNG图片，只保留RGB通道
                        img = np.array(Image.open(img_path))[:, :, :3]  # 只取前3个通道
                        if img.shape != (50, 50, 3):
                            print(f"Skipping {img_path} due to invalid shape: {img.shape}")
                            continue
                            
                        # 将图片归一化到 [0,1]
                        img = img.astype(np.float32) / 255.0
                        
                        # 计算均值和标准差
                        mean = np.mean(img, axis=(0, 1))
                        std = np.std(img, axis=(0, 1))
                        
                        total_mean += mean
                        total_std += std
                        total_images += 1
                        
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue
        else:
            print(f"Class directory {class_dir} does not exist")    
    
    if total_images == 0:
        raise ValueError("No valid images found in the dataset")
        
    # 计算最终的均值和标准差
    mean = total_mean / total_images
    std = total_std / total_images
    
    # 保存结果
    stats = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }
    
    stats_path = os.path.join('utils', 'dataset_stats.json')
    os.makedirs('utils', exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Dataset statistics saved to {stats_path}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    
    return mean, std

def load_dataset_stats():
    """加载数据集统计信息"""
    stats_path = os.path.join('utils', 'dataset_stats.json')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            "Dataset statistics not found. Please run calculate_dataset_stats first."
        )
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    return np.array(stats['mean']), np.array(stats['std'])
