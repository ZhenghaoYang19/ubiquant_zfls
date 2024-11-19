import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from torch.utils.data import DataLoader
import umap
from models.resnet import resnet18
import os

def visualize_features(features, labels, method='tsne', save_path=None, include_unknown=False):
    """
    可视化特征分布
    
    Args:
        features: torch.Tensor, 特征向量
        labels: torch.Tensor, 标签
        method: str, 'tsne' 或 'umap'
        save_path: str, 保存路径，如果为None则显示图像
        include_unknown: bool, 是否包含未知类（第21类）
    """
    # 转换为numpy数组
    features = features.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 降维
    print(f"Performing {method.upper()} dimensionality reduction...")
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embedded = reducer.fit_transform(features)
    else:  # umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedded = reducer.fit_transform(features)
    
    # 清理之前的图像状态并创建新图形
    plt.close('all')  # 关闭所有图形
    fig = plt.figure(figsize=(15, 10))
    
    # 定义标记样式和颜色
    markers = ['o', 's', '^', 'D']  # 圆形、方形、三角形、菱形
    colors = ['#FF4B4B', '#4B4BFF', '#4BFF4B', '#FFB74B', '#B74BFF']  # 红、蓝、绿、橙、紫
    
    # 确定要绘制的类别数量
    num_classes = 21 if include_unknown else 20
    
    # 为每个类别分配标记和颜色
    for i in range(num_classes):
        marker_idx = i % len(markers)
        color_idx = i % len(colors)
        
        mask = labels == i
        if i == 20:  # 未知类使用特殊标记
            plt.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                c='gray',  # 使用灰色
                marker='*',  # 使用星形
                s=150,  # 稍微大一点
                alpha=0.6,
                label='Unknown',
                edgecolors='white',
                linewidth=0.5
            )
        else:
            plt.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                c=colors[color_idx],
                marker=markers[marker_idx],
                s=100,
                alpha=0.6,
                label=f'Class {i}',
                edgecolors='white',
                linewidth=0.5
            )
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Feature Distribution ({method.upper()})', fontsize=14, pad=20)
    
    # 调整图例
    plt.legend(bbox_to_anchor=(1.05, 1), 
            loc='upper left', 
            borderaxespad=0,
            ncol=1,  # 使用单列显示图例
            fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 先保存再显示
    if save_path:
        print(f"Saving plot to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.5)
    
    plt.show()
    plt.close()



def load_features_labels(model_path, data_loader, device=None):
    """
    加载模型和数据，提取特征和标签
    
    Args:
        model_path: str, 模型参数文件路径
        data_loader: DataLoader, 数据加载器
        device: torch.device, 计算设备
    
    Returns:
        features: torch.Tensor, 特征向量
        labels: torch.Tensor, 标签
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = resnet18(num_classes=20)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 收集特征和标签
    features_list = []
    labels_list = []
    
    print("Collecting features...")
    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(device)
            _, features = model(images, return_features=True)
            features_list.append(features.cpu())
            labels_list.append(labels)
    
    features = torch.cat(features_list)
    labels = torch.cat(labels_list)
    
    return features, labels

if __name__ == "__main__":
    from train import GameDataset
    import torchvision.transforms as transforms
    from utils.data_stats import load_dataset_stats
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据集统计信息
    mean, std = load_dataset_stats()
    
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    # # 加载训练数据集
    # train_dataset = GameDataset(
    #     data_dir='jk_zfls/round0_train',
    #     num_labels=20,
    #     transform=transform
    # )
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=400,
    #     shuffle=False,  # 设为False以保持数据顺序
    #     num_workers=4,
    #     pin_memory=True
    # )
    # 加载特征
    # features, labels = load_features_labels(
    #     model_path='models/best_model_99.25.pth',
    #     data_loader=train_loader,
    #     device=device
    # )
    
    eval_dataset = GameDataset(
        data_dir='jk_zfls/round0_eval',
        num_labels=21,
        transform=transform
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=400,
        shuffle=False,
    )
    features, labels = load_features_labels(
        model_path='models/best_model_99.25.pth',
        data_loader=eval_loader,
        device=device
    )
    # 可视化特征
    # print("Visualizing features using t-SNE...")
    # visualize_features(
    #     features=features,
    #     labels=labels,
    #     method='tsne',
    #     save_path='outputs/tsne_features.png'
    # )
    
    print("Visualizing features using UMAP...")
    visualize_features(
        features=features,
        labels=labels,
        method='umap',
        save_path='outputs/eval_dataset_features.png',
        include_unknown=True
    )
