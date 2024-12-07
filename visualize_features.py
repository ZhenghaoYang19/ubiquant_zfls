import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import umap
import os
from post_train import collect_features, prepare_data_and_model

def visualize_features(features, labels, method='tsne', save_path=None, include_unknown=False, reducer=None):
    """
    可视化特征分布
    
    Args:
        features: torch.Tensor, 特征向量
        labels: torch.Tensor, 标签
        method: str, 'tsne' 或 'umap'
        save_path: str, 保存路径，如果为None则显示图像
        include_unknown: bool, 是否包含未知类（第21类）
        reducer: object, 预训练好的降维模型，如果为None则重新训练
    """
    # 转换为numpy数组
    features = features.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 降维
    print(f"Performing {method.upper()} dimensionality reduction...")
    if reducer is None:
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            embedded = reducer.fit_transform(features)
        else:  # umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedded = reducer.fit_transform(features)
    else:
        # 使用预训练的模型进行转换
        embedded = reducer.transform(features)
    
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
    
    return reducer  # 返回训练好的降维模型


if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, train_loader, val_loader, device = prepare_data_and_model(model_path='models/best_model_99.92_02.pth', model_type='resnet18', batch_size=128)
    
    ## 加载特征
    print("Collecting features of training set...")
    features, labels = collect_features(
        model=model,
        loader=train_loader,
        device=device
    )
    
    print("Collecting features of validation set...")
    val_features, val_labels = collect_features(
        model=model,
        loader=val_loader,
        device=device
    )
    
    print("Visualizing features using UMAP...")
    umap_reducer = visualize_features(
        features=features,
        labels=labels,
        method='umap',
        save_path='outputs/resnet18_train_FeatureMap.png',
        include_unknown=False
    )
    
    visualize_features(
        features=val_features,
        labels=val_labels,
        method='umap',
        save_path='outputs/resnet18_val_FeatureMap.png',
        include_unknown=True,
        reducer=umap_reducer  # 使用训练集训练好的UMAP模型
    )
    
    visualize_features(
        features=val_features,
        labels=val_labels,
        method='tsne',
        save_path='outputs/resnet18_val_FeatureMap_01.png',
        include_unknown=True,
    )
