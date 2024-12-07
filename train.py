import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
import wandb
from PIL import Image
from models.resnet import resnet18, resnet34, resnet50
from models.openmax import OpenMax
# from models.metamax import MetaMax
from utils.data_stats import calculate_dataset_stats, load_dataset_stats
from utils.eval_utils import evaluate_known_classes, evaluate_openmax, evaluate_metamax
from pprint import pprint
import math


class GameDataset(Dataset):
    def __init__(self, data_dir, num_labels=20, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.image_paths = []
        
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist")
            
        # 遍历数据目录加载图片和标签
        for class_dir in range(num_labels):  # 训练集为0-19类,验证集为0-20类
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
                                
                            self.images.append(img)
                            self.labels.append(class_dir)
                            self.image_paths.append(img_path)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
                            continue
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print(f"Loaded {len(self.images)} images from {data_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        path = self.image_paths[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, path



def train(num_epochs = 20, batch_size = 256, learning_rate = 0.001, dropout_rate = 0.3, patience = 10, model_type='resnet34'):
    from post_train import collect_features
    os.makedirs('models', exist_ok=True)
    os.makedirs('wandb_logs', exist_ok=True)
    images_path = os.path.join('jk_zfls', 'round0_train')
    # 尝试加载已保存的数据集统计信息，如果不存在则重新计算
    try:
        mean, std = load_dataset_stats()
        print("Loaded pre-calculated dataset statistics")
    except FileNotFoundError:
        print("FileNotFound, Calculating dataset statistics...")
        mean, std = calculate_dataset_stats(images_path)
        
    wandb.init(
        project="jk_zfls",
        name=f"{model_type}-training",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "model": f"{model_type}",
            "num_classes": 20
        },
        dir="./wandb_logs"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 计算填充值 (将均值从[0,1]转换为[0,255])
    fill_value = tuple(int(x * 255) for x in mean)
    
    # 增加数据增强
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            fill=fill_value  # 使用数据集的均值作为填充值
        ),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 验证集不需要数据增强
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 加载数据集
    train_dataset = GameDataset('jk_zfls/round0_train', num_labels=20, transform=transform)
    val_dataset = GameDataset('jk_zfls/round0_eval', num_labels=21, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 根据选择加载不同的模型
    if model_type == 'resnet18':
        model = resnet18(num_classes=20, dropout_rate=dropout_rate)
    elif model_type == 'resnet34':
        model = resnet34(num_classes=20, dropout_rate=dropout_rate)
    elif model_type == 'resnet50':
        model = resnet50(num_classes=20, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    # 加载模型（和已有参数）
    # checkpoint = torch.load('models/best_model_99.75.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 定义损失函数和优化器，使用更小的学习率
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # 使用带 warmup 的 cosine 调度器
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = len(train_loader) * 2      # 2个epoch的warmup
    
    # 定义warmup调度器和ReduceLROnPlateau调度器
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,  # 从0.1倍的学习率开始
        end_factor=1.0,    # 最终达到设定的学习率
        total_iters=num_warmup_steps
    )

    reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    patience_counter = 0  # 计数器，记录连续没有提升的轮数
    best_params = {
        'epoch': None,
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'loss': None,
        'best_val_acc': 0
    }
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        
        for batch_idx, (images, labels, paths) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
            
            # 在warmup阶段更新学习率
            if epoch * len(train_loader) + batch_idx < num_warmup_steps:
                warmup_scheduler.step()
        
        train_loss = total_loss / len(train_loader)
        
        # 验证阶段（只验证已知类别）
        val_loss, val_acc, val_errors = evaluate_known_classes(model, val_loader, criterion, device)
        
        # 记录到wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })
        
        print(f'Epoch {epoch}:')
        print(f'Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.2f}%')
        
        # 验证阶段后更新ReduceLROnPlateau
        reduce_scheduler.step(val_acc)
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.2e}')
        
        # 记录最佳模型（基于验证集准确率）
        if val_acc >= best_params['best_val_acc']:
            patience_counter = 0  # 重置计数器
            best_params.update({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'best_val_acc': val_acc
            })
        else:
            patience_counter += 1  # 增加计数器
            print(f'Validation accuracy did not improve. Patience: {patience_counter}/{patience}')
        
        # 早停检查
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} consecutive epochs.")
            break
            
        if val_acc == 100:
            print(f'Achieved 100% accuracy at epoch {epoch}')
            break
            
    
    # 训练完成后，保存最佳模型的参数
    print("Saving best model parameters...")
    torch.save(best_params, f'models/{model_type}_{best_params["best_val_acc"]:.2f}.pth')
    
    # 使用最佳模型收集features
    print("Collecting features from best model for OpenMax/MetaMax training...")
    model.load_state_dict(best_params['model_state_dict'])
    model.eval()
    features, labels = collect_features(model, train_loader, device, return_logits=False)
    
    # 训练OpenMax/MetaMax
    openmax = OpenMax(num_classes=20)
    openmax.fit(features, labels)
    
    # metamax = MetaMax(num_classes=20)
    # metamax.fit(features, labels)
    
    # 保存模型
    torch.save(openmax, 'models/openmax.pth')
    # torch.save(metamax, 'models/metamax.pth')
    print("OpenMax and MetaMax models saved")
    # 在训练完OpenMax后添加评估
    print("Evaluating OpenMax and MetaMax...")
    val_features, val_logits, val_labels = collect_features(model, val_loader, device, return_logits=True)

    overall_acc, known_acc, unknown_acc = evaluate_openmax(openmax, val_features, val_logits, val_labels, multiplier=0.5)
    print(f"Multiplier: 0.5, Overall Accuracy: {overall_acc:.2f}%")
    # evaluate_metamax(metamax, val_features, val_labels, device)
    wandb.finish()

if __name__ == '__main__':
    train(num_epochs=100, batch_size=64, learning_rate=1e-4, dropout_rate=0.3, patience=20, model_type='resnet50')
