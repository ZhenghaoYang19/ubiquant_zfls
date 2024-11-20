import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
import wandb
from PIL import Image
from models.resnet import resnet18
from models.openmax import OpenMax
from models.metamax import MetaMax
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



def train(num_epochs = 20, batch_size = 256, learning_rate = 0.001, dropout_rate = 0.3, patience = 10):
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
        name="resnet18-openmax-training",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "model": "resnet18-openmax",
            "num_classes": 20
        },
        dir="./wandb_logs"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 增加数据增强
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomAffine(
        #     degrees=[-15, 15],                      # 限制旋转角度在±15度以内
        #     translate=(0.1, 0.1),                   # 在水平和垂直方向上最多移动10%的图像大小
        #     fill=255                                # 填充白色（使用 fill 而不是 fillcolor）
        # ),
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
    
    # 加载模型（和已有参数）
    model = resnet18(num_classes=20, dropout_rate=dropout_rate)
    checkpoint = torch.load('models/best_model_99.75.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 定义损失函数和优化器，使用更小的学习率
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate * 0.1, weight_decay=1e-4)
    
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # 使用带 warmup 的 cosine 调度器
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = len(train_loader) * 2      # 2个epoch的warmup
    
    def warmup_cosine_schedule(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_schedule)
    
    best_val_acc = 0
    patience_counter = 0  # 计数器，记录连续没有提升的轮数
    
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
        
        train_loss = total_loss / len(train_loader)
        
        # 验证阶段（只验证已知类别）
        val_loss, val_acc, val_errors = evaluate_known_classes(model, val_loader, criterion, device)
        if val_acc > 98:
            pprint(val_errors)
        
        # 记录到wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })
        
        print(f'Epoch {epoch}:')
        print(f'Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.2f}%')
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型（基于验证集准确率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0  # 重置计数器
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }
            torch.save(save_dict, 'models/best_model.pth')
            print(f'Saved best model at epoch {epoch}')
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
            
        print(f'Best val acc: {best_val_acc:.2f}%')
    
    # 训练完成后，加载最佳模型的参数
    print("Loading best model parameters...")
    checkpoint = torch.load('models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 使用最佳模型收集features
    print("Collecting features from best model for OpenMax/MetaMax training...")
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels, paths in train_loader:
            images = images.to(device)
            _, features = model(images, return_features=True)  # 获取features
            features_list.append(features)
            labels_list.append(labels)
    
    features = torch.cat(features_list)
    labels = torch.cat(labels_list)
    
    # 训练OpenMax/MetaMax
    openmax = OpenMax(num_classes=20)
    metamax = MetaMax(num_classes=20)
    
    openmax.fit(features, labels)
    metamax.fit(features, labels)
    
    # 保存模型
    torch.save(openmax, 'models/openmax.pth')
    torch.save(metamax, 'models/metamax.pth')
    print("OpenMax and MetaMax models saved")
    # 在训练完OpenMax后添加评估
    evaluate_openmax(openmax, model, val_loader, device)
    evaluate_metamax(metamax, model, val_loader, device)
    wandb.finish()

if __name__ == '__main__':
    train(num_epochs=60, batch_size=64, learning_rate=0.001, dropout_rate=0.3, patience=10)
