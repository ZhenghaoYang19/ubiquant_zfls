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
from utils.data_stats import calculate_dataset_stats, load_dataset_stats

class GameDataset(Dataset):
    def __init__(self, data_dir, num_labels=20, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
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
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def evaluate_known_classes(model, data_loader, criterion, device):
    """评估已知类别（0-19）的性能"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    errors = []  # 用于存储错误预测的信息
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            # 只评估已知类别的样本
            mask = labels < 20
            if not mask.any():
                continue
                
            images = images[mask].to(device)
            labels = labels[mask].to(device)
            
            outputs, _ = model(images, return_features=True)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            # 记录错误预测的样本
            incorrect_mask = ~predicted.eq(labels)
            if incorrect_mask.any():
                incorrect_indices = torch.where(incorrect_mask)[0]
                for idx in incorrect_indices:
                    errors.append({
                        'batch': batch_idx,
                        'true_label': labels[idx].item(),
                        'predicted': predicted[idx].item(),
                        'sample_idx': batch_idx * data_loader.batch_size + idx.item()
                    })
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    if total == 0:
        return 0, 0, []
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(data_loader)
    
    # 打印错误预测的信息
    if errors:
        print("\nIncorrect predictions:")
        for error in errors:
            print(f"Sample {error['sample_idx']}: True label = {error['true_label']}, "
                  f"Predicted = {error['predicted']}")
    
    return avg_loss, accuracy, errors

def train(num_epochs = 20):
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
            "learning_rate": 0.001,
            "batch_size": 128,
            "epochs": 50,
            "model": "resnet18-openmax",
            "num_classes": 20
        },
        dir="./wandb_logs"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用计算得到的均值和标准差进行数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 加载训练集和验证集（验证集只用已知类别的数据）
    train_dataset = GameDataset('jk_zfls/round0_train', num_labels=20, transform=transform)
    val_dataset = GameDataset('jk_zfls/round0_eval', num_labels=21, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # 加载模型
    model = resnet18(num_classes=20)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # 用于存储logits
    logits_list = []
    labels_list = []
    

    best_val_acc = 0
    patience = 5 #早停的耐心值
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, features = model(images, return_features=True)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 收集logits
            logits_list.append(outputs.detach())  
            labels_list.append(labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 验证阶段（只验证已知类别）
        val_loss, val_acc, val_errors = evaluate_known_classes(model, val_loader, criterion, device)
        print(val_errors)
        
        # 记录到wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })
        
        print(f'Epoch {epoch}:')
        print(f'Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.2f}%')
        print(f'Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.2f}%')
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型（基于验证集准确率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'logits': torch.cat(logits_list),
                'labels': torch.cat(labels_list)
            }
            torch.save(save_dict, 'models/best_model.pth')
            print(f'Saved best model at epoch {epoch}')
        
        # 早停计时器
        # if patience_counter >= patience:
        #     print("Early stopping triggered")
        #     break
        
        if val_acc == 100:
            print(f'Epoch {epoch}: Loss = {train_loss:.4f}, Accuracy = {train_acc:.2f}%')
            break
        
    
    # 训练完成后，训练OpenMax
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    openmax = OpenMax(num_classes=20)
    openmax.fit(logits, labels)
    # 保存OpenMax模型
    torch.save(openmax, 'models/openmax.pth')
    
    wandb.finish()

if __name__ == '__main__':
    train(num_epochs=20)
