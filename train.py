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
from pprint import pprint
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
    errors = []  # 用于储错误预测的信息
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            # 只评估已知类别的样本
            mask = labels < 20
            if not mask.any():
                continue
                
            images = images[mask].to(device)
            labels = labels[mask].to(device)
            
            outputs, logits = model(images, return_logits=True)
            loss = criterion(logits, labels)
            
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
    
    # # 打印错误预测的信息
    # if errors:
    #     print("\nIncorrect predictions:")
    #     for error in errors:
    #         print(f"Sample {error['sample_idx']}: True label = {error['true_label']}, "
    #             f"Predicted = {error['predicted']}")
    
    return avg_loss, accuracy, errors

def evaluate_openmax(openmax, model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            _, logits = model(images, return_logits=True)
            
            # 批量预测
            openmax_probs = openmax.predict(logits)
            predictions = torch.argmax(openmax_probs, dim=1)
            
            # 分别统计已知类和未知类的准确率
            known_mask = labels < 20
            unknown_mask = labels == 20
            
            # 已知类统计
            if known_mask.any():
                known_correct += (predictions[known_mask] == labels[known_mask]).sum().item()
                known_total += known_mask.sum().item()
            
            # 未知类统计
            if unknown_mask.any():
                unknown_correct += (predictions[unknown_mask] == labels[unknown_mask]).sum().item()
                unknown_total += unknown_mask.sum().item()
            
            # 总体统计
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    # 计算准确率
    overall_acc = 100. * correct / total if total > 0 else 0
    known_acc = 100. * known_correct / known_total if known_total > 0 else 0
    unknown_acc = 100. * unknown_correct / unknown_total if unknown_total > 0 else 0
    
    print("\n=== OpenMax Evaluation Results ===")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"Known Classes Accuracy: {known_acc:.2f}%")
    print(f"Unknown Class Accuracy: {unknown_acc:.2f}%")
    
    return overall_acc, known_acc, unknown_acc

def train(num_epochs = 20, batch_size = 256, learning_rate = 0.001, dropout_rate = 0.3):
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
        transforms.RandomRotation([-15, 15]),       # 限制旋转角度在±15度以内
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
    
    # 加载模型
    model = resnet18(num_classes=20, dropout_rate=dropout_rate)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
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
            outputs, logits = model(images, return_logits=True)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            # 收集logits
            logits_list.append(logits.detach().cpu())
            labels_list.append(labels.cpu())
            
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
        pprint(val_errors)
        
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
        scheduler.step()
        
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
        print(f'best val acc: {best_val_acc:.2f}%')
    
    # 训练完成后，训练OpenMax
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    openmax = OpenMax(num_classes=20)
    openmax.fit(logits, labels)
    # 保存OpenMax模型
    torch.save(openmax, 'models/openmax.pth')
    print("OpenMax model saved at: models/openmax.pth")
    # 在训练完OpenMax后添加评估
    evaluate_openmax(openmax, model, val_loader, device)
    
    wandb.finish()

if __name__ == '__main__':
    train(num_epochs=20, batch_size=256, learning_rate=0.001, dropout_rate=0.3)
