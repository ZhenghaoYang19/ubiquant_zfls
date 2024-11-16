import torch
from torch.utils.data import DataLoader
from models.resnet import resnet18
from models.openmax import OpenMax
from train import GameDataset, evaluate_openmax
from torchvision import transforms
from utils.data_stats import load_dataset_stats

def train_openmax():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据集统计信息
    mean, std = load_dataset_stats()
    
    # 准备数据集和转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 加载训练集和验证集
    train_dataset = GameDataset('jk_zfls/round0_train', num_labels=20, transform=transform)
    val_dataset = GameDataset('jk_zfls/round0_eval', num_labels=21, transform=transform)
    
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=4, pin_memory=True)
    
    # 加载预训练模型
    model = resnet18(num_classes=20)
    checkpoint = torch.load('models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 收集所有训练数据的logits
    logits_list = []
    labels_list = []
    
    print("Collecting logits from training set...")
    for augment_round in range(3):  # 收集3轮数据
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(device)
                _, logits = model(images, return_logits=True)
                logits_list.append(logits.cpu())
                labels_list.append(labels)
    
    # 合并所有logits和labels
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    
    print("Training OpenMax...")
    # 训练OpenMax
    openmax = OpenMax(num_classes=20)
    openmax.fit(logits, labels)
    
    # 保存OpenMax模型
    torch.save(openmax, 'models/openmax.pth')
    print("OpenMax model saved at: models/openmax.pth")
    
    # 评估OpenMax性能
    print("Evaluating OpenMax...")
    evaluate_openmax(openmax, model, val_loader, device)

if __name__ == '__main__':
    train_openmax()
