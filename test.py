import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.resnet import resnet18
from models.openmax import OpenMax
from models.metamax import MetaMax
from train import GameDataset
from utils.data_stats import load_dataset_stats
from utils.eval_utils import evaluate_known_classes, evaluate_openmax, evaluate_metamax
import os
from pprint import pprint

def test_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据集统计信息
    mean, std = load_dataset_stats()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 加载验证集
    test_dataset = GameDataset('jk_zfls/round0_eval', num_labels=21, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    # 加载基础模型
    model = resnet18(num_classes=20)
    checkpoint = torch.load('models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 加载OpenMax和MetaMax模型
    try:
        # openmax = torch.load('models/best_openmax.pth')
        metamax = torch.load('models/best_metamax.pth')
        print("Successfully loaded OpenMax and MetaMax models")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # 测试基础ResNet
    print("\n=== Testing ResNet (Known Classes Only) ===")
    test_resnet(model, test_loader, device)
    
    # 测试ResNet + OpenMax
    # print("\n=== Testing ResNet + OpenMax ===")
    # test_openmax(model, openmax, test_loader, device)
    
    # 测试ResNet + MetaMax
    print("\n=== Testing ResNet + MetaMax ===")
    test_metamax(model, metamax, test_loader, device)

def test_resnet(model, test_loader, device):
    """测试基础ResNet在已知类别上的性能"""
    correct = 0
    total = 0
    errors = []
    
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(test_loader):
            # 只测试已知类别
            mask = labels < 20
            if not mask.any():
                continue
                
            images = images[mask].to(device)
            labels = labels[mask].to(device)
            paths = [paths[i] for i in range(len(paths)) if mask[i]]
            
            logits, _ = model(images, return_features=True)
            _, predicted = logits.max(1)
            
            # 记录错误预测
            incorrect_mask = ~predicted.eq(labels)
            if incorrect_mask.any():
                incorrect_indices = torch.where(incorrect_mask)[0]
                for idx in incorrect_indices:
                    errors.append({
                        'true_label': labels[idx].item(),
                        'predicted': predicted[idx].item(),
                        'image_path': paths[idx]
                    })
            
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100. * correct / total if total > 0 else 0
    print(f"Known Classes Accuracy: {accuracy:.2f}%")
    if errors:
        print("\nErrors in known classes:")
        pprint(errors)

def test_openmax(model, openmax, test_loader, device, threshold=0.5):
    """测试ResNet + OpenMax的性能"""
    correct = 0
    total = 0
    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0
    errors = []
    
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            logits, features = model(images, return_features=True)
            openmax_probs = openmax.predict(features, logits)
            
            max_probs, predictions = torch.max(openmax_probs[:, :-1], dim=1)
            predictions[max_probs < threshold] = 20
            
            # 记录错误预测
            incorrect_mask = ~predictions.eq(labels)
            if incorrect_mask.any():
                incorrect_indices = torch.where(incorrect_mask)[0]
                for idx in incorrect_indices:
                    errors.append({
                        'true_label': labels[idx].item(),
                        'predicted': predictions[idx].item(),
                        'image_path': paths[idx]
                    })
            
            # 分别统计已知类和未知类的准确率
            known_mask = labels < 20
            unknown_mask = labels == 20
            
            if known_mask.any():
                known_correct += (predictions[known_mask] == labels[known_mask]).sum().item()
                known_total += known_mask.sum().item()
            
            if unknown_mask.any():
                unknown_correct += (predictions[unknown_mask] == labels[unknown_mask]).sum().item()
                unknown_total += unknown_mask.sum().item()
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    overall_acc = 100. * correct / total if total > 0 else 0
    known_acc = 100. * known_correct / known_total if known_total > 0 else 0
    unknown_acc = 100. * unknown_correct / unknown_total if unknown_total > 0 else 0
    
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"Known Classes Accuracy: {known_acc:.2f}%")
    print(f"Unknown Class Accuracy: {unknown_acc:.2f}%")
    if errors:
        print("\nErrors:")
        pprint(errors)

def test_metamax(model, metamax, test_loader, device, threshold=0.5):
    """测试ResNet + MetaMax的性能"""
    correct = 0
    total = 0
    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0
    errors = []
    
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            logits, features = model(images, return_features=True)
            metamax_probs = metamax.predict(features, logits)
            
            max_probs, predictions = torch.max(metamax_probs[:, :-1], dim=1)
            predictions[max_probs < threshold] = 20
            
            # 记录错误预测
            incorrect_mask = ~predictions.eq(labels)
            if incorrect_mask.any():
                incorrect_indices = torch.where(incorrect_mask)[0]
                for idx in incorrect_indices:
                    errors.append({
                        'true_label': labels[idx].item(),
                        'predicted': predictions[idx].item(),
                        'image_path': paths[idx]
                    })
            
            # 分别统计已知类和未知类的准确率
            known_mask = labels < 20
            unknown_mask = labels == 20
            
            if known_mask.any():
                known_correct += (predictions[known_mask] == labels[known_mask]).sum().item()
                known_total += known_mask.sum().item()
            
            if unknown_mask.any():
                unknown_correct += (predictions[unknown_mask] == labels[unknown_mask]).sum().item()
                unknown_total += unknown_mask.sum().item()
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    overall_acc = 100. * correct / total if total > 0 else 0
    known_acc = 100. * known_correct / known_total if known_total > 0 else 0
    unknown_acc = 100. * unknown_correct / unknown_total if unknown_total > 0 else 0
    
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"Known Classes Accuracy: {known_acc:.2f}%")
    print(f"Unknown Class Accuracy: {unknown_acc:.2f}%")
    if errors:
        print("\nErrors:")
        pprint(errors)

if __name__ == '__main__':
    test_models()
