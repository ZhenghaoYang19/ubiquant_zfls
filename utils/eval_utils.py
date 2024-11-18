import torch
import numpy as np

def evaluate_known_classes(model, data_loader, criterion, device):
    """评估已知类别（0-19）的性能"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    errors = []  # 用于储存错误预测的信息
    
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(data_loader):
            # 只评估已知类别的样本
            mask = labels < 20
            if not mask.any():
                continue
                
            images = images[mask].to(device)
            labels = labels[mask].to(device)
            paths = np.array(paths)[mask]
            
            logits, features = model(images, return_features=True)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            
            # 记录错误预测的样本
            incorrect_mask = ~predicted.eq(labels)
            if incorrect_mask.any():
                incorrect_indices = torch.where(incorrect_mask)[0]
                for idx in incorrect_indices:
                    errors.append({
                        # 'batch': batch_idx,
                        'true_label': labels[idx].item(),
                        'predicted': predicted[idx].item(),
                        # 'sample_idx': batch_idx * data_loader.batch_size + idx.item(),
                        'image_path': paths[idx]  # 添加图片路径信息
                    })
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    if total == 0:
        return 0, 0, []
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, accuracy, errors

def evaluate_openmax(openmax, model, val_loader, device, threshold=0.3, verbose=False):
    model.eval()
    correct = 0
    total = 0
    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0
    
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # 获取features和logits
            logits, features = model(images, return_features=True)
            
            # 使用features和logits进行预测
            openmax_probs = openmax.predict(features, logits)
            
            # 使用阈值判断未知类别
            max_probs, predictions = torch.max(openmax_probs[:, :-1], dim=1)
            # 如果最大概率小于阈值，则判定为未知类别(20)
            predictions[max_probs < threshold] = 20
            
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

    if verbose:
        print(f"\n=== OpenMax Evaluation Results ===")
        print(f"Threshold: {threshold}")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        print(f"Known Classes Accuracy: {known_acc:.2f}%")
        print(f"Unknown Class Accuracy: {unknown_acc:.2f}%")
        
    return overall_acc, known_acc, unknown_acc

def evaluate_metamax(metamax, model, val_loader, device, threshold=0.3, verbose=False):
    """评估MetaMax模型的性能
    Args:
        metamax: MetaMax模型实例
        model: 基础特征提取器
        val_loader: 验证集数据加载器
        device: 计算设备
        threshold: 未知类别判断阈值
        verbose: 是否打印详细信息
    Returns:
        overall_acc: 总体准确率
        known_acc: 已知类别准确率
        unknown_acc: 未知类别准确率
    """
    model.eval()
    correct = 0
    total = 0
    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0
    
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # 获取features和logits
            logits, features = model(images, return_features=True)
            
            # 使用MetaMax进行预测
            metamax_probs = metamax.predict(features, logits)
            
            # 使用阈值判断未知类别
            max_probs, predictions = torch.max(metamax_probs[:, :-1], dim=1)
            # 如果最大概率小于阈值，则判定为未知类别(20)
            predictions[max_probs < threshold] = 20
            
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

    if verbose:
        print(f"\n=== MetaMax Evaluation Results ===")
        print(f"Threshold: {threshold}")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        print(f"Known Classes Accuracy: {known_acc:.2f}%")
        print(f"Unknown Class Accuracy: {unknown_acc:.2f}%")
        
    return overall_acc, known_acc, unknown_acc