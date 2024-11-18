import torch
from torch.utils.data import DataLoader
from models.resnet import resnet18
from models.openmax import OpenMax
from models.metamax import MetaMax
from train import GameDataset
from utils.eval_utils import evaluate_openmax, evaluate_metamax
from torchvision import transforms
from utils.data_stats import load_dataset_stats
from pprint import pprint
def prepare_data_and_model(model_path='models/best_model.pth'):
    """准备数据和模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据集统计信息和准备数据
    mean, std = load_dataset_stats()
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
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, train_loader, val_loader, device

def collect_features(model, train_loader, device):
    """收集特征和标签"""
    features_list = []
    labels_list = []
    
    print("Collecting features and logits from training set...")
    with torch.no_grad():
        for images, labels, paths in train_loader:
            images = images.to(device)
            _, features = model(images, return_features=True)
            features_list.append(features.cpu())
            labels_list.append(labels)
    
    return torch.cat(features_list), torch.cat(labels_list)

def train_openmax(features, labels, model, val_loader, device):
    """训练和评估OpenMax模型"""
    # OpenMax特定的超参数搜索空间
    alpha_range = [5, 10, 15]
    # tailsize_range = [20]
    tailsize_range = [15, 20, 25, 30]
    threshold_range = [0.06, 0.08, 0.1, 0.12, 0.14, 0.16]
    
    best_params = {
        'alpha': None,
        'tailsize': None,
        'threshold': None,
        'accuracy': .0,
        'model': None
    }
    
    print("\n=== Training OpenMax ===")
    for alpha in alpha_range:
        for tailsize in tailsize_range:
            print(f"\nTesting OpenMax with alpha={alpha}, tailsize={tailsize}")
            
            openmax = OpenMax(num_classes=20, tailsize=tailsize, alpha=alpha)
            openmax.fit(features, labels)
            
            for threshold in threshold_range:
                overall_acc, known_acc, unknown_acc = evaluate_openmax(
                    openmax, model, val_loader, device, threshold=threshold, verbose=False
                )
                
                if overall_acc > best_params['accuracy']:
                    best_params.update({
                        'alpha': alpha,
                        'tailsize': tailsize,
                        'threshold': threshold,
                        'accuracy': overall_acc,
                        'model': openmax
                    })
                    if overall_acc > 90.0:
                        print(f"\nNew best OpenMax parameters found:")
                        print(f"Alpha: {alpha}")
                        print(f"Tailsize: {tailsize}")
                        print(f"Threshold: {threshold}")
                        print(f"Overall Accuracy: {overall_acc:.2f}%")
                        print(f"Known Classes Accuracy: {known_acc:.2f}%")
                        print(f"Unknown Class Accuracy: {unknown_acc:.2f}%")
    
    return best_params

def train_metamax(features, labels, model, val_loader, device):
    """训练和评估MetaMax模型"""
    # MetaMax特定的超参数搜索空间
    meta_ratio_range = [0.05, 0.1, 0.15, 0.2, 0.25]
    threshold_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    best_params = {
        'meta_ratio': None,
        'threshold': None,
        'accuracy': .0,
        'model': None
    }
    
    print("\n=== Training MetaMax ===")
    for meta_ratio in meta_ratio_range:
        print(f"\nTesting MetaMax with meta_ratio={meta_ratio}")
        metamax = MetaMax(num_classes=20, meta_ratio=meta_ratio)
        metamax.fit(features, labels)
        
        for threshold in threshold_range:
            overall_acc, known_acc, unknown_acc = evaluate_metamax(
                metamax, model, val_loader, device, threshold=threshold, verbose=False
            )
            
            if overall_acc > best_params['accuracy']:
                best_params.update({
                    'meta_ratio': meta_ratio,
                    'threshold': threshold,
                    'accuracy': overall_acc,
                    'model': metamax
                })
                if overall_acc > 90.0:
                    print(f"\nNew best MetaMax parameters found:")
                    print(f"Meta Ratio: {meta_ratio}")
                    print(f"Threshold: {threshold}")
                    print(f"Overall Accuracy: {overall_acc:.2f}%")
                    print(f"Known Classes Accuracy: {known_acc:.2f}%")
                    print(f"Unknown Class Accuracy: {unknown_acc:.2f}%")
    
    return best_params

if __name__ == '__main__':
    # 准备数据和模型
    model, train_loader, val_loader, device = prepare_data_and_model(model_path='models/best_model.pth')
    
    # 收集特征
    features, labels = collect_features(model, train_loader, device)
    
    # 训练OpenMax
    best_openmax_params = train_openmax(features, labels, model, val_loader, device)
    print("\nSaving OpenMax model...")
    pprint(best_openmax_params)
    torch.save(best_openmax_params['model'], 'models/best_openmax.pth')
    print(f"OpenMax model saved to models/best_openmax.pth")

    # 训练MetaMax
    best_metamax_params = train_metamax(features, labels, model, val_loader, device)
    print("\nSaving MetaMax model...")
    pprint(best_metamax_params)
    torch.save(best_metamax_params['model'], 'models/best_metamax.pth')
    print(f"MetaMax model saved to models/best_metamax.pth")

