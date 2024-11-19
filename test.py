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
    test_loader = DataLoader(test_dataset, batch_size=400, shuffle=False, num_workers=4, pin_memory=True)
    
    # 加载基础模型
    model = resnet18(num_classes=20)
    checkpoint = torch.load('models/best_model_99.25.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 加载OpenMax和MetaMax模型
    try:
        openmax = torch.load('models/best_openmax.pth')
        # metamax = torch.load('models/best_metamax.pth')
        print("Successfully loaded OpenMax and MetaMax models")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # 测试基础ResNet
    print("\n=== Testing ResNet (Known Classes Only) ===")
    _, accuracy, errors = evaluate_known_classes(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    print(f"Known Classes Accuracy: {accuracy:.2f}%")
    if errors:
        print("\nErrors in known classes:")
        pprint(errors)
    
    # 测试ResNet + OpenMax
    print("\n=== Testing ResNet + OpenMax ===")
    evaluate_openmax(openmax, model, test_loader, device, verbose=True)
    
    # 测试ResNet + MetaMax
    # print("\n=== Testing ResNet + MetaMax ===")
    # evaluate_metamax(metamax, model, test_loader, device, threshold=0.5, verbose=True)

if __name__ == '__main__':
    test_models()
