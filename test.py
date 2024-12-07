import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.resnet import resnet18, resnet50
from train import GameDataset
from utils.data_stats import load_dataset_stats
from utils.eval_utils import evaluate_known_classes, evaluate_openmax
from post_train import collect_features
from pprint import pprint

def test_models(model_type=None, model_path=None, openmax_path=None, data_path=None, threshold=0.05, fraction=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据集统计信息
    mean, std = load_dataset_stats()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 加载验证集
    eval_dataset = GameDataset(data_path, num_labels=21, transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=400, shuffle=False, num_workers=4, pin_memory=True)
    
    # 加载基础模型
    if model_type == 'resnet18':
        model = resnet18(num_classes=20)
    elif model_type == 'resnet50':
        model = resnet50(num_classes=20)
    else:
        print(f"Unsupported model type: {model_type}")
        return
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 加载OpenMax和MetaMax模型
    try:
        openmax = torch.load(openmax_path)
        print("Successfully loaded OpenMax model")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # 测试基础ResNet
    print("\n=== Testing ResNet (Known Classes Only) ===")
    _, accuracy, errors = evaluate_known_classes(model, eval_loader, torch.nn.CrossEntropyLoss(), device)
    print(f"Known Classes Accuracy: {accuracy:.2f}%")
    if errors:
        print("\nErrors in known classes:")
        pprint(errors)
    
    # 测试ResNet + OpenMax
    print("\n=== Testing ResNet + OpenMax ===")
    features, logits, labels = collect_features(model, eval_loader, device, return_logits=True)
    overall_acc, known_acc, unknown_acc = evaluate_openmax(openmax, features, logits, labels, threshold=threshold, fraction=fraction, verbose=True)
    print(f"OpenMax Accuracy: {overall_acc:.2f}%, Known Classes Accuracy: {known_acc:.2f}%, Unknown Classes Accuracy: {unknown_acc:.2f}%")

    
    # 测试ResNet + MetaMax
    # print("\n=== Testing ResNet + MetaMax ===")
    # evaluate_metamax(metamax, model, test_loader, device, threshold=0.5, verbose=True)

if __name__ == '__main__':
    test_models(model_type='resnet18', model_path='models/best_model_99.92_02.pth', openmax_path='models/best_openmax_95.62_02.pth', data_path='jk_zfls/round0_test', threshold=0.05, fraction=0.1666)
