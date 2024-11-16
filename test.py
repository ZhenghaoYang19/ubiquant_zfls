import torch
from models.resnet import resnet18
import numpy as np
from PIL import Image
from utils.data_stats import load_dataset_stats
import os
from collections import Counter
from IPython import embed

class ResNetClassifier:
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(num_classes=20)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 加载数据集统计信息
        self.mean, self.std = load_dataset_stats()
    
    def predict_batch(self, images):
        """批量预测多张图片"""
        try:
            # 预处理
            imgs = torch.from_numpy(images).float()
            imgs = imgs.permute(0, 3, 1, 2)
            imgs = imgs / 255.0
            
            mean = torch.tensor(self.mean, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor(self.std, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
            
            imgs = imgs.to(self.device)
            imgs = (imgs - mean) / std
            
            with torch.no_grad():
                outputs = self.model(imgs)
                _, preds = torch.max(outputs, dim=1)
                return preds.cpu().numpy()
                
        except Exception as e:
            print(f"Error during batch prediction: {e}")
            return None
    
    def predict(self, img_path):
        """单张图片预测"""
        try:
            img = np.array(Image.open(img_path))[:, :, :3]
            img = np.expand_dims(img, axis=0)
            predictions = self.predict_batch(img)
            return predictions[0] if predictions is not None else None
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None


class OpenMaxClassifier:
    def __init__(self, model_path='models/best_model.pth', openmax_path='models/openmax.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(num_classes=20)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 加载OpenMax模型
        self.openmax = torch.load(openmax_path)
        
        # 加载数据集统计信息
        self.mean, self.std = load_dataset_stats()
    
    def predict_batch(self, images):
        """批量预测多张图片
        Args:
            images: numpy array, shape (batch_size, 50, 50, 3)
        Returns:
            list: 预测的类别
        """
        try:
            # 预处理
            imgs = torch.from_numpy(images).float()
            imgs = imgs.permute(0, 3, 1, 2)
            imgs = imgs / 255.0
            
            # 使用从数据集计算得到的均值和标准差，并确保使用float32类型
            mean = torch.tensor(self.mean, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor(self.std, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
            
            # 先将imgs移到设备上，再进行标准化
            imgs = imgs.to(self.device)
            imgs = (imgs - mean) / std
            
            with torch.no_grad():
                _, logits = self.model(imgs, return_logits=True)
                openmax_probs = self.openmax.predict(logits)
                max_probs, preds = torch.max(openmax_probs, dim=1)
                
                # 将低于阈值的预测设为未知类
                preds[max_probs < 0.5] = 20
                
                return preds.cpu().numpy()
                
        except Exception as e:
            print(f"Error during batch prediction: {e}")
            return None
    
    def predict(self, img_path):
        """单张图片预测的包装函数"""
        try:
            img = np.array(Image.open(img_path))[:, :, :3]
            
            # 扩展维度并使用批量预测
            img = np.expand_dims(img, axis=0)
            predictions = self.predict_batch(img)
            
            return predictions[0] if predictions is not None else None
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

def test_unknown_class(classifier, unknown_class_dir):
    """专门测试未知类（第20类）的性能"""
    predictions = []
    
    if not os.path.exists(unknown_class_dir):
        print(f"Error: Unknown class directory {unknown_class_dir} does not exist")
        return
    
    print("\nProcessing unknown class (20)...")
    
    # 遍历未知类下的所有图片
    for img_name in os.listdir(unknown_class_dir):
        if img_name.endswith('.png'):
            img_path = os.path.join(unknown_class_dir, img_name)
            prediction = classifier.predict(img_path)
            predictions.append({
                'true_label': 20,
                'predicted': prediction,
                'image_path': img_path
            })
            print(f"Image: {img_name}, True label: 20, Predicted: {prediction}")
    
    # 统计结果
    total = len(predictions)
    correct = sum(1 for p in predictions if p['predicted'] == 20)  # 预测为未知类的数量
    accuracy = correct / total * 100 if total > 0 else 0
    
    print("\n=== Unknown Class Test Results ===")
    print(f"Total unknown class images tested: {total}")
    print(f"Correctly identified as unknown: {correct}")
    print(f"Unknown class detection rate: {accuracy:.2f}%")
    
    # 打印预测分布
    print("\n=== Prediction Distribution ===")
    pred_counter = Counter(p['predicted'] for p in predictions)
    for label, count in sorted(pred_counter.items()):
        print(f"Predicted as Class {label}: {count} images ({count/total*100:.2f}%)")
    
    # 打印错误预测的案例
    print("\n=== Incorrectly Classified Unknown Samples ===")
    for p in predictions:
        if p['predicted'] != 20:
            print(f"Image: {p['image_path']}")
            print(f"Incorrectly predicted as class: {p['predicted']}")

def test_all_classes(classifier, eval_dir):
    """测试所有21个类别的性能"""
    predictions = []
    incorrect_predictions = []
    class_accuracies = {}
    print("\nProcessing all classes (0-20)...")
    
    # 遍历所有类别
    for class_idx in range(21):  # 0-20类
        class_dir = os.path.join(eval_dir, f"{class_idx:02d}")
        if not os.path.exists(class_dir):
            print(f"Warning: Directory for class {class_idx} does not exist")
            continue
            
        class_predictions = []
        # 遍历当前类别下的所有图片
        for img_name in os.listdir(class_dir):
            if img_name.endswith('.png'):
                img_path = os.path.join(class_dir, img_name)
                prediction = classifier.predict(img_path)
                # 只记录错误预测
                if prediction != class_idx:
                    incorrect_predictions.append({
                        'image_path': img_path,
                        'true_label': class_idx,
                        'predicted': prediction
                    })
                predictions.append((class_idx, prediction))
                class_predictions.append(prediction == class_idx)
                # print(f"Class {class_idx}, Image: {img_name}, Predicted: {prediction}")
        
        # # 计算当前类别的准确率
        if class_predictions:
            class_acc = sum(class_predictions) / len(class_predictions) * 100
            class_accuracies[class_idx] = class_acc
    
    # 计算并打印总体准确率
    total = len(predictions)
    correct = sum(1 for true, pred in predictions if true == pred)
    accuracy = correct / total * 100 if total > 0 else 0
    
    print("\n=== Overall Results ===")
    print(f"Total images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # 打印每个类别的准确率
    print("\n=== Per-Class Accuracies ===")
    for class_idx, accuracy in sorted(class_accuracies.items()):
        print(f"Class {class_idx}: {accuracy:.2f}%")
    
    # # 打印混淆矩阵相关信息
    # print("\n=== Confusion Matrix Statistics ===")
    # for true_label in range(21):
    #     class_preds = [p['predicted'] for p in predictions if p['true_label'] == true_label]
    #     if class_preds:
    #         print(f"\nTrue Class {true_label}:")
    #         pred_counter = Counter(class_preds)
    #         for pred_label, count in sorted(pred_counter.items()):
    #             print(f"  Predicted as {pred_label}: {count} times ({count/len(class_preds)*100:.2f}%)")

if __name__ == '__main__':
    # classifier = OpenMaxClassifier()
    classifier = ResNetClassifier()
    
    # # 测试未知类
    # print("\n=== Testing Unknown Class Only ===")
    # test_unknown_class(classifier, 'jk_zfls/round0_eval/20')
    
    # 测试所有类别
    print("\n=== Testing All Classes ===")
    test_all_classes(classifier, 'jk_zfls/round0_eval')
