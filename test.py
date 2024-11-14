import torch
from models.resnet import resnet18
import numpy as np
from PIL import Image
from utils.data_stats import load_dataset_stats

class GameClassifier:
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
    
    def predict(self, img_path):
        try:
            # 读取PNG图片
            img = np.array(Image.open(img_path))
            if img.shape != (50, 50, 3):
                raise ValueError(f"Invalid image shape: {img.shape}")
            
            # 转换为batch格式
            img = np.expand_dims(img, axis=0)  # (1, 50, 50, 3)
            
            imgs = torch.from_numpy(img).float()
            imgs = imgs.permute(0, 3, 1, 2)
            imgs = imgs / 255.0
            
            # 使用从数据集计算得到的均值和标准差
            mean = torch.tensor(self.mean).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor(self.std).view(1, 3, 1, 1).to(self.device)
            imgs = (imgs - mean) / std
            
            imgs = imgs.to(self.device)
            
            with torch.no_grad():
                logits, features = self.model(imgs, return_features=True)
                
                # 使用OpenMax进行预测
                probs = self.openmax.predict(features[0].cpu().numpy())
                # 获取最大概率的类别（包括未知类）
                pred = np.argmax(probs)
                if pred == len(probs) - 1:  # 如果是未知类
                    pred = 20  # 设置为第21类
                
                return pred
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

# 使用示例
if __name__ == '__main__':
    # 初始化分类器
    classifier = GameClassifier()
    
    # 测试一张图片
    test_img_path = 'jk_zfls/round0_eval/00/label00_293.png'
    prediction = classifier.predict(test_img_path)
    print(f"Prediction for {test_img_path}: {prediction}")
