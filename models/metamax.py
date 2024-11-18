import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MetaClassifier(nn.Module):
    """改进的元分类器网络结构"""
    def __init__(self, in_features=512, hidden_dim=256, num_classes=21):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 残差连接的映射层
        self.shortcut = nn.Sequential(
            nn.Linear(in_features, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2)
        )
        
        # 最终分类层
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        # 主路径
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # 残差连接
        residual = self.shortcut(x)
        out = out + residual
        
        # 最终分类
        out = self.classifier(out)
        return out

class MetaMax:
    def __init__(self, num_classes=20, meta_ratio=0.2):
        self.num_classes = num_classes
        self.meta_ratio = meta_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用改进的元分类器
        self.meta_classifier = MetaClassifier(
            in_features=512,
            hidden_dim=256,
            num_classes=num_classes + 1  # +1 表示未知类别
        ).to(self.device)
        
        self.known_classes = set(range(num_classes))
        self.pseudo_unknown_classes = set()
        
    def select_pseudo_unknown(self):
        """随机选择一部分类别作为伪未知类"""
        num_unknown = int(self.num_classes * self.meta_ratio)
        self.pseudo_unknown_classes = set(random.sample(range(self.num_classes), num_unknown))
        self.known_classes = set(range(self.num_classes)) - self.pseudo_unknown_classes
        
    def fit(self, features, labels):
        """训练MetaMax模型"""
        self.meta_classifier.train()
        
        # 将数据移到设备上
        features = features.to(self.device)
        labels = labels.to(self.device)
        
        # 创建优化器
        optimizer = torch.optim.Adam(self.meta_classifier.parameters(), lr=0.001)
        
        # 训练多个epoch，每次重新选择伪未知类
        num_epochs = 5
        batch_size = 128
        
        for epoch in range(num_epochs):
            # 重新选择伪未知类
            self.select_pseudo_unknown()
            
            # 准备训练数据
            known_mask = torch.tensor([label.item() not in self.pseudo_unknown_classes 
                                    for label in labels])  
            unknown_mask = ~known_mask
            
            # 创建新的标签，将伪未知类标记为num_classes（最后一类）
            new_labels = labels.clone()
            new_labels[unknown_mask] = self.num_classes
            
            # 批量训练
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size]
                batch_labels = new_labels[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.meta_classifier(batch_features)
                loss = F.cross_entropy(outputs, batch_labels)
                loss.backward()
                optimizer.step()
    
    def predict(self, features, logits=None):
        """预测包括未知类的概率分布"""
        self.meta_classifier.eval()
        features = features.to(self.device)
        
        with torch.no_grad():
            outputs = self.meta_classifier(features)
            probs = F.softmax(outputs, dim=1)
            
        return probs
