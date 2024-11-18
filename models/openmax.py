import torch
import torch.nn.functional as F
from libmr import MR

class OpenMax:
    def __init__(self, num_classes=20, tailsize=20, alpha=10):
        self.num_classes = num_classes
        self.tailsize = tailsize
        self.alpha = alpha
        self.mavs = None  # Mean Activation Vectors
        self.weibulls = []
        
    def fit(self, features, labels):
        """训练OpenMax模型
        Args:
            features: torch.Tensor, 模型特征层输出 (N, feature_dim)
            labels: torch.Tensor, 标签 (N,)
        """
        self.mavs = []
        
        # 计算每个类别的均值激活向量(MAV)
        for c in range(self.num_classes):
            class_features = features[labels == c]
            mav = class_features.mean(dim=0)
            self.mavs.append(mav)
            
        self.mavs = torch.stack(self.mavs)
        
        # 为每个类别拟合Weibull分布
        for c in range(self.num_classes):
            class_features = features[labels == c]
            # 在同一设备上计算距离
            dists = torch.cdist(class_features, self.mavs[c:c+1])
            # 拟合Weibull分布（这部分仍需要numpy因为libmr没有cuda实现）
            mr = MR()
            tailtofit = sorted(dists.cpu().numpy())[-self.tailsize:]
            mr.fit_high(tailtofit, self.tailsize)
            self.weibulls.append(mr)
    
    def predict(self, features, logits):
        """批量预测
        Args:
            features: torch.Tensor, shape (batch_size, feature_dim)
            logits: torch.Tensor, shape (batch_size, num_classes)
        Returns:
            torch.Tensor: shape (batch_size, num_classes + 1)
        """
        batch_size = features.shape[0]
        device = features.device
        
        # 确保mavs在正确的设备上
        if self.mavs.device != device:
            self.mavs = self.mavs.to(device)
        
        # 获取原始激活值
        activations = F.softmax(logits, dim=1)
        modified_activations = activations.clone()
        
        # 计算到每个类别MAV的距离
        dists = torch.cdist(features, self.mavs)
        
        # 计算alpha权重
        alpha_weights = torch.tensor(
            [(self.alpha - i) / self.alpha for i in range(self.alpha)]
        ).to(device)
        
        # 批量计算Weibull分数
        weibull_scores = torch.zeros((batch_size, self.num_classes)).to(device)
        for c in range(self.num_classes):
            # 只在需要用到numpy时才转到CPU
            weibull_scores[:, c] = torch.tensor([
                self.weibulls[c].w_score(dist.item()) 
                for dist in dists[:, c]
            ]).to(device)
        
        # 对每个样本，使用 Weibull 分数对激活值进行调整，降低样本属于未知类别的可能性。
        for i in range(batch_size):
            # 按距离排序获取前alpha个类别
            sorted_idx = torch.argsort(dists[i])[:self.alpha]
            
            # 修改激活向量，使用更温和的衰减
            for j, idx in enumerate(sorted_idx):
                weight = alpha_weights[j] * weibull_scores[i, idx]
                modified_activations[i, idx] *= (1 - weight)
        
        # 计算未知类概率
        unknown_probs = 1 - modified_activations.sum(dim=1, keepdim=True)
        
        # 拼接已知类和未知类的概率
        probs = torch.cat([modified_activations, unknown_probs], dim=1)
        
        return probs