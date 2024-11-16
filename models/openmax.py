import torch
import torch.nn.functional as F
from libmr import MR

class OpenMax:
    def __init__(self, num_classes=20, tailsize=20, alpha=10, reg_lambda=0.1):
        self.num_classes = num_classes
        self.tailsize = tailsize
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.mavs = None
        self.weibulls = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, logits, labels):
        """训练OpenMax模型
        Args:
            logits: torch.Tensor, 模型最后一层的输出 (N, num_classes)
            labels: torch.Tensor, 标签 (N,)
        """
        # 确保数据在正确的设备上
        logits = logits.to(self.device)
        labels = labels.to(self.device)
        
        self.mavs = []
        # 计算每个类别的均值激活向量(MAV)
        for c in range(self.num_classes):
            class_logits = logits[labels == c]
            mav = class_logits.mean(dim=0)
            mav = mav / (1 + self.reg_lambda * torch.norm(mav))
            self.mavs.append(mav)
        self.mavs = torch.stack(self.mavs).to(self.device)  # (num_classes, num_classes)
    
        # 为每个类别拟合Weibull分布
        for c in range(self.num_classes):
            class_logits = logits[labels == c]
            # 计算到MAV的欧氏距离
            dists = torch.cdist(class_logits, self.mavs[c:c+1])
            # 拟合Weibull分布（这部分仍需要numpy因为libmr没有cuda实现）
            mr = MR()
            tailtofit = sorted(dists.cpu().numpy())[-self.tailsize:]
            mr.fit_high(tailtofit, self.tailsize)
            self.weibulls.append(mr)
    
    def predict(self, logits):
        """批量预测多个样本
        Args:
            logits: torch.Tensor, shape (batch_size, num_classes)
        Returns:
            torch.Tensor: shape (batch_size, num_classes + 1) 包含未知类的概率
        """
        # 确保logits和mavs在同一个设备上
        logits = logits.to(self.device)
        if self.mavs is not None:
            self.mavs = self.mavs.to(self.device)
        
        batch_size = logits.shape[0]
        
        # 计算到每个类别MAV的距离
        dists = torch.cdist(logits, self.mavs)  # shape: (batch_size, num_classes)
        
        # 初始化批量预测结果
        modified_activations = F.softmax(logits, dim=1).clone()   # shape: (batch_size, num_classes)
        
        # 计算alpha权重
        alpha_weights = torch.tensor(
            [(self.alpha - i) / self.alpha for i in range(self.alpha)]
        ).to(logits.device)
        
        # 批量计算Weibull分数
        weibull_scores = torch.zeros((batch_size, self.num_classes)).to(logits.device)
        for c in range(self.num_classes):
            weibull_scores[:, c] = torch.tensor([
                self.weibulls[c].w_score(dist.item()) 
                for dist in dists[:, c]
            ]).to(logits.device)
        
        # 对每个样本进行处理
        for i in range(batch_size):
            # 按距离排序获取前alpha个类别
            sorted_idx = torch.argsort(dists[i])[:self.alpha]
            
            # 修改激活向量，使用更温和的衰减
            for j, idx in enumerate(sorted_idx):
                weight = alpha_weights[j] * weibull_scores[i, idx]
                weight = torch.clamp(weight, 0, 0.9)  # 限制最大衰减
                modified_activations[i, idx] *= (1 - weight)
        
        # 调整未知类的概率计算，使用更严格的阈值
        unknown_probs = torch.clamp(1 - modified_activations.sum(dim=1, keepdim=True), 0, 0.7)  # 降低未知类的最大概率
        
        # 提高高置信度预测的阈值，更大程度降低其被判为未知类的可能性
        max_probs, _ = modified_activations.max(dim=1)
        high_conf_mask = (max_probs > 0.7).unsqueeze(1)  # 降低高置信度的阈值
        unknown_probs = torch.where(high_conf_mask, unknown_probs * 0.3, unknown_probs)  # 更大程度降低未知类概率
        
        # 对已知类的概率进行boost
        modified_activations = modified_activations * 1.2  # 提升已知类的概率
        modified_activations = torch.clamp(modified_activations, 0, 1)  # 确保不超过1
        
        # 拼接已知类和未知类的概率
        probs = torch.cat([modified_activations, unknown_probs], dim=1)
        
        return probs