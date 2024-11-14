import torch
import torch.nn.functional as F
from libmr import MR

class OpenMax:
    def __init__(self, num_classes=20, tailsize=20, alpha=10):
        self.num_classes = num_classes
        self.tailsize = tailsize
        self.alpha = alpha
        self.mavs = None
        self.weibulls = []
        
    def fit(self, logits, labels):
        """训练OpenMax模型
        Args:
            logits: torch.Tensor, 模型最后一层的输出 (N, num_classes)
            labels: torch.Tensor, 标签 (N,)
        """
        self.mavs = []
        # 计算每个类别的均值激活向量(MAV)
        for c in range(self.num_classes):
            class_logits = logits[labels == c]
            mav = class_logits.mean(dim=0)
            self.mavs.append(mav)
        self.mavs = torch.stack(self.mavs)  # (num_classes, num_classes)
        
        # 为每个类别拟合Weibull分布
        for c in range(self.num_classes):
            class_logits = logits[labels == c]
            # 计算到MAV的欧氏距离
            dists = torch.cdist(class_logits, self.mavs[c:c+1])
            # 拟合Weibull分布（这部分仍需要numpy因为libmr的限制）
            mr = MR()
            tailtofit = sorted(dists.cpu().numpy())[-self.tailsize:]
            mr.fit_high(tailtofit, self.tailsize)
            self.weibulls.append(mr)
    
    def predict(self, logit):
        """预测单个样本
        Args:
            logit: torch.Tensor, shape (num_classes,)
        Returns:
            torch.Tensor: shape (num_classes + 1,) 包含未知类的概率
        """
        # 计算到每个类别MAV的距离
        dists = torch.cdist(logit.unsqueeze(0), self.mavs)[0]
        
        # 计算Weibull分数（仍需要numpy）
        weibull_scores = torch.zeros(self.num_classes)
        for c in range(self.num_classes):
            weibull_scores[c] = self.weibulls[c].w_score(dists[c].item())
        
        # 计算OpenMax概率
        alpha_weights = torch.tensor(
            [(self.alpha - i) / self.alpha for i in range(self.alpha)]
        )
        
        # 按距离排序获取前alpha个类别
        sorted_idx = torch.argsort(dists)[:self.alpha]
        
        # 初始化修改后的激活向量
        modified_activation = logit.clone()
        for i, idx in enumerate(sorted_idx):
            modified_activation[idx] *= (1 - alpha_weights[i] * weibull_scores[idx])
        
        # 添加未知类的概率
        unknown_prob = 1 - modified_activation.sum()
        probs = torch.cat([modified_activation, unknown_prob.unsqueeze(0)])
        
        return probs