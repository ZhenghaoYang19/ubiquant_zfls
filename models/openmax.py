import torch
import torch.nn.functional as F
from libmr import MR

class OpenMax:
    def __init__(self, num_classes=20, tailsize=20, alpha=5):
        self.num_classes = num_classes
        self.tailsize = tailsize
        self.alpha = alpha
        self.mavs = None
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
    
    def predict(self, features, logits, multiplier):
        """批量预测
        Args:
            features: torch.Tensor, shape (batch_size, feature_dim)
            logits: torch.Tensor, shape (batch_size, num_classes)
            multiplier: float, Weibull分数的缩放因子
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
        
        # 对每个样本，使用 Weibull 分数对激活值进行调整，降低/增大样本属于未知类别的可能性。
        # 确保 weibull_scores 不会超过1
        weibull_scores = torch.clamp(weibull_scores * multiplier, max=1.0)
        
        # 对每个样本，使用调整后的Weibull分数修改激活值
        for i in range(batch_size):
            # 按距离排序获取前alpha个类别
            sorted_idx = torch.argsort(dists[i])[:self.alpha]
            
            # 修改激活向量
            for j, idx in enumerate(sorted_idx):
                weight = alpha_weights[j] * weibull_scores[i, idx]
                modified_activations[i, idx] *= (1 - weight)
        
        # 计算未知类概率
        unknown_probs = 1 - modified_activations.sum(dim=1, keepdim=True)
        
        # 拼接已知类和未知类的概率
        probs = torch.cat([modified_activations, unknown_probs], dim=1)
        
        return probs
    
    def fit_cosine(self, features, labels):
        """使用余弦距离训练OpenMax模型
        Args:
            features: torch.Tensor, shape (N, feature_dim)
            labels: torch.Tensor, shape (N,)
        """
        device = features.device
        self.mavs = []
        
        # 计算每个类别的MAV
        for c in range(self.num_classes):
            class_features = features[labels == c]
            mav = class_features.mean(dim=0)
            self.mavs.append(mav)
        self.mavs = torch.stack(self.mavs).to(device)
        
        # 为每个类别拟合Weibull分布
        features_norm = F.normalize(features, p=2, dim=1)
        mavs_norm = F.normalize(self.mavs, p=2, dim=1)
        
        self.weibulls = []
        for c in range(self.num_classes):
            class_features = features_norm[labels == c]
            # 计算余弦距离
            cos_sim = torch.mm(class_features, mavs_norm[c:c+1].t())
            dists = (1 - cos_sim).squeeze()
            
            # 拟合Weibull分布
            mr = MR()
            tailtofit = sorted(dists.cpu().numpy())[-self.tailsize:]
            mr.fit_high(tailtofit, self.tailsize)
            self.weibulls.append(mr)
    
    def predict_cosine(self, features, logits, weight_factor=2.0):
        """使用余弦距离的批量预测
        Args:
            features: torch.Tensor, shape (batch_size, feature_dim)
            logits: torch.Tensor, shape (batch_size, num_classes)
            weight_factor: float, 权重调整因子
        Returns:
            torch.Tensor: shape (batch_size, num_classes + 1)
        """
        batch_size = features.shape[0]
        device = features.device
        
        # 确保mavs在正确的设备上
        if self.mavs.device != device:
            self.mavs = self.mavs.to(device)
        
        # 计算softmax
        activations = F.softmax(logits, dim=1)
        modified_activations = activations.clone()
        
        # 特征归一化
        features_norm = F.normalize(features, p=2, dim=1)
        mavs_norm = F.normalize(self.mavs, p=2, dim=1)
        
        # 计算余弦距离
        cos_sim = torch.mm(features_norm, mavs_norm.t())
        dists = 1 - cos_sim
        
        # 计算alpha权重
        alpha_weights = torch.tensor(
            [(self.alpha - i) / self.alpha for i in range(self.alpha)]
        ).to(device)
        
        # 获取每个样本的前alpha个最近类别的索引
        _, top_k_indices = torch.topk(dists, k=self.alpha, dim=1, largest=False)
        
        # 创建batch_indices
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.alpha)
        
        # 计算Weibull分数
        weibull_scores = torch.zeros((batch_size, self.num_classes), device=device)
        for c in range(self.num_classes):
            weibull_scores[:, c] = torch.tensor([
                self.weibulls[c].w_score(d.cpu().item()) 
                for d in dists[:, c]
            ], device=device)
            

        # 获取对应的Weibull分数
        selected_weibull_scores = weibull_scores[batch_indices, top_k_indices]
        
        # 计算权重
        weights = torch.minimum(
            torch.ones_like(selected_weibull_scores),
            selected_weibull_scores * alpha_weights.unsqueeze(0) * weight_factor
        )
        
        # 使用scatter操作更新modified_activations
        weight_matrix = torch.zeros_like(modified_activations)
        weight_matrix.scatter_(1, top_k_indices, weights)
        modified_activations *= (1 - weight_matrix)
        
        # 计算未知类概率
        unknown_probs = 1 - modified_activations.sum(dim=1, keepdim=True)
        # from IPython import embed; embed()
        # 拼接结果
        return torch.cat([modified_activations, unknown_probs], dim=1)