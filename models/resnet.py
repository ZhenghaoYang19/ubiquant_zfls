import torch
import torch.nn as nn

# 定义残差块
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 判断是否需要跳跃连接的调整
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out
    
# 定义 ResNet-18 结构
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=20, dropout_rate=0.5):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.dropout_rate = dropout_rate
        
        # 输入图片较小(50x50)，减小kernel_size和stride
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 移除maxpool层，因为输入图片较小
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 在全连接层前添加dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 添加特征归一化
        self.feature_norm = nn.LayerNorm(512 * block.expansion)
        
        # 在模型初始化时就将所有参数移到GPU
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
    def forward(self, x, return_logits=False):
        # 输入x的形状为 [batch_size, 3, 50, 50]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        features = self.feature_norm(features)  
        # 在全连接层前使用dropout
        features = self.dropout(features)
        logits = self.fc(features)
        
        # 调整logits的scale
        # logits = logits * 10  # 增大logits的scale以产生更明显的区分
        
        probs = torch.softmax(logits, dim=1)
        
        if return_logits:
            return probs, logits
        else:
            return probs

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

def resnet18(num_classes=21, dropout_rate=0.5):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dropout_rate=0.5)

# 使用示例
def process_batch_images(imgs):
    """
    处理一批图像
    Args:
        imgs: shape [144, 50, 50, 3] 的张量
    Returns:
        predictions: shape [12, 12] 的分类结果
    """
    # 转换为PyTorch期望的格式 [144, 3, 50, 50]
    imgs = imgs.permute(0, 3, 1, 2)
    
    # 实例化模型
    model = resnet18()
    
    # 前向传播
    with torch.no_grad():
        outputs = model(imgs)  # outputs shape: [144, 21]
    
    # 获取最可能的类别
    _, predicted = torch.max(outputs, 1)
    
    # 重塑为12x12网格
    predictions = predicted.reshape(12, 12)
    
    return predictions
