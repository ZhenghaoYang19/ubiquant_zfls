import torch
import torch.nn as nn
from utils.data_stats import load_dataset_stats, calculate_dataset_stats
from torchvision import transforms
import os
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
    
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out
    
# 定义 ResNet 结构
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=20, dropout_rate=0.5, base_width=64):
        super(ResNet, self).__init__()
        self.in_channels = base_width
        self.dropout_rate = dropout_rate
        
        # 输入图片较小(50x50)，减小kernel_size和stride
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 根据不同模型调整通道数
        self.layer1 = self._make_layer(block, base_width, layers[0], stride=1)
        self.dropout1 = nn.Dropout(p=dropout_rate/2)
        
        self.layer2 = self._make_layer(block, base_width*2, layers[1], stride=2)
        self.dropout2 = nn.Dropout(p=dropout_rate/2)
        
        self.layer3 = self._make_layer(block, base_width*4, layers[2], stride=2)
        self.dropout3 = nn.Dropout(p=dropout_rate/2)
        
        self.layer4 = self._make_layer(block, base_width*8, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        # 根据base_width调整最终特征维度
        final_channels = base_width * 8 * block.expansion
        self.fc = nn.Linear(final_channels, num_classes)
        
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
    def forward(self, x, return_features=False):
        # 输入x的形状为 [batch_size, 3, 50, 50]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.dropout2(x)
        
        x = self.layer3(x)
        x = self.dropout3(x)
        
        x = self.layer4(x)
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        features = self.dropout(features)
        logits = self.fc(features)
        
        if return_features:
            return logits, features
        return logits

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

def resnet18(num_classes=21, dropout_rate=0.5):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dropout_rate=dropout_rate, base_width=64)

def resnet34(num_classes=21, dropout_rate=0.5):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, dropout_rate=dropout_rate, base_width=96)

def resnet50(num_classes=21, dropout_rate=0.5):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, dropout_rate=dropout_rate, base_width=128)

# 使用示例
class ImageClassifier:
    def __init__(self, model_type, model_path, openmax_path, multiplier=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images_path = os.path.join('jk_zfls', 'round0_train')
        # 尝试加载已保存的数据集统计信息，如果不存在则重新计算
        try:
            self.mean, self.std = load_dataset_stats()
            print("Loaded pre-calculated dataset statistics")
        except FileNotFoundError:
            print("FileNotFound, Calculating dataset statistics...")
            self.mean, self.std = calculate_dataset_stats(images_path)
        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # 加载模型
        if model_type == 'resnet18':
            self.model = resnet18(num_classes=20)
        elif model_type == 'resnet34':
            self.model = resnet34(num_classes=20)
        elif model_type == 'resnet50':
            self.model = resnet50(num_classes=20)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 加载OpenMax模型
        self.openmax = torch.load(openmax_path)
        self.multiplier = multiplier
        
    def resnet_predict(self, patches):
        """
        仅使用ResNet模型预测图像块的类别（不使用OpenMax）
        Args:
            patches: shape (144, 50, 50, 3) 的numpy array
        Returns:
            predictions: shape (144,) 的类别预测结果
        """
        with torch.no_grad():
            # 直接转换为tensor并归一化 (144,3,50,50)
            patches_tensor = (torch.from_numpy(patches).float().permute(0, 3, 1, 2) / 255.0).to(self.device)
            patches_tensor = transforms.Normalize(mean=self.mean, std=self.std)(patches_tensor)
            
            # 一次性获取所有预测结果
            logits, _ = self.model(patches_tensor, return_features=True)
            predictions = torch.argmax(logits, dim=1)

        return predictions.cpu()
    
    def predict(self, patches):
        """
        使用OpenMax预测图像块的类别
        Args:
            patches: shape (144, 50, 50, 3) 的numpy array
        Returns:
            predictions: shape (144,) 的类别预测结果
        """
        # 一次性处理所有144个patches
        with torch.no_grad():
            # 直接转换为tensor并归一化 (144,3,50,50)
            patches_tensor = (torch.from_numpy(patches).float().permute(0, 3, 1, 2) / 255.0).to(self.device)
            patches_tensor = transforms.Normalize(mean=self.mean, std=self.std)(patches_tensor)
            
            # 一次性获取所有预测结果
            logits, features = self.model(patches_tensor, return_features=True)
            openmax_probs = self.openmax.predict(features, logits, multiplier=self.multiplier)
            predictions = torch.argmax(openmax_probs, dim=1)

        return predictions.cpu(), openmax_probs

