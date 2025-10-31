import torch
import torch.nn as nn
import torchvision.models as models

class testNet(nn.Module):
    """
    基于ResNet50的特征提取网络，使用三重注意力机制增强特征表达
    输出维度：2048
    """

    def __init__(self):
        super().__init__()
        # 加载预训练的ResNet50模型
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # 第一层特征提取
        self.first_layer = self.resnet50.conv1
        self.first_bn = self.resnet50.bn1
        self.first_relu = self.resnet50.relu
        self.first_maxpool = self.resnet50.maxpool
        # 最后一层特征提取，去掉最后的全连接层 输出 b 2048 1 1
        self.feature_extractor = nn.Sequential(*list(self.resnet50.children())[:-1])
        #注意力机制模块
        #自适应池化，特征图->b channels 7 7
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        # 1*1卷积核，特征图->b 2048 H W
        self.channel_adjust = nn.Conv2d(64, 2048, kernel_size=1)

    def forward(self, x):
        # 提取第一层特征
        first_features = self.first_layer(x)
        first_features = self.first_bn(first_features)
        first_features = self.first_relu(first_features)
        first_features = self.first_maxpool(first_features)
        # 调整维度为64x7x7
        first_features = self.adaptive_pool(first_features)
        # 调整维度为2048x7x7
        first_features = self.channel_adjust(first_features)
        #池化展平成2048x1x1
        first_features = nn.AdaptiveAvgPool2d((1, 1))(first_features)
        first_features = torch.flatten(first_features, 1)  # 展平为一维向量 [B, 2048]

        # 提取最后层特征
        last_features = self.feature_extractor(x)  # [B, 3, H, W] -> [B, 2048, 1, 1]
        last_features = torch.flatten(last_features, 1)  # 展平为一维向量 [B, 2048]

        return first_features, last_features


# 修改后的CombinedNet，添加隐藏层进行特征融合
class ResEVIQA(nn.Module):
    """
    多模态特征融合网络，结合三种不同的特征提取器
    通过注意力机制进行特征融合，预测图像质量分数
    """

    def __init__(self):
        super().__init__()
        # 初始化三个特征提取网络
        self.test_net = testNet()  # 输出2048维特征

        # 修改全连接层，现在输入是融合后的特征维度
        self.fc = nn.Sequential(
            nn.Linear(4096, 512),  # 调整为融合后的维度
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # 预测质量分数
        )

    def to(self, device):
        """将模型及其所有组件移到指定设备"""
        super().to(device)
        return self

    def forward(self, x):
        """模型前向传播过程"""
        # 提取全局特征
        first_test_features, last_test_features = self.test_net(x)  # [B, 2048]
        fused_features = torch.cat((first_test_features, last_test_features), dim=1)
        # 通过全连接网络预测最终质量分数
        quality_score = self.fc(fused_features)
        return quality_score

if __name__ == "__main__":
    # 创建模型实例
    model = ResEVIQA()
    # 生成随机输入并移动到指定设备
    input_image = torch.randn(2, 3, 224, 224)
    # 前向传播计算质量分数
    x = model(input_image)
    print("计算得到的质量分数:", x)