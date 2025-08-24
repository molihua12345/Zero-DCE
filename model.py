import torch
import torch.nn as nn
import torch.nn.functional as F

class DCENet(nn.Module):
    """
    Zero-DCE网络模型
    基于论文: Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
    """
    def __init__(self, n_iter=8):
        super(DCENet, self).__init__()
        self.n_iter = n_iter
        
        # 7层对称卷积网络，不包含下采样操作
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(32, 3 * n_iter, kernel_size=3, stride=1, padding=1, bias=True)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入的低光图像 (batch, 3, H, W)
        Returns:
            enhanced_image: 增强后的图像
            A_list: 曲线参数图列表
        """
        # 通过7层卷积网络提取特征
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.conv6(x5))
        x_r = torch.tanh(self.conv7(x6))  # 输出范围限制在[-1, 1]
        
        # 将输出通道拆分成n_iter个参数图
        A_list = torch.split(x_r, 3, dim=1)
        
        # 迭代应用增强公式
        enhanced_image = x
        for A in A_list:
            # 核心增强公式: E(x) = L(x) + A(x) * L(x) * (1 - L(x))
            enhanced_image = enhanced_image + A * (torch.pow(enhanced_image, 2) - enhanced_image)
        
        return enhanced_image, A_list

if __name__ == "__main__":
    # 测试模型
    model = DCENet()
    x = torch.randn(1, 3, 256, 256)
    enhanced, A_list = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Enhanced shape: {enhanced.shape}")
    print(f"Number of curve parameter maps: {len(A_list)}")
    print(f"Each parameter map shape: {A_list[0].shape}")