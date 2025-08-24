import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroDCELoss(nn.Module):
    """
    Zero-DCE的无参考损失函数
    包含四个损失项：空间一致性、曝光控制、色彩恒常性、光照平滑
    """
    def __init__(self, spa_weight=1, exp_weight=10, col_weight=5, tv_weight=1600):
        super(ZeroDCELoss, self).__init__()
        self.spa_weight = spa_weight
        self.exp_weight = exp_weight
        self.col_weight = col_weight
        self.tv_weight = tv_weight
        
    def forward(self, input_img, enhanced_img, A_list):
        """
        计算总损失
        Args:
            input_img: 原始低光图像
            enhanced_img: 增强后图像
            A_list: 曲线参数图列表
        """
        # 空间一致性损失
        L_spa = self.spatial_consistency_loss(input_img, enhanced_img)
        
        # 曝光控制损失
        L_exp = self.exposure_control_loss(enhanced_img)
        
        # 色彩恒常性损失
        L_col = self.color_constancy_loss(enhanced_img)
        
        # 光照平滑损失
        L_tv = self.illumination_smoothness_loss(A_list)
        
        # 总损失
        total_loss = (self.spa_weight * L_spa + 
                     self.exp_weight * L_exp + 
                     self.col_weight * L_col + 
                     self.tv_weight * L_tv)
        
        return total_loss, L_spa, L_exp, L_col, L_tv
    
    def spatial_consistency_loss(self, input_img, enhanced_img):
        """
        空间一致性损失：保持增强前后图像的空间结构一致性
        """
        # 计算原图的梯度
        input_grad_h = torch.abs(input_img[:, :, :, 1:] - input_img[:, :, :, :-1])
        input_grad_v = torch.abs(input_img[:, :, 1:, :] - input_img[:, :, :-1, :])
        
        # 计算增强图的梯度
        enhanced_grad_h = torch.abs(enhanced_img[:, :, :, 1:] - enhanced_img[:, :, :, :-1])
        enhanced_grad_v = torch.abs(enhanced_img[:, :, 1:, :] - enhanced_img[:, :, :-1, :])
        
        # 计算梯度差异
        loss_h = torch.mean(torch.abs(input_grad_h - enhanced_grad_h))
        loss_v = torch.mean(torch.abs(input_grad_v - enhanced_grad_v))
        
        return loss_h + loss_v
    
    def exposure_control_loss(self, enhanced_img, patch_size=16, mean_val=0.6):
        """
        曝光控制损失：控制图像的局部曝光水平
        """
        # 转换为灰度图
        gray_img = 0.299 * enhanced_img[:, 0, :, :] + 0.587 * enhanced_img[:, 1, :, :] + 0.114 * enhanced_img[:, 2, :, :]
        gray_img = gray_img.unsqueeze(1)
        
        # 使用平均池化计算局部平均亮度
        avg_pool = nn.AvgPool2d(patch_size)
        avg_brightness = avg_pool(gray_img)
        
        # 计算与理想曝光值的差异
        loss = torch.mean(torch.pow(avg_brightness - mean_val, 2))
        
        return loss
    
    def color_constancy_loss(self, enhanced_img):
        """
        色彩恒常性损失：保持色彩平衡
        """
        # 计算RGB三个通道的全局平均值
        mean_r = torch.mean(enhanced_img[:, 0, :, :])
        mean_g = torch.mean(enhanced_img[:, 1, :, :])
        mean_b = torch.mean(enhanced_img[:, 2, :, :])
        
        # 计算通道间的差异
        loss_rg = torch.pow(mean_r - mean_g, 2)
        loss_gb = torch.pow(mean_g - mean_b, 2)
        loss_rb = torch.pow(mean_r - mean_b, 2)
        
        return loss_rg + loss_gb + loss_rb
    
    def illumination_smoothness_loss(self, A_list):
        """
        光照平滑损失：确保曲线参数图的平滑性
        """
        loss = 0
        for A in A_list:
            # 计算水平和垂直方向的梯度
            grad_h = torch.pow(A[:, :, :, 1:] - A[:, :, :, :-1], 2)
            grad_v = torch.pow(A[:, :, 1:, :] - A[:, :, :-1, :], 2)
            
            # 累加梯度损失
            loss += torch.mean(grad_h) + torch.mean(grad_v)
        
        return loss

if __name__ == "__main__":
    # 测试损失函数
    criterion = ZeroDCELoss()
    
    # 创建测试数据
    input_img = torch.randn(2, 3, 256, 256)
    enhanced_img = torch.randn(2, 3, 256, 256)
    A_list = [torch.randn(2, 3, 256, 256) for _ in range(8)]
    
    # 计算损失
    total_loss, L_spa, L_exp, L_col, L_tv = criterion(input_img, enhanced_img, A_list)
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Spatial Consistency Loss: {L_spa.item():.4f}")
    print(f"Exposure Control Loss: {L_exp.item():.4f}")
    print(f"Color Constancy Loss: {L_col.item():.4f}")
    print(f"Illumination Smoothness Loss: {L_tv.item():.4f}")