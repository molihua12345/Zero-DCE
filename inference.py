import os
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import DCENet

def load_model(checkpoint_path, device):
    """
    加载训练好的模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取模型参数
    if 'args' in checkpoint:
        n_iter = checkpoint['args'].n_iter if hasattr(checkpoint['args'], 'n_iter') else 8
    else:
        n_iter = 8
    
    # 创建模型
    model = DCENet(n_iter=n_iter).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Loss: {checkpoint.get('loss', 'Unknown')}")
    
    return model

def enhance_image(model, image_path, device, image_size=256):
    """
    增强单张图像
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 模型推理
    model.eval()
    with torch.no_grad():
        enhanced_tensor, _ = model(input_tensor)
        enhanced_tensor = torch.clamp(enhanced_tensor, 0, 1)
    
    # 转换回PIL图像
    to_pil = transforms.ToPILImage()
    enhanced_image = to_pil(enhanced_tensor.squeeze(0).cpu())
    
    # 恢复原始尺寸
    enhanced_image = enhanced_image.resize(original_size, Image.LANCZOS)
    
    return image, enhanced_image

def save_comparison(original, enhanced, save_path):
    """
    保存对比图像
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(enhanced)
    axes[1].set_title('Enhanced Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Zero-DCE Image Enhancement Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to save enhanced image')
    parser.add_argument('--comparison', type=str, help='Path to save comparison image')
    parser.add_argument('--image_size', type=int, default=256, help='Processing image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"Error: Input image {args.input} not found!")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found!")
        return
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 增强图像
    print(f"Processing image: {args.input}")
    original_image, enhanced_image = enhance_image(model, args.input, device, args.image_size)
    
    # 保存增强后的图像
    if args.output:
        enhanced_image.save(args.output)
        print(f"Enhanced image saved to {args.output}")
    else:
        # 默认保存路径
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"{input_name}_enhanced.png"
        enhanced_image.save(output_path)
        print(f"Enhanced image saved to {output_path}")
    
    # 保存对比图像
    if args.comparison:
        save_comparison(original_image, enhanced_image, args.comparison)
    else:
        # 默认保存路径
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        comparison_path = f"{input_name}_comparison.png"
        save_comparison(original_image, enhanced_image, comparison_path)
    
    print("Inference completed!")

if __name__ == '__main__':
    main()