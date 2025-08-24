import os
import torch
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from model import DCENet
from dataset import get_dataloader

try:
    import pyiqa
except ImportError:
    print("Warning: pyiqa not installed. Please install it using: pip install pyiqa")
    pyiqa = None

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

def enhance_image(model, image, device):
    """
    使用模型增强单张图像
    """
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # 添加batch维度
        enhanced_image, _ = model(image)
        enhanced_image = enhanced_image.squeeze(0).cpu()  # 移除batch维度
        
        # 确保像素值在[0,1]范围内
        enhanced_image = torch.clamp(enhanced_image, 0, 1)
        
    return enhanced_image

def evaluate_with_qalign(model, dataloader, device, save_dir=None, qalign_model_path=None):
    """
    使用Q-Align指标评估模型
    """
    if pyiqa is None:
        print("Error: pyiqa is not installed. Cannot compute Q-Align scores.")
        return None
    
    # 初始化Q-Align评估器
    try:
        if qalign_model_path and os.path.exists(qalign_model_path):
            # 使用本地模型路径
            print(f"Loading Q-Align model from local path: {qalign_model_path}")
            # 根据Q-Align文档，支持多种本地模型加载方式
            try:
                # 方法1：直接指定本地路径
                qalign_metric = pyiqa.create_metric('qalign', device=device, pretrained_model_path=qalign_model_path)
                print("Q-Align metric initialized successfully from local model (method 1)")
            except Exception as e1:
                print(f"Method 1 failed: {e1}")
                try:
                    # 方法2：使用model_path参数
                    qalign_metric = pyiqa.create_metric('qalign', device=device, model_path=qalign_model_path)
                    print("Q-Align metric initialized successfully from local model (method 2)")
                except Exception as e2:
                    print(f"Method 2 failed: {e2}")
                    # 方法3：设置环境变量后加载
                    import os
                    old_cache_dir = os.environ.get('TRANSFORMERS_CACHE', '')
                    os.environ['TRANSFORMERS_CACHE'] = qalign_model_path
                    qalign_metric = pyiqa.create_metric('qalign', device=device)
                    if old_cache_dir:
                        os.environ['TRANSFORMERS_CACHE'] = old_cache_dir
                    else:
                        os.environ.pop('TRANSFORMERS_CACHE', None)
                    print("Q-Align metric initialized successfully from local model (method 3)")
        else:
            # 尝试在线下载（原有逻辑）
            print("Attempting to download Q-Align model online...")
            print("Warning: 中国网络可能无法访问Hugging Face，建议使用 --qalign_model_path 指定本地模型")
            qalign_metric = pyiqa.create_metric('qalign', device=device)
            print("Q-Align metric initialized successfully")
    except Exception as e:
        print(f"Error initializing Q-Align metric: {e}")
        if qalign_model_path:
            print("Failed to load from local path, trying alternative initialization...")
        else:
            print("Trying alternative initialization...")
        try:
            if qalign_model_path and os.path.exists(qalign_model_path):
                qalign_metric = pyiqa.create_metric('qalign', pretrained_model_path=qalign_model_path)
                print("Q-Align metric initialized with CPU from local model")
            else:
                qalign_metric = pyiqa.create_metric('qalign')
                print("Q-Align metric initialized with CPU")
        except Exception as e2:
            print(f"Failed to initialize Q-Align metric: {e2}")
            if qalign_model_path:
                print("Suggestion: Please ensure the local Q-Align model path is correct and the model files are complete.")
            else:
                print("Suggestion: Please download Q-Align model locally and use --qalign_model_path parameter.")
            return None
    
    model.eval()
    scores = []
    
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'original'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'enhanced'), exist_ok=True)
    
    # 图像转换器（用于保存图像）
    to_pil = transforms.ToPILImage()
    
    with torch.no_grad():
        for batch_idx, (images, img_ids) in enumerate(tqdm(dataloader, desc='Evaluating')):
            images = images.to(device)
            
            # 增强图像
            enhanced_images, _ = model(images)
            enhanced_images = torch.clamp(enhanced_images, 0, 1)
            
            # 计算每张图像的Q-Align分数
            for i in range(images.size(0)):
                try:
                    # 获取单张图像
                    original_img = images[i]
                    enhanced_img = enhanced_images[i]
                    img_id = img_ids[i]
                    
                    # 计算Q-Align分数（只对增强后的图像）
                    score = qalign_metric(enhanced_img.unsqueeze(0)).item()
                    scores.append(score)
                    
                    # 保存图像（可选）
                    if save_dir:
                        # 保存原图
                        original_pil = to_pil(original_img.cpu())
                        original_pil.save(os.path.join(save_dir, 'original', f'{img_id:05d}.png'))
                        
                        # 保存增强图
                        enhanced_pil = to_pil(enhanced_img.cpu())
                        enhanced_pil.save(os.path.join(save_dir, 'enhanced', f'{img_id:05d}.png'))
                    
                except Exception as e:
                    print(f"Error processing image {img_id}: {e}")
                    continue
            
            # 限制评估数量（可选，用于快速测试）
            if batch_idx >= 50:  # 只评估前50个batch
                break
    
    if scores:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"\nQ-Align Evaluation Results:")
        print(f"Number of images: {len(scores)}")
        print(f"Mean Q-Align score: {mean_score:.4f} ± {std_score:.4f}")
        print(f"Min score: {np.min(scores):.4f}")
        print(f"Max score: {np.max(scores):.4f}")
        
        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'num_images': len(scores),
            'all_scores': scores
        }
    else:
        print("No valid scores computed.")
        return None

def evaluate_without_qalign(model, dataloader, device, save_dir=None):
    """
    不使用Q-Align的基础评估（仅保存增强结果）
    """
    model.eval()
    
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'original'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'enhanced'), exist_ok=True)
    
    # 图像转换器
    to_pil = transforms.ToPILImage()
    
    processed_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, img_ids) in enumerate(tqdm(dataloader, desc='Processing')):
            images = images.to(device)
            
            # 增强图像
            enhanced_images, _ = model(images)
            enhanced_images = torch.clamp(enhanced_images, 0, 1)
            
            # 保存图像
            for i in range(images.size(0)):
                original_img = images[i]
                enhanced_img = enhanced_images[i]
                img_id = img_ids[i]
                
                if save_dir:
                    # 保存原图
                    original_pil = to_pil(original_img.cpu())
                    original_pil.save(os.path.join(save_dir, 'original', f'{img_id:05d}.png'))
                    
                    # 保存增强图
                    enhanced_pil = to_pil(enhanced_img.cpu())
                    enhanced_pil.save(os.path.join(save_dir, 'enhanced', f'{img_id:05d}.png'))
                
                processed_count += 1
    
    print(f"\nProcessed {processed_count} images")
    if save_dir:
        print(f"Results saved to {save_dir}")
    
    return {'num_images': processed_count}

def main():
    parser = argparse.ArgumentParser(description='Zero-DCE Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='LLIEDet/images', help='Path to image directory')
    parser.add_argument('--val_annotation', type=str, default='LLIEDet/coco_annotations/val.json', help='Path to validation annotation file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--use_qalign', action='store_true', help='Use Q-Align metric for evaluation')
    parser.add_argument('--qalign_model_path', type=str, default=None, help='Path to local Q-Align model directory (to avoid downloading from Hugging Face)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 创建数据加载器
    val_dataloader = get_dataloader(
        data_dir=args.data_dir,
        annotation_file=args.val_annotation,
        batch_size=args.batch_size,
        shuffle=False,
        image_size=args.image_size
    )
    
    print(f'Validation dataset size: {len(val_dataloader.dataset)}')
    
    # 评估模型
    if args.use_qalign and pyiqa is not None:
        results = evaluate_with_qalign(model, val_dataloader, device, args.save_dir, args.qalign_model_path)
    else:
        if args.use_qalign:
            print("Warning: Q-Align evaluation requested but pyiqa not available. Using basic evaluation.")
        results = evaluate_without_qalign(model, val_dataloader, device, args.save_dir)
    
    print("\nEvaluation completed!")

if __name__ == '__main__':
    main()