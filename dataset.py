import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class LLIEDetDataset(Dataset):
    """
    LLIEDet数据集加载器
    用于加载低光照图像数据
    """
    def __init__(self, data_dir, annotation_file, transform=None, image_size=256):
        """
        Args:
            data_dir: 图像数据目录路径
            annotation_file: 标注文件路径 (train.json 或 val.json)
            transform: 图像变换
            image_size: 图像尺寸
        """
        self.data_dir = data_dir
        self.image_size = image_size
        
        # 加载标注文件
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # 获取所有图像ID
        self.image_ids = list(set([ann['image_id'] for ann in self.annotations['annotations']]))
        
        # 创建图像ID到文件名的映射
        self.id_to_filename = {}
        for img_info in self.annotations.get('images', []):
            self.id_to_filename[img_info['id']] = img_info['file_name']
        
        # 如果没有images字段，根据image_id推断文件名
        if not self.id_to_filename:
            for img_id in self.image_ids:
                # 查找对应的图像文件
                possible_extensions = ['.jpg', '.png', '.jpeg']
                for ext in possible_extensions:
                    filename = f"2015_{img_id:05d}{ext}"
                    if os.path.exists(os.path.join(data_dir, filename)):
                        self.id_to_filename[img_id] = filename
                        break
        
        # 设置默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                # 不进行归一化，保持[0,1]范围
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # 获取图像ID
        img_id = self.image_ids[idx]
        
        # 获取图像文件名
        if img_id in self.id_to_filename:
            filename = self.id_to_filename[img_id]
        else:
            # 如果没有找到映射，尝试常见的命名格式
            possible_extensions = ['.jpg', '.png', '.jpeg']
            filename = None
            for ext in possible_extensions:
                test_filename = f"2015_{img_id:05d}{ext}"
                if os.path.exists(os.path.join(self.data_dir, test_filename)):
                    filename = test_filename
                    break
            
            if filename is None:
                raise FileNotFoundError(f"Cannot find image file for ID {img_id}")
        
        # 加载图像
        img_path = os.path.join(self.data_dir, filename)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个默认的黑色图像
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, img_id

def get_dataloader(data_dir, annotation_file, batch_size=8, shuffle=True, num_workers=4, image_size=256):
    """
    创建数据加载器
    """
    dataset = LLIEDetDataset(
        data_dir=data_dir,
        annotation_file=annotation_file,
        image_size=image_size
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

if __name__ == "__main__":
    # 测试数据加载器
    data_dir = "LLIEDet/images"
    train_annotation = "LLIEDet/coco_annotations/train.json"
    
    if os.path.exists(data_dir) and os.path.exists(train_annotation):
        # 创建数据集
        dataset = LLIEDetDataset(data_dir, train_annotation)
        print(f"Dataset size: {len(dataset)}")
        
        # 测试加载一个样本
        if len(dataset) > 0:
            image, img_id = dataset[0]
            print(f"Image shape: {image.shape}")
            print(f"Image ID: {img_id}")
            print(f"Image value range: [{image.min():.3f}, {image.max():.3f}]")
        
        # 创建数据加载器
        dataloader = get_dataloader(data_dir, train_annotation, batch_size=4)
        
        # 测试批量加载
        for batch_idx, (images, img_ids) in enumerate(dataloader):
            print(f"Batch {batch_idx}: {images.shape}")
            if batch_idx >= 2:  # 只测试前3个批次
                break
    else:
        print("Data directory or annotation file not found!")
        print(f"Looking for: {data_dir} and {train_annotation}")