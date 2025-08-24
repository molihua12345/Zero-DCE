#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的Zero-DCE训练脚本
包含高级优化功能：混合精度训练、梯度裁剪、早停机制、多种学习率调度器等
"""

import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import argparse
from tqdm import tqdm
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from model import DCENet
from loss import ZeroDCELoss
from dataset import get_dataloader

def save_sample_images(model, sample_batch, epoch, save_dir, device):
    """
    保存样本图像对比
    """
    model.eval()
    with torch.no_grad():
        sample_images = sample_batch.to(device)
        enhanced_images, _ = model(sample_images)
        
        # 转换为PIL图像并保存
        for i in range(min(4, sample_images.size(0))):
            # 原图
            orig_img = sample_images[i].cpu().clamp(0, 1)
            orig_pil = transforms.ToPILImage()(orig_img)
            
            # 增强图
            enh_img = enhanced_images[i].cpu().clamp(0, 1)
            enh_pil = transforms.ToPILImage()(enh_img)
            
            # 拼接保存
            combined = Image.new('RGB', (orig_pil.width * 2, orig_pil.height))
            combined.paste(orig_pil, (0, 0))
            combined.paste(enh_pil, (orig_pil.width, 0))
            
            combined.save(os.path.join(save_dir, f'epoch_{epoch}_sample_{i}.jpg'))
    
    model.train()

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None, 
                   use_amp=False, scaler=None, clip_grad_norm=None):
    """
    训练一个epoch（支持混合精度和梯度裁剪）
    """
    model.train()
    total_loss = 0.0
    total_spa_loss = 0.0
    total_exp_loss = 0.0
    total_col_loss = 0.0
    total_tv_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            # 混合精度训练
            with autocast():
                enhanced_images, A_list = model(images)
                total_loss_batch, spa_loss, exp_loss, col_loss, tv_loss = criterion(
                    images, enhanced_images, A_list
                )
            
            scaler.scale(total_loss_batch).backward()
            
            if clip_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准训练
            enhanced_images, A_list = model(images)
            total_loss_batch, spa_loss, exp_loss, col_loss, tv_loss = criterion(
                images, enhanced_images, A_list
            )
            
            total_loss_batch.backward()
            
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
        
        # 累计损失
        total_loss += total_loss_batch.item()
        total_spa_loss += spa_loss.item()
        total_exp_loss += exp_loss.item()
        total_col_loss += col_loss.item()
        total_tv_loss += tv_loss.item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{total_loss_batch.item():.4f}',
            'Spa': f'{spa_loss.item():.4f}',
            'Exp': f'{exp_loss.item():.4f}',
            'Col': f'{col_loss.item():.4f}',
            'TV': f'{tv_loss.item():.4f}'
        })
        
        # 记录batch级别的指标
        if writer and batch_idx % 100 == 0:
            global_step = (epoch - 1) * len(dataloader) + batch_idx
            writer.add_scalar('Batch/Total_Loss', total_loss_batch.item(), global_step)
            writer.add_scalar('Batch/Spatial_Loss', spa_loss.item(), global_step)
            writer.add_scalar('Batch/Exposure_Loss', exp_loss.item(), global_step)
            writer.add_scalar('Batch/Color_Loss', col_loss.item(), global_step)
            writer.add_scalar('Batch/TV_Loss', tv_loss.item(), global_step)
    
    # 计算平均损失
    num_batches = len(dataloader)
    avg_total_loss = total_loss / num_batches
    avg_spa_loss = total_spa_loss / num_batches
    avg_exp_loss = total_exp_loss / num_batches
    avg_col_loss = total_col_loss / num_batches
    avg_tv_loss = total_tv_loss / num_batches
    
    return avg_total_loss, avg_spa_loss, avg_exp_loss, avg_col_loss, avg_tv_loss

def get_scheduler(optimizer, scheduler_type, **kwargs):
    """
    获取学习率调度器
    """
    if scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, 
                                       step_size=kwargs.get('step_size', 50), 
                                       gamma=kwargs.get('gamma', 0.5))
    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                  T_max=kwargs.get('T_max', 200),
                                                  eta_min=kwargs.get('eta_min', 1e-6))
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                  mode='min',
                                                  factor=kwargs.get('factor', 0.5),
                                                  patience=kwargs.get('patience', 20))
    elif scheduler_type == 'exponential':
        return optim.lr_scheduler.ExponentialLR(optimizer, 
                                              gamma=kwargs.get('gamma', 0.95))
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def main():
    parser = argparse.ArgumentParser(description='Zero-DCE Advanced Training')
    
    # 基础参数
    parser.add_argument('--data_dir', type=str, default='LLIEDet/images', help='Path to image directory')
    parser.add_argument('--train_annotation', type=str, default='LLIEDet/coco_annotations/train.json', help='Path to train annotation file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--n_iter', type=int, default=8, help='Number of curve iterations')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for tensorboard logs')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency (epochs)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    
    # 损失函数权重
    parser.add_argument('--spa_weight', type=float, default=1, help='Spatial consistency loss weight')
    parser.add_argument('--exp_weight', type=float, default=10, help='Exposure control loss weight')
    parser.add_argument('--col_weight', type=float, default=5, help='Color constancy loss weight')
    parser.add_argument('--tv_weight', type=float, default=1600, help='TV loss weight')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    
    # 学习率调度器
    parser.add_argument('--scheduler', type=str, default='step', 
                       choices=['step', 'cosine', 'plateau', 'exponential'], help='Scheduler type')
    parser.add_argument('--step_size', type=int, default=50, help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for StepLR/ExponentialLR')
    parser.add_argument('--patience', type=int, default=20, help='Patience for ReduceLROnPlateau')
    
    # 高级功能
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--clip_grad_norm', type=float, default=0, help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--early_stopping', type=int, default=0, help='Early stopping patience (0 to disable)')
    parser.add_argument('--save_samples', action='store_true', help='Save sample images during training')
    parser.add_argument('--sample_freq', type=int, default=20, help='Sample saving frequency (epochs)')
    
    # 数据增强
    parser.add_argument('--use_augmentation', action='store_true', help='Use data augmentation')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    if args.save_samples:
        os.makedirs(os.path.join(args.save_dir, 'samples'), exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建模型
    model = DCENet(n_iter=args.n_iter).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # 创建损失函数
    criterion = ZeroDCELoss(
        spa_weight=args.spa_weight,
        exp_weight=args.exp_weight,
        col_weight=args.col_weight,
        tv_weight=args.tv_weight
    ).to(device)
    
    # 创建优化器
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 创建学习率调度器
    scheduler_kwargs = {
        'step_size': args.step_size,
        'gamma': args.gamma,
        'T_max': args.epochs,
        'patience': args.patience
    }
    scheduler = get_scheduler(optimizer, args.scheduler, **scheduler_kwargs)
    
    # 混合精度训练
    scaler = GradScaler() if args.use_amp else None
    
    # 创建数据加载器
    train_dataloader = get_dataloader(
        data_dir=args.data_dir,
        annotation_file=args.train_annotation,
        batch_size=args.batch_size,
        shuffle=True,
        image_size=args.image_size
    )
    
    print(f'Training dataset size: {len(train_dataloader.dataset)}')
    print(f'Number of batches per epoch: {len(train_dataloader)}')
    
    # 准备样本图像
    sample_batch = None
    if args.save_samples:
        sample_batch = next(iter(train_dataloader))[0][:4]  # 取前4张图像作为样本
    
    # 创建tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # 恢复训练
    start_epoch = 1
    best_loss = float('inf')
    patience_counter = 0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint.get('loss', float('inf'))
            print(f"Resumed from epoch {start_epoch-1} with loss {best_loss:.4f}")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # 训练循环
    start_time = time.time()
    
    print("\n" + "="*60)
    print("Training Configuration:")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Scheduler: {args.scheduler}")
    print(f"  Mixed Precision: {args.use_amp}")
    print(f"  Gradient Clipping: {args.clip_grad_norm if args.clip_grad_norm > 0 else 'Disabled'}")
    print(f"  Early Stopping: {args.early_stopping if args.early_stopping > 0 else 'Disabled'}")
    print(f"  Loss Weights: spa={args.spa_weight}, exp={args.exp_weight}, col={args.col_weight}, tv={args.tv_weight}")
    print("="*60 + "\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # 训练一个epoch
        avg_total_loss, avg_spa_loss, avg_exp_loss, avg_col_loss, avg_tv_loss = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device, epoch, writer,
            use_amp=args.use_amp, scaler=scaler, 
            clip_grad_norm=args.clip_grad_norm if args.clip_grad_norm > 0 else None
        )
        
        # 更新学习率
        if args.scheduler == 'plateau':
            scheduler.step(avg_total_loss)
        else:
            scheduler.step()
        
        # 记录epoch级别的指标
        writer.add_scalar('Epoch/Total_Loss', avg_total_loss, epoch)
        writer.add_scalar('Epoch/Spatial_Loss', avg_spa_loss, epoch)
        writer.add_scalar('Epoch/Exposure_Loss', avg_exp_loss, epoch)
        writer.add_scalar('Epoch/Color_Loss', avg_col_loss, epoch)
        writer.add_scalar('Epoch/TV_Loss', avg_tv_loss, epoch)
        writer.add_scalar('Epoch/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 计算损失比例
        total_weighted_loss = (args.spa_weight * avg_spa_loss + 
                             args.exp_weight * avg_exp_loss + 
                             args.col_weight * avg_col_loss + 
                             args.tv_weight * avg_tv_loss)
        
        spa_ratio = (args.spa_weight * avg_spa_loss) / total_weighted_loss
        exp_ratio = (args.exp_weight * avg_exp_loss) / total_weighted_loss
        col_ratio = (args.col_weight * avg_col_loss) / total_weighted_loss
        tv_ratio = (args.tv_weight * avg_tv_loss) / total_weighted_loss
        
        writer.add_scalar('Loss_Ratios/Spatial', spa_ratio, epoch)
        writer.add_scalar('Loss_Ratios/Exposure', exp_ratio, epoch)
        writer.add_scalar('Loss_Ratios/Color', col_ratio, epoch)
        writer.add_scalar('Loss_Ratios/TV', tv_ratio, epoch)
        
        # 打印训练信息
        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch}/{args.epochs} - '
              f'Loss: {avg_total_loss:.4f} - '
              f'Time: {elapsed_time:.1f}s - '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  Loss Ratios - Spa: {spa_ratio:.3f}, Exp: {exp_ratio:.3f}, Col: {col_ratio:.3f}, TV: {tv_ratio:.3f}')
        
        # 保存样本图像
        if args.save_samples and epoch % args.sample_freq == 0 and sample_batch is not None:
            save_sample_images(model, sample_batch, epoch, 
                             os.path.join(args.save_dir, 'samples'), device)
        
        # 保存最佳模型
        is_best = avg_total_loss < best_loss
        if is_best:
            best_loss = avg_total_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'  ✓ Best model saved with loss: {best_loss:.4f}')
        else:
            patience_counter += 1
        
        # 早停检查
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"\nEarly stopping triggered after {args.early_stopping} epochs without improvement")
            break
        
        # 定期保存检查点
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_total_loss,
                'args': args
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_total_loss,
        'args': args
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    writer.close()
    
    print("\n" + "="*60)
    print('Training completed!')
    print(f'Best loss: {best_loss:.4f}')
    print(f'Total training time: {(time.time() - start_time) / 3600:.2f} hours')
    print(f'Final epoch: {epoch}')
    if args.early_stopping > 0:
        print(f'Early stopping patience used: {patience_counter}/{args.early_stopping}')
    print("="*60)

if __name__ == '__main__':
    main()