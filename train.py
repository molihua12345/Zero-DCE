import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import time

from model import DCENet
from loss import ZeroDCELoss
from dataset import get_dataloader

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """
    训练一个epoch
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
        
        # 前向传播
        enhanced_images, A_list = model(images)
        
        # 计算损失
        total_loss_batch, spa_loss, exp_loss, col_loss, tv_loss = criterion(
            images, enhanced_images, A_list
        )
        
        # 反向传播
        optimizer.zero_grad()
        total_loss_batch.backward()
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
            'SPA': f'{spa_loss.item():.4f}',
            'EXP': f'{exp_loss.item():.4f}',
            'COL': f'{col_loss.item():.4f}',
            'TV': f'{tv_loss.item():.4f}'
        })
        
        # 记录到tensorboard
        if writer and batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Total_Loss', total_loss_batch.item(), global_step)
            writer.add_scalar('Train/Spatial_Loss', spa_loss.item(), global_step)
            writer.add_scalar('Train/Exposure_Loss', exp_loss.item(), global_step)
            writer.add_scalar('Train/Color_Loss', col_loss.item(), global_step)
            writer.add_scalar('Train/TV_Loss', tv_loss.item(), global_step)
    
    # 计算平均损失
    avg_total_loss = total_loss / len(dataloader)
    avg_spa_loss = total_spa_loss / len(dataloader)
    avg_exp_loss = total_exp_loss / len(dataloader)
    avg_col_loss = total_col_loss / len(dataloader)
    avg_tv_loss = total_tv_loss / len(dataloader)
    
    return avg_total_loss, avg_spa_loss, avg_exp_loss, avg_col_loss, avg_tv_loss

def main():
    parser = argparse.ArgumentParser(description='Zero-DCE Training')
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
    
    # 损失函数权重
    parser.add_argument('--spa_weight', type=float, default=1, help='Spatial consistency loss weight')
    parser.add_argument('--exp_weight', type=float, default=10, help='Exposure control loss weight')
    parser.add_argument('--col_weight', type=float, default=5, help='Color constancy loss weight')
    parser.add_argument('--tv_weight', type=float, default=1600, help='TV loss weight')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
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
    
    # 创建tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # 训练循环
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # 训练一个epoch
        avg_total_loss, avg_spa_loss, avg_exp_loss, avg_col_loss, avg_tv_loss = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device, epoch, writer
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录epoch级别的指标
        writer.add_scalar('Epoch/Total_Loss', avg_total_loss, epoch)
        writer.add_scalar('Epoch/Spatial_Loss', avg_spa_loss, epoch)
        writer.add_scalar('Epoch/Exposure_Loss', avg_exp_loss, epoch)
        writer.add_scalar('Epoch/Color_Loss', avg_col_loss, epoch)
        writer.add_scalar('Epoch/TV_Loss', avg_tv_loss, epoch)
        writer.add_scalar('Epoch/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 打印训练信息
        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch}/{args.epochs} - '
              f'Loss: {avg_total_loss:.4f} - '
              f'Time: {elapsed_time:.1f}s - '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 保存最佳模型
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'Best model saved with loss: {best_loss:.4f}')
        
        # 定期保存检查点
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss,
                'args': args
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_total_loss,
        'args': args
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    writer.close()
    print('Training completed!')
    print(f'Best loss: {best_loss:.4f}')
    print(f'Total training time: {(time.time() - start_time) / 3600:.2f} hours')

if __name__ == '__main__':
    main()