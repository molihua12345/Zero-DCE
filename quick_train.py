#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速训练脚本
提供预设的优化配置，让用户快速开始高质量训练
"""

import os
import argparse
import subprocess
import sys

# 预设的优化配置
OPTIMIZED_CONFIGS = {
    'best': {
        'name': '最佳质量配置',
        'description': '追求最高图像质量，训练时间较长',
        'params': {
            'lr': 1e-4,
            'batch_size': 8,
            'epochs': 200,
            'spa_weight': 2.0,
            'exp_weight': 20,
            'col_weight': 10,
            'tv_weight': 1600,
            'scheduler': 'cosine',
            'optimizer': 'adamw',
            'use_amp': True,
            'clip_grad_norm': 1.0,
            'early_stopping': 30,
            'save_samples': True
        }
    },
    'balanced': {
        'name': '平衡配置',
        'description': '质量与速度的平衡，推荐日常使用',
        'params': {
            'lr': 2e-4,
            'batch_size': 16,
            'epochs': 150,
            'spa_weight': 1.5,
            'exp_weight': 15,
            'col_weight': 8,
            'tv_weight': 1200,
            'scheduler': 'cosine',
            'optimizer': 'adamw',
            'use_amp': True,
            'clip_grad_norm': 1.0,
            'early_stopping': 25,
            'save_samples': True
        }
    },
    'fast': {
        'name': '快速训练配置',
        'description': '快速训练，适合测试和调试',
        'params': {
            'lr': 5e-4,
            'batch_size': 24,
            'epochs': 100,
            'spa_weight': 1.0,
            'exp_weight': 12,
            'col_weight': 6,
            'tv_weight': 1000,
            'scheduler': 'step',
            'optimizer': 'adam',
            'use_amp': True,
            'clip_grad_norm': 0.5,
            'early_stopping': 20,
            'save_samples': False
        }
    },
    'detail': {
        'name': '细节保持配置',
        'description': '最大化保持图像细节，适合高分辨率图像',
        'params': {
            'lr': 1e-4,
            'batch_size': 12,
            'epochs': 180,
            'spa_weight': 3.0,
            'exp_weight': 10,
            'col_weight': 5,
            'tv_weight': 2000,
            'scheduler': 'plateau',
            'optimizer': 'adamw',
            'use_amp': True,
            'clip_grad_norm': 1.5,
            'early_stopping': 35,
            'save_samples': True
        }
    },
    'low_memory': {
        'name': '低显存配置',
        'description': '适合显存较小的GPU（4GB以下）',
        'params': {
            'lr': 2e-4,
            'batch_size': 4,
            'epochs': 200,
            'spa_weight': 1.5,
            'exp_weight': 15,
            'col_weight': 8,
            'tv_weight': 1200,
            'scheduler': 'cosine',
            'optimizer': 'adamw',
            'use_amp': True,
            'clip_grad_norm': 1.0,
            'early_stopping': 30,
            'save_samples': False,
            'image_size': 256  # 降低图像尺寸
        }
    }
}

def print_config_info():
    """
    打印所有可用配置的信息
    """
    print("\n" + "="*60)
    print("可用的预设配置：")
    print("="*60)
    
    for key, config in OPTIMIZED_CONFIGS.items():
        print(f"\n🔧 {key}: {config['name']}")
        print(f"   描述: {config['description']}")
        
        params = config['params']
        print(f"   参数: lr={params['lr']}, batch_size={params['batch_size']}, epochs={params['epochs']}")
        print(f"         损失权重: spa={params['spa_weight']}, exp={params['exp_weight']}, col={params['col_weight']}, tv={params['tv_weight']}")
        print(f"         优化器: {params['optimizer']}, 调度器: {params['scheduler']}")
        
        # 估算训练时间和显存需求
        if params['batch_size'] <= 8:
            memory_req = "低 (4-6GB)"
        elif params['batch_size'] <= 16:
            memory_req = "中 (6-8GB)"
        else:
            memory_req = "高 (8GB+)"
        
        if params['epochs'] <= 100:
            time_est = "短 (2-4小时)"
        elif params['epochs'] <= 150:
            time_est = "中 (4-6小时)"
        else:
            time_est = "长 (6-10小时)"
        
        print(f"   预估: 显存需求={memory_req}, 训练时间={time_est}")

def run_training(config_name, data_dir, custom_args):
    """
    运行训练
    """
    if config_name not in OPTIMIZED_CONFIGS:
        print(f"错误: 未知配置 '{config_name}'")
        print("可用配置:", list(OPTIMIZED_CONFIGS.keys()))
        return False
    
    config = OPTIMIZED_CONFIGS[config_name]
    params = config['params']
    
    print(f"\n{'='*60}")
    print(f"开始训练: {config['name']}")
    print(f"描述: {config['description']}")
    print(f"{'='*60}")
    
    # 构建训练命令
    cmd = ['python', 'train_advanced.py']
    
    # 添加基础参数
    cmd.extend(['--data_dir', data_dir])
    cmd.extend(['--save_dir', f'models/{config_name}_model'])
    cmd.extend(['--log_dir', f'logs/{config_name}_training'])
    
    # 添加配置参数
    for param, value in params.items():
        if param == 'use_amp' and value:
            cmd.append('--use_amp')
        elif param == 'save_samples' and value:
            cmd.append('--save_samples')
        elif param not in ['use_amp', 'save_samples']:
            cmd.extend([f'--{param}', str(value)])
    
    # 添加自定义参数
    if custom_args:
        cmd.extend(custom_args)
    
    print(f"执行命令: {' '.join(cmd)}")
    print(f"\n开始训练...")
    
    # 创建输出目录
    os.makedirs(f'models/{config_name}_model', exist_ok=True)
    os.makedirs(f'logs/{config_name}_training', exist_ok=True)
    
    # 运行训练
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ 训练完成！")
        print(f"模型保存在: models/{config_name}_model/")
        print(f"日志保存在: logs/{config_name}_training/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 训练失败，错误码: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ 训练被用户中断")
        return False
    except Exception as e:
        print(f"\n✗ 训练出错: {e}")
        return False

def check_requirements():
    """
    检查训练环境
    """
    print("检查训练环境...")
    
    # 检查必要文件
    required_files = ['train_advanced.py', 'model.py', 'loss.py']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ 缺少必要文件: {', '.join(missing_files)}")
        return False
    
    # 检查PyTorch
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"✓ 可用GPU数量: {gpu_count}")
        else:
            print("⚠ 未检测到CUDA GPU，将使用CPU训练（速度较慢）")
    
    except ImportError:
        print("✗ 未安装PyTorch")
        return False
    
    print("✓ 环境检查通过")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Zero-DCE快速训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python quick_train.py --list                    # 查看所有配置
  python quick_train.py balanced                  # 使用平衡配置训练
  python quick_train.py best --data_dir mydata    # 使用最佳配置训练
  python quick_train.py fast --epochs 50         # 快速训练50轮
        """
    )
    
    parser.add_argument('config', nargs='?', help='配置名称 (balanced/best/fast/detail/low_memory)')
    parser.add_argument('--data_dir', type=str, default='LLIEDet/images', help='训练数据目录')
    parser.add_argument('--list', action='store_true', help='列出所有可用配置')
    parser.add_argument('--check', action='store_true', help='检查训练环境')
    
    # 允许传递额外参数给训练脚本
    args, unknown_args = parser.parse_known_args()
    
    print("\n🚀 Zero-DCE 快速训练工具")
    print("=" * 40)
    
    # 检查环境
    if args.check or not args.list:
        if not check_requirements():
            print("\n请先安装必要的依赖包")
            sys.exit(1)
    
    # 列出配置
    if args.list:
        print_config_info()
        print("\n使用方法: python quick_train.py <配置名称> --data_dir <数据目录>")
        return
    
    # 检查配置参数
    if not args.config:
        print("\n请指定配置名称，或使用 --list 查看可用配置")
        print("可用配置:", list(OPTIMIZED_CONFIGS.keys()))
        sys.exit(1)
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"\n错误: 数据目录不存在: {args.data_dir}")
        print("请确保数据目录存在，或使用 --data_dir 指定正确路径")
        sys.exit(1)
    
    # 显示配置信息
    if args.config in OPTIMIZED_CONFIGS:
        config = OPTIMIZED_CONFIGS[args.config]
        print(f"\n选择配置: {config['name']}")
        print(f"描述: {config['description']}")
        print(f"数据目录: {args.data_dir}")
        
        if unknown_args:
            print(f"额外参数: {' '.join(unknown_args)}")
        
        
        # 开始训练
        success = run_training(args.config, args.data_dir, unknown_args)
        
        if success:
            print(f"\n🎉 训练成功完成！")
            print(f"\n后续步骤:")
            print(f"1. 查看训练日志: tensorboard --logdir logs/{args.config}_training")
            print(f"2. 测试模型: python test.py --model_path models/{args.config}_model/best_model.pth")
            print(f"3. 评估质量: python evaluate.py --model_path models/{args.config}_model/best_model.pth")
        else:
            print(f"\n❌ 训练失败，请检查错误信息")
    else:
        print(f"\n错误: 未知配置 '{args.config}'")
        print("可用配置:", list(OPTIMIZED_CONFIGS.keys()))
        print("使用 --list 查看详细信息")

if __name__ == '__main__':
    main()