#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动参数调优脚本
通过网格搜索或随机搜索自动寻找最佳的训练参数组合
"""

import os
import json
import time
import argparse
import itertools
import random
from datetime import datetime
import subprocess
import torch

# 预定义的参数搜索空间
PARAM_SEARCH_SPACE = {
    'lr': [1e-4, 2e-4, 5e-4, 1e-3],
    'batch_size': [8, 12, 16, 24],
    'spa_weight': [1.0, 1.5, 2.0, 3.0],
    'exp_weight': [8, 10, 12, 15, 20],
    'col_weight': [3, 5, 8, 10],
    'tv_weight': [800, 1000, 1200, 1600, 2000],
    'scheduler': ['step', 'cosine', 'plateau']
}

# 推荐的参数组合
RECOMMENDED_CONFIGS = [
    {
        'name': 'balanced',
        'lr': 2e-4,
        'batch_size': 16,
        'spa_weight': 1.5,
        'exp_weight': 15,
        'col_weight': 8,
        'tv_weight': 1200,
        'scheduler': 'cosine'
    },
    {
        'name': 'high_quality',
        'lr': 1e-4,
        'batch_size': 8,
        'spa_weight': 2.0,
        'exp_weight': 20,
        'col_weight': 10,
        'tv_weight': 1600,
        'scheduler': 'cosine'
    },
    {
        'name': 'fast_training',
        'lr': 5e-4,
        'batch_size': 24,
        'spa_weight': 1.0,
        'exp_weight': 12,
        'col_weight': 6,
        'tv_weight': 1000,
        'scheduler': 'step'
    },
    {
        'name': 'detail_preserving',
        'lr': 1e-4,
        'batch_size': 12,
        'spa_weight': 3.0,
        'exp_weight': 10,
        'col_weight': 5,
        'tv_weight': 2000,
        'scheduler': 'plateau'
    }
]

def run_training_experiment(config, base_args, experiment_id):
    """
    运行单次训练实验
    """
    print(f"\n{'='*60}")
    print(f"Running Experiment {experiment_id}: {config.get('name', 'custom')}")
    print(f"{'='*60}")
    
    # 构建训练命令
    cmd = [
        'python', 'train_advanced.py',
        '--data_dir', base_args.data_dir,
        '--epochs', str(base_args.epochs),
        '--save_dir', f'experiments/exp_{experiment_id:03d}',
        '--log_dir', f'logs/exp_{experiment_id:03d}',
        '--lr', str(config['lr']),
        '--batch_size', str(config['batch_size']),
        '--spa_weight', str(config['spa_weight']),
        '--exp_weight', str(config['exp_weight']),
        '--col_weight', str(config['col_weight']),
        '--tv_weight', str(config['tv_weight']),
        '--scheduler', config['scheduler']
    ]
    
    # 添加可选参数
    if base_args.use_amp:
        cmd.append('--use_amp')
    if base_args.clip_grad_norm > 0:
        cmd.extend(['--clip_grad_norm', str(base_args.clip_grad_norm)])
    if base_args.early_stopping > 0:
        cmd.extend(['--early_stopping', str(base_args.early_stopping)])
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Config: {config}")
    
    # 创建实验目录
    exp_dir = f'experiments/exp_{experiment_id:03d}'
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    config_file = os.path.join(exp_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 运行训练
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=base_args.timeout)
        success = result.returncode == 0
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired:
        success = False
        stdout = ""
        stderr = "Training timeout"
    except Exception as e:
        success = False
        stdout = ""
        stderr = str(e)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 解析结果
    best_loss = float('inf')
    final_epoch = 0
    
    if success:
        # 尝试从最佳模型文件中读取损失
        best_model_path = os.path.join(exp_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            try:
                checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
                best_loss = checkpoint.get('loss', float('inf'))
                final_epoch = checkpoint.get('epoch', 0)
            except:
                pass
        
        # 从stdout中解析损失（备选方案）
        if best_loss == float('inf'):
            for line in stdout.split('\n'):
                if 'Best loss:' in line:
                    try:
                        best_loss = float(line.split('Best loss:')[1].strip())
                    except:
                        pass
    
    # 保存实验结果
    result_data = {
        'experiment_id': experiment_id,
        'config': config,
        'success': success,
        'best_loss': best_loss,
        'final_epoch': final_epoch,
        'training_time': training_time,
        'timestamp': datetime.now().isoformat(),
        'stdout': stdout[-2000:] if len(stdout) > 2000 else stdout,  # 保留最后2000字符
        'stderr': stderr[-1000:] if len(stderr) > 1000 else stderr   # 保留最后1000字符
    }
    
    result_file = os.path.join(exp_dir, 'result.json')
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    # 打印结果
    status = "✓ SUCCESS" if success else "✗ FAILED"
    print(f"\nExperiment {experiment_id} {status}")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Final Epoch: {final_epoch}")
    print(f"Training Time: {training_time/60:.1f} minutes")
    
    if not success:
        print(f"Error: {stderr}")
    
    return result_data

def generate_random_configs(search_space, num_configs):
    """
    生成随机参数配置
    """
    configs = []
    for i in range(num_configs):
        config = {}
        for param, values in search_space.items():
            config[param] = random.choice(values)
        config['name'] = f'random_{i+1}'
        configs.append(config)
    return configs

def generate_grid_configs(search_space, max_configs=50):
    """
    生成网格搜索配置（限制数量）
    """
    # 计算所有可能的组合数
    total_combinations = 1
    for values in search_space.values():
        total_combinations *= len(values)
    
    print(f"Total possible combinations: {total_combinations}")
    
    if total_combinations <= max_configs:
        # 如果组合数不多，使用完整网格搜索
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        
        configs = []
        for i, combination in enumerate(itertools.product(*param_values)):
            config = dict(zip(param_names, combination))
            config['name'] = f'grid_{i+1}'
            configs.append(config)
        
        return configs
    else:
        # 如果组合数太多，使用随机采样
        print(f"Too many combinations, using random sampling of {max_configs} configs")
        return generate_random_configs(search_space, max_configs)

def analyze_results(results_dir):
    """
    分析实验结果
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    results = []
    
    # 收集所有实验结果
    for exp_dir in os.listdir(results_dir):
        if exp_dir.startswith('exp_'):
            result_file = os.path.join(results_dir, exp_dir, 'result.json')
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        result = json.load(f)
                    results.append(result)
                except:
                    continue
    
    if not results:
        print("No experiment results found.")
        return
    
    # 过滤成功的实验
    successful_results = [r for r in results if r['success'] and r['best_loss'] != float('inf')]
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful experiments: {len(successful_results)}")
    print(f"Success rate: {len(successful_results)/len(results)*100:.1f}%")
    
    if not successful_results:
        print("No successful experiments to analyze.")
        return
    
    # 按损失排序
    successful_results.sort(key=lambda x: x['best_loss'])
    
    print(f"\nTOP 5 BEST CONFIGURATIONS:")
    print("-" * 60)
    
    for i, result in enumerate(successful_results[:5]):
        config = result['config']
        print(f"\nRank {i+1}: {config.get('name', 'unknown')} (Exp {result['experiment_id']})")
        print(f"  Best Loss: {result['best_loss']:.4f}")
        print(f"  Training Time: {result['training_time']/60:.1f} min")
        print(f"  Final Epoch: {result['final_epoch']}")
        print(f"  Config: lr={config['lr']}, bs={config['batch_size']}, "
              f"spa={config['spa_weight']}, exp={config['exp_weight']}, "
              f"col={config['col_weight']}, tv={config['tv_weight']}, "
              f"sched={config['scheduler']}")
    
    # 参数重要性分析
    print(f"\nPARAMETER IMPORTANCE ANALYSIS:")
    print("-" * 60)
    
    # 分析每个参数对性能的影响
    param_analysis = {}
    for param in ['lr', 'batch_size', 'spa_weight', 'exp_weight', 'col_weight', 'tv_weight', 'scheduler']:
        param_values = {}
        for result in successful_results:
            value = result['config'][param]
            if value not in param_values:
                param_values[value] = []
            param_values[value].append(result['best_loss'])
        
        # 计算每个值的平均损失
        avg_losses = {}
        for value, losses in param_values.items():
            avg_losses[value] = sum(losses) / len(losses)
        
        # 找到最佳值
        best_value = min(avg_losses.keys(), key=lambda x: avg_losses[x])
        param_analysis[param] = {
            'best_value': best_value,
            'best_avg_loss': avg_losses[best_value],
            'all_values': avg_losses
        }
        
        print(f"  {param}: best={best_value} (avg_loss={avg_losses[best_value]:.4f})")
    
    # 生成推荐配置
    print(f"\nRECOMMENDED CONFIGURATION:")
    print("-" * 60)
    
    recommended_config = {}
    for param, analysis in param_analysis.items():
        recommended_config[param] = analysis['best_value']
    
    print("Based on parameter importance analysis:")
    for param, value in recommended_config.items():
        print(f"  --{param} {value}")
    
    # 保存分析结果
    analysis_result = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(results),
        'successful_experiments': len(successful_results),
        'success_rate': len(successful_results)/len(results),
        'top_5_configs': successful_results[:5],
        'parameter_analysis': param_analysis,
        'recommended_config': recommended_config
    }
    
    analysis_file = os.path.join(results_dir, 'analysis_result.json')
    with open(analysis_file, 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    print(f"\nAnalysis results saved to: {analysis_file}")

def main():
    parser = argparse.ArgumentParser(description='Auto Parameter Tuning for Zero-DCE')
    
    # 基础参数
    parser.add_argument('--data_dir', type=str, default='LLIEDet/images', help='Path to training data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs per experiment')
    parser.add_argument('--timeout', type=int, default=7200, help='Timeout per experiment (seconds)')
    
    # 搜索策略
    parser.add_argument('--strategy', type=str, default='recommended', 
                       choices=['recommended', 'random', 'grid'], help='Search strategy')
    parser.add_argument('--num_experiments', type=int, default=10, help='Number of experiments for random search')
    parser.add_argument('--max_grid_size', type=int, default=50, help='Maximum grid size for grid search')
    
    # 高级选项
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--early_stopping', type=int, default=20, help='Early stopping patience')
    
    # 分析选项
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze existing results')
    parser.add_argument('--results_dir', type=str, default='experiments', help='Results directory')
    
    args = parser.parse_args()
    
    # 创建实验目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.analyze_only:
        analyze_results(args.results_dir)
        return
    
    print(f"{'='*60}")
    print("ZERO-DCE AUTOMATIC PARAMETER TUNING")
    print(f"{'='*60}")
    print(f"Strategy: {args.strategy}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Epochs per experiment: {args.epochs}")
    print(f"Timeout per experiment: {args.timeout/60:.1f} minutes")
    
    # 生成实验配置
    if args.strategy == 'recommended':
        configs = RECOMMENDED_CONFIGS.copy()
        print(f"Using {len(configs)} recommended configurations")
    elif args.strategy == 'random':
        configs = generate_random_configs(PARAM_SEARCH_SPACE, args.num_experiments)
        print(f"Generated {len(configs)} random configurations")
    elif args.strategy == 'grid':
        configs = generate_grid_configs(PARAM_SEARCH_SPACE, args.max_grid_size)
        print(f"Generated {len(configs)} grid search configurations")
    
    print(f"\nStarting {len(configs)} experiments...")
    
    # 运行实验
    results = []
    start_time = time.time()
    
    for i, config in enumerate(configs, 1):
        try:
            result = run_training_experiment(config, args, i)
            results.append(result)
        except KeyboardInterrupt:
            print("\nExperiments interrupted by user")
            break
        except Exception as e:
            print(f"\nExperiment {i} failed with error: {e}")
            continue
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"COMPLETED {len(results)} EXPERIMENTS")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"{'='*60}")
    
    # 分析结果
    if results:
        analyze_results(args.results_dir)
    
    print(f"\nAll experiment results saved in: {args.results_dir}")
    print("To re-analyze results later, run:")
    print(f"python auto_tune_params.py --analyze_only --results_dir {args.results_dir}")

if __name__ == '__main__':
    main()