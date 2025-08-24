# Zero-DCE: 零样本低光照图像增强

基于论文 "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement" 的实现。

## 项目概述

本项目实现了Zero-DCE（零参考深度曲线估计）方法，用于低光照图像增强。该方法的核心思想是学习像素级的亮度调整曲线，而不是直接生成增强后的像素值。通过四个无参考损失函数的约束，模型能够在没有成对参考图像的情况下学习有效的图像增强策略。

### 主要特点

- **无需成对数据**: 不需要低光照图像和对应的正常光照图像对
- **轻量级网络**: 仅使用7层卷积网络，参数量小
- **端到端训练**: 通过无参考损失函数直接优化
- **实时处理**: 网络结构简单，推理速度快

## 技术原理

### 核心公式

对于输入的低光图像 L(x)，增强后的图像 E(x) 通过以下公式得到：

```
E(x) = L(x) + A(x) · L(x) · (1 - L(x))
```

其中 A(x) 是网络学习的曲线参数图。

### 损失函数

1. **空间一致性损失 (L_spa)**: 保持增强前后图像的空间结构一致性
2. **曝光控制损失 (L_exp)**: 控制图像的局部曝光水平
3. **色彩恒常性损失 (L_col)**: 保持色彩平衡
4. **光照平滑损失 (L_tv)**: 确保曲线参数图的平滑性

总损失函数：
```
L_total = W_spa·L_spa + W_exp·L_exp + W_col·L_col + W_tv·L_tv
```

## 项目结构

```
Zero-DCE/
├── model.py           # DCE-Net模型定义
├── loss.py            # 损失函数实现
├── dataset.py         # 数据加载器
├── train.py           # 训练脚本
├── evaluate.py        # 评估脚本
├── inference.py       # 推理脚本
├── requirements.txt   # 依赖包列表
├── README.md          # 项目说明
├── checkpoints/      # 模型检查点
├── logs/            # 训练日志
└── results/         # 评估结果
```


## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 主要依赖包

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- pyiqa >= 0.1.7 (用于Q-Align评估)
- PIL, numpy, tqdm, tensorboard

## 使用方法

### 1. 数据准备

确保LLIEDet数据集已正确放置：
- 图像文件在 `LLIEDet/images/` 目录下
- 标注文件在 `LLIEDet/coco_annotations/` 目录下

### 2. 训练模型

#### 快速开始（推荐新手）

使用预设的优化配置，一键开始高质量训练：

```bash
# 查看所有可用配置
python quick_train.py --list

# 平衡配置（推荐）
python quick_train.py balanced --data_dir LLIEDet/images

# 最佳质量配置
python quick_train.py best --data_dir LLIEDet/images

# 快速训练配置
python quick_train.py fast --data_dir LLIEDet/images

# 低显存配置（4GB以下GPU）
python quick_train.py low_memory --data_dir LLIEDet/images
```

#### 基础训练
```bash
# 基础训练
python train.py --data_dir LLIEDet/images --batch_size 8 --epochs 100

# 使用GPU训练
python train.py --data_dir LLIEDet/images --batch_size 16 --epochs 200 --device cuda

# 自定义参数训练
python train.py --batch_size 16 --epochs 200 --lr 1e-4 --image_size 256

# 调整损失函数权重
python train.py --spa_weight 1 --exp_weight 10 --col_weight 5 --tv_weight 1600
```

#### 高级训练（手动配置）
```bash
# 使用改进的训练脚本
python train_advanced.py --data_dir LLIEDet/images --batch_size 16 --epochs 250 --use_amp --save_samples

# 平衡型配置（推荐）
python train_advanced.py \
    --lr 2e-4 \
    --batch_size 16 \
    --epochs 250 \
    --spa_weight 1.5 \
    --exp_weight 15 \
    --col_weight 8 \
    --tv_weight 1200 \
    --scheduler cosine \
    --use_amp \
    --clip_grad_norm 1.0 \
    --early_stopping 50
```

### 自动参数调优（推荐）

使用自动参数调优脚本快速找到最佳参数组合：

```bash
# 使用推荐配置（4个预设配置）
python auto_tune_params.py --data_dir LLIEDet/images --epochs 50

# 随机搜索（10个随机配置）
python auto_tune_params.py --strategy random --num_experiments 10 --data_dir LLIEDet/images

# 网格搜索（自动限制数量）
python auto_tune_params.py --strategy grid --max_grid_size 20 --data_dir LLIEDet/images

# 分析已有实验结果
python auto_tune_params.py --analyze_only --results_dir experiments
```

自动调优功能：
- 🎯 **智能搜索**：推荐配置、随机搜索、网格搜索
- 📊 **自动分析**：参数重要性分析、性能排名
- 💾 **结果保存**：完整的实验记录和配置
- 🔄 **断点续传**：支持中断后继续分析
- 📈 **可视化报告**：生成详细的分析报告

### 3. 评估模型

```bash
# 使用Q-Align指标评估（在线下载模型）
python evaluate.py --checkpoint checkpoints/best_model.pth --use_qalign

# 使用本地Q-Align模型评估（推荐，避免网络问题）
python evaluate.py --checkpoint checkpoints/best_model.pth --use_qalign --qalign_model_path ./models/qalign

# 基础评估（仅保存增强结果）
python evaluate.py --checkpoint checkpoints/best_model.pth --save_dir results
```


### 4. 单张图像推理

```bash
# 增强单张图像
python inference.py --checkpoint checkpoints/best_model.pth --input path/to/image.jpg

# 指定输出路径
python inference.py --checkpoint checkpoints/best_model.pth --input input.jpg --output enhanced.png --comparison comparison.png
```

## 训练参数说明

### 主要参数

- `--batch_size`: 批次大小 (默认: 8)
- `--epochs`: 训练轮数 (默认: 200)
- `--lr`: 学习率 (默认: 1e-4)
- `--image_size`: 图像尺寸 (默认: 256)
- `--n_iter`: 曲线迭代次数 (默认: 8)

### 损失函数权重

- `--spa_weight`: 空间一致性损失权重 (默认: 1)
- `--exp_weight`: 曝光控制损失权重 (默认: 10)
- `--col_weight`: 色彩恒常性损失权重 (默认: 5)
- `--tv_weight`: 光照平滑损失权重 (默认: 1600)

## 评估指标

本项目使用pyiqa库实现的Q-Align作为主要评估指标，这是一个无参考图像质量评估指标，特别适用于低光照图像增强任务。

## 实验结果

训练过程中会自动保存：
- 最佳模型 (`best_model.pth`)
- 定期检查点 (`checkpoint_epoch_*.pth`)
- 最终模型 (`final_model.pth`)
- TensorBoard日志 (在 `logs/` 目录)

## 改进方向

1. **网络结构优化**: 可以尝试更深的网络或注意力机制
2. **损失函数改进**: 引入感知损失或对抗损失
3. **数据增强**: 添加更多的数据增强策略
4. **多尺度处理**: 实现多尺度的图像增强
5. **后处理优化**: 添加去噪或锐化后处理步骤

## 参考文献

```bibtex
@inproceedings{guo2020zero,
  title={Zero-reference deep curve estimation for low-light image enhancement},
  author={Guo, Chunle and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={1780--1789},
  year={2020}
}
```

## 项目说明

本项目是基于Zero-DCE论文的实现，代码结构清晰，注释详细，便于理解和扩展。项目遵循学术诚信原则，明确标注了代码和模型的来源，并在原有方法基础上进行了适配和优化。

### 主要贡献

1. 完整实现了Zero-DCE方法的所有核心组件
2. 适配了LLIEDet数据集的数据加载
3. 集成了pyiqa库的Q-Align评估指标
4. 提供了完整的训练、评估和推理流程
5. 添加了详细的文档和使用说明

## 联系方式

如有问题或建议，请通过项目仓库的Issue功能进行反馈。