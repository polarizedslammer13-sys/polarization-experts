#!/usr/bin/env python3
"""
1010 Exp2 Baseline Only (with Time Monitoring)

Only runs Exp2: 1007 model (256x256 training) + evaluate at 64x64
Added comprehensive time monitoring for performance testing
No model saving to focus on speed testing
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast  # 添加混合精度支持
from torchvision import models
from tqdm import tqdm
from scipy.stats import pearsonr
import random
import time
import json
from datetime import datetime
import math
import matplotlib.pyplot as plt
import cv2

# 修改BASE_DIR为实际数据路径
BASE_DIR = "/root/autodl-tmp/facedataset_0825"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==================== Dataset ====================

class Dataset256(Dataset):
    """256x256 GT (upsampled from 64x64)"""

    def __init__(self, speckles_path, patterns_path, indices,
                 pol_channel=2, color_channel=2, max_value=255):
        self.speckles_mmap = np.load(speckles_path, mmap_mode='r')
        self.patterns_mmap = np.load(patterns_path, mmap_mode='r')
        self.indices = indices
        self.pol_channel = pol_channel
        self.color_channel = color_channel
        self.max_value = max_value

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        speckle_idx = original_idx * 3 + self.color_channel

        speckle = self.speckles_mmap[speckle_idx, self.pol_channel].astype(np.float32).copy()
        pattern = self.patterns_mmap[speckle_idx].astype(np.float32).copy()

        speckle = speckle / 255.0
        pattern = pattern / float(self.max_value)

        # Upsample to 256x256
        pattern = cv2.resize(pattern, (256, 256), interpolation=cv2.INTER_LINEAR)

        x = torch.from_numpy(speckle).unsqueeze(0).float()
        gt = torch.from_numpy(pattern).float()
        return x, gt


# ==================== Model Architecture ====================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual, inplace=True)


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, residual=False):
        super().__init__()
        self.residual = residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        if residual:
            self.res_block = ResidualBlock(out_c)

    def forward(self, x):
        out = self.conv(x)
        if self.residual:
            out = self.res_block(out)
        return out


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1, inplace=True)
        psi = self.psi(psi)
        return x * psi


class UNetPro256(nn.Module):
    """1007's model: outputs 256x256"""

    def __init__(self, in_channels=1, base=48):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base, residual=False)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2, residual=True)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4, residual=True)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base * 4, base * 8, residual=True)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = DoubleConv(base * 8, base * 16, residual=True)
        self.pool5 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            DoubleConv(base * 16, base * 32, residual=True),
            ResidualBlock(base * 32)
        )

        self.up5 = nn.ConvTranspose2d(base * 32, base * 16, 2, stride=2)
        self.att5 = AttentionGate(F_g=base * 16, F_l=base * 16, F_int=base * 8)
        self.dec5 = DoubleConv(base * 32, base * 16, residual=True)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.att4 = AttentionGate(F_g=base * 8, F_l=base * 8, F_int=base * 4)
        self.dec4 = DoubleConv(base * 16, base * 8, residual=True)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.att3 = AttentionGate(F_g=base * 4, F_l=base * 4, F_int=base * 2)
        self.dec3 = DoubleConv(base * 8, base * 4, residual=True)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att2 = AttentionGate(F_g=base * 2, F_l=base * 2, F_int=base)
        self.dec2 = DoubleConv(base * 4, base * 2, residual=True)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att1 = AttentionGate(F_g=base, F_l=base, F_int=base // 2)
        self.dec1 = DoubleConv(base * 2, base, residual=False)

        self.final = nn.Sequential(
            nn.Conv2d(base, base // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base // 2, 1, 1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))

        b = self.bottleneck(self.pool5(e5))

        d5 = self.up5(b)
        e5_att = self.att5(d5, e5)
        d5 = self.dec5(torch.cat([d5, e5_att], dim=1))

        d4 = self.up4(d5)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return torch.sigmoid(self.final(d1))


# ==================== Loss Functions ====================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def pcc_loss(y_true, y_pred):
    x_flat = y_pred.view(y_pred.size(0), -1)
    y_flat = y_true.view(y_true.size(0), -1)

    x_mean = torch.mean(x_flat, dim=1, keepdim=True)
    y_mean = torch.mean(y_flat, dim=1, keepdim=True)

    x_centered = x_flat - x_mean
    y_centered = y_flat - y_mean

    numerator = torch.sum(x_centered * y_centered, dim=1)
    denominator = torch.sqrt(torch.sum(x_centered ** 2, dim=1) * torch.sum(y_centered ** 2, dim=1))
    pcc = numerator / (denominator + 1e-8)
    return 1 - torch.mean(pcc)


def ssim_loss(y_true, y_pred, window_size=11, sigma=1.5):
    def gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        return g.view(1, 1, -1) * g.view(1, -1, 1)

    # 确保输入是4D张量，并且是单通道
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
    if y_pred.dim() == 3:
        y_pred = y_pred.unsqueeze(1)
    
    # 如果有多通道，只取第一个通道或者平均
    if y_true.size(1) > 1:
        y_true = y_true.mean(dim=1, keepdim=True)
    if y_pred.size(1) > 1:
        y_pred = y_pred.mean(dim=1, keepdim=True)

    window = gaussian_window(window_size, sigma).to(y_true.device)
    
    # 确保window是4D张量：(1, 1, window_size, window_size)
    if window.dim() == 3:
        window = window.unsqueeze(0)
    
    mu1 = F.conv2d(y_true, window, padding=window_size // 2)
    mu2 = F.conv2d(y_pred, window, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(y_true * y_true, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(y_pred * y_pred, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(y_true * y_pred, window, padding=window_size // 2) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, y_true, y_pred):
        y_true_3ch = y_true.repeat(1, 3, 1, 1)
        y_pred_3ch = y_pred.repeat(1, 3, 1, 1)
        
        features_true = self.vgg(y_true_3ch)
        features_pred = self.vgg(y_pred_3ch)
        
        return F.mse_loss(features_pred, features_true)


def edge_loss(y_true, y_pred):
    # 确保输入是4D张量，并且是单通道
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
    if y_pred.dim() == 3:
        y_pred = y_pred.unsqueeze(1)
    
    # 如果有多通道，只取第一个通道或者平均
    if y_true.size(1) > 1:
        y_true = y_true.mean(dim=1, keepdim=True)
    if y_pred.size(1) > 1:
        y_pred = y_pred.mean(dim=1, keepdim=True)
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(y_true.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(y_true.device)
    
    edges_true_x = F.conv2d(y_true, sobel_x, padding=1)
    edges_true_y = F.conv2d(y_true, sobel_y, padding=1)
    edges_true = torch.sqrt(edges_true_x**2 + edges_true_y**2)
    
    edges_pred_x = F.conv2d(y_pred, sobel_x, padding=1)
    edges_pred_y = F.conv2d(y_pred, sobel_y, padding=1)
    edges_pred = torch.sqrt(edges_pred_x**2 + edges_pred_y**2)
    
    return F.mse_loss(edges_pred, edges_true)


class AdvancedLoss(nn.Module):
    def __init__(self, device, w_pcc=0.25, w_ssim=0.25, w_percep=0.35, w_edge=0.15):
        super().__init__()
        self.w_pcc = w_pcc
        self.w_ssim = w_ssim
        self.w_percep = w_percep
        self.w_edge = w_edge
        self.perceptual = PerceptualLoss(device)

    def forward(self, y_pred, y_true):
        # 确保维度正确：应该是 [batch, 1, height, width]
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(1)
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)
        
        # 调试信息：打印维度（在第一次调用时）
        if not hasattr(self, '_debug_printed'):
            print(f"Debug - y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
            self._debug_printed = True
        
        loss_pcc = pcc_loss(y_true, y_pred)
        loss_ssim = ssim_loss(y_true, y_pred)
        loss_percep = self.perceptual(y_true, y_pred)
        loss_edge = edge_loss(y_true, y_pred)

        total_loss = (self.w_pcc * loss_pcc + 
                     self.w_ssim * loss_ssim + 
                     self.w_percep * loss_percep + 
                     self.w_edge * loss_edge)

        return {
            'total_loss': total_loss,
            'pcc_loss': loss_pcc,
            'ssim_loss': loss_ssim,
            'perceptual_loss': loss_percep,
            'edge_loss': loss_edge
        }


# ==================== Training & Evaluation ====================

def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, scaler=None):
    model.train()
    epoch_losses = []
    epoch_start = time.time()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        batch_start = time.time()
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        forward_start = time.time()
        # 使用混合精度前向传播
        if scaler is not None:
            with autocast():
                outputs = model(inputs)
                loss_dict = loss_fn(outputs, targets)
        else:
            outputs = model(inputs)
            loss_dict = loss_fn(outputs, targets)
        forward_time = time.time() - forward_start
        
        loss_start = time.time()
        loss_time = time.time() - loss_start
        
        backward_start = time.time()
        optimizer.zero_grad()
        
        # 使用混合精度反向传播
        if scaler is not None:
            scaler.scale(loss_dict['total_loss']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict['total_loss'].backward()
            optimizer.step()
        backward_time = time.time() - backward_start
        
        batch_time = time.time() - batch_start
        
        epoch_losses.append(loss_dict['total_loss'].item())
        
        # 每10个batch显示一次详细时间信息
        if batch_idx % 10 == 0:
            progress_bar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}",
                'Batch_time': f"{batch_time:.3f}s",
                'Forward': f"{forward_time:.3f}s",
                'Backward': f"{backward_time:.3f}s"
            })
    
    epoch_time = time.time() - epoch_start
    avg_loss = np.mean(epoch_losses)
    
    return {
        'total_loss': avg_loss,
        'epoch_time': epoch_time,
        'batches_per_second': len(train_loader) / epoch_time
    }


def evaluate(model, val_loader, device, eval_at_64=False):
    model.eval()
    eval_start = time.time()
    
    total_pcc = 0
    total_ssim = 0
    total_mse = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            inference_start = time.time()
            outputs = model(inputs)
            inference_time = time.time() - inference_start
            
            if eval_at_64:
                # Resize from 256x256 to 64x64 for evaluation
                outputs = F.interpolate(outputs, size=(64, 64), mode='bilinear', align_corners=False)
                targets = F.interpolate(targets.unsqueeze(1), size=(64, 64), mode='bilinear', align_corners=False).squeeze(1)
            
            # Calculate metrics
            for i in range(outputs.size(0)):
                pred_np = outputs[i, 0].cpu().numpy().flatten()
                true_np = targets[i].cpu().numpy().flatten()
                
                pcc_val, _ = pearsonr(pred_np, true_np)
                total_pcc += pcc_val if not np.isnan(pcc_val) else 0
                
                # SSIM calculation
                pred_img = outputs[i, 0].cpu().numpy()
                true_img = targets[i].cpu().numpy()
                ssim_val = calculate_ssim(true_img, pred_img)
                total_ssim += ssim_val
                
                # MSE
                mse_val = F.mse_loss(outputs[i, 0], targets[i]).item()
                total_mse += mse_val
            
            num_batches += outputs.size(0)
    
    eval_time = time.time() - eval_start
    
    return {
        'pcc': total_pcc / num_batches,
        'ssim': total_ssim / num_batches,
        'mse': total_mse / num_batches,
        'eval_time': eval_time,
        'samples_per_second': num_batches / eval_time
    }


def calculate_ssim(img1, img2, window_size=11, sigma=1.5):
    """Simple SSIM calculation for single channel images"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
    return ssim


# ==================== Main Experiment ====================

def run_exp2_baseline(train_loader, val_loader, test_loader, device):
    """Run Exp2 baseline with comprehensive time monitoring and RTX 4090 optimizations"""
    
    print(f"\n{'=' * 80}")
    print("EXP2 BASELINE: 1007 model (256x256) evaluated at 64x64")
    print("RTX 4090 Optimized Version with Mixed Precision")
    print(f"{'=' * 80}")
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(BASE_DIR, f"exp2_baseline_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results directory: {results_dir}")
    
    # 模型初始化时间
    model_init_start = time.time()
    model = UNetPro256(base=48).to(device)
    model_init_time = time.time() - model_init_start
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=60)
    
    # 初始化混合精度训练
    scaler = GradScaler() if device.type == 'cuda' else None
    
    loss_fn = AdvancedLoss(
        device,
        w_pcc=0.25,
        w_ssim=0.25, 
        w_percep=0.35,
        w_edge=0.15
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Model initialization time: {model_init_time:.3f}s")
    print(f"Mixed precision training: {'Enabled' if scaler else 'Disabled'}")
    print(f"Loss weights: PCC=0.25, SSIM=0.25, Percep=0.35, Edge=0.15\n")
    
    # 训练时间记录
    time_log = {
        'model_init_time': model_init_time,
        'mixed_precision': scaler is not None,
        'batch_sizes': {
            'train': train_loader.batch_size,
            'val': val_loader.batch_size,
            'test': test_loader.batch_size
        },
        'epoch_times': [],
        'training_metrics': [],
        'validation_metrics': []
    }
    
    best_pcc = -1.0
    patience_counter = 0
    total_training_start = time.time()
    
    # 训练循环
    for epoch in range(60):
        epoch_total_start = time.time()
        
        current_lr = scheduler.step(epoch)
        
        # 训练一个epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, scaler)
        
        # 验证
        val_metrics = evaluate(model, val_loader, device, eval_at_64=True)
        
        epoch_total_time = time.time() - epoch_total_start
        
        # 记录时间信息
        epoch_info = {
            'epoch': epoch + 1,
            'total_time': epoch_total_time,
            'train_time': train_metrics['epoch_time'],
            'val_time': val_metrics['eval_time'],
            'batches_per_second': train_metrics['batches_per_second'],
            'val_samples_per_second': val_metrics['samples_per_second'],
            'lr': current_lr,
            'train_loss': train_metrics['total_loss'],
            'val_pcc': val_metrics['pcc'],
            'val_ssim': val_metrics['ssim']
        }
        
        time_log['epoch_times'].append(epoch_info)
        
        # 更新最佳模型（不保存，只记录）
        if val_metrics['pcc'] > best_pcc:
            best_pcc = val_metrics['pcc']
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 输出进度（每5个epoch或最后一个epoch）
        if epoch % 5 == 0 or epoch == 59:
            print(f"Epoch {epoch + 1:3d}/60: "
                  f"Loss={train_metrics['total_loss']:.4f} | "
                  f"Val PCC={val_metrics['pcc']:.4f} (best={best_pcc:.4f}) | "
                  f"SSIM={val_metrics['ssim']:.4f} | "
                  f"Time={epoch_total_time:.1f}s | "
                  f"Train={train_metrics['epoch_time']:.1f}s | "
                  f"Val={val_metrics['eval_time']:.1f}s | "
                  f"LR={current_lr:.6f}")
        
        # 早停
        if patience_counter >= 25:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    total_training_time = time.time() - total_training_start
    
    # 最终测试
    print("\nRunning final test evaluation...")
    test_start = time.time()
    test_metrics = evaluate(model, test_loader, device, eval_at_64=True)
    test_time = time.time() - test_start
    
    # 结果总结
    print(f"\n{'=' * 60}")
    print("EXP2 BASELINE RESULTS (RTX 4090 Optimized)")
    print(f"{'=' * 60}")
    print(f"Best Validation PCC: {best_pcc:.4f}")
    print(f"Test PCC:           {test_metrics['pcc']:.4f}")
    print(f"Test SSIM:          {test_metrics['ssim']:.4f}")
    print(f"Test MSE:           {test_metrics['mse']:.4f}")
    print(f"\nTIMING SUMMARY:")
    print(f"Model init:         {model_init_time:.3f}s")
    print(f"Total training:     {total_training_time / 3600:.2f}h ({total_training_time:.1f}s)")
    print(f"Final test:         {test_time:.3f}s")
    print(f"Average epoch:      {np.mean([t['total_time'] for t in time_log['epoch_times']]):.1f}s")
    print(f"Average train/epoch: {np.mean([t['train_time'] for t in time_log['epoch_times']]):.1f}s")
    print(f"Average val/epoch:   {np.mean([t['val_time'] for t in time_log['epoch_times']]):.1f}s")
    print(f"Mixed precision:     {'Enabled' if scaler else 'Disabled'}")
    
    # 保存详细时间日志
    time_log.update({
        'total_training_time': total_training_time,
        'test_time': test_time,
        'best_val_pcc': float(best_pcc),
        'final_test_metrics': {
            'pcc': float(test_metrics['pcc']),
            'ssim': float(test_metrics['ssim']),
            'mse': float(test_metrics['mse'])
        }
    })
    
    # 保存结果
    with open(os.path.join(results_dir, 'time_log.json'), 'w') as f:
        json.dump(time_log, f, indent=2)
    
    # 生成时间分析报告
    generate_time_report(time_log, results_dir)
    
    return time_log


def generate_time_report(time_log, results_dir):
    """生成详细的时间分析报告"""
    
    report = []
    report.append("EXP2 BASELINE - RTX 4090 OPTIMIZED TIME ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 优化配置信息
    report.append("OPTIMIZATION SETTINGS:")
    report.append(f"  Mixed precision training: {time_log.get('mixed_precision', False)}")
    report.append(f"  Training batch size:      {time_log['batch_sizes']['train']}")
    report.append(f"  Validation batch size:    {time_log['batch_sizes']['val']}")
    report.append(f"  Test batch size:          {time_log['batch_sizes']['test']}")
    report.append("")
    
    # 总体统计
    report.append("OVERALL STATISTICS:")
    report.append(f"  Model initialization: {time_log['model_init_time']:.3f}s")
    report.append(f"  Total training time:  {time_log['total_training_time'] / 3600:.2f}h")
    report.append(f"  Test evaluation time: {time_log['test_time']:.3f}s")
    report.append("")
    
    # Epoch统计
    epoch_times = [t['total_time'] for t in time_log['epoch_times']]
    train_times = [t['train_time'] for t in time_log['epoch_times']]
    val_times = [t['val_time'] for t in time_log['epoch_times']]
    
    report.append("PER-EPOCH STATISTICS:")
    report.append(f"  Average epoch time:    {np.mean(epoch_times):.1f}s (±{np.std(epoch_times):.1f}s)")
    report.append(f"  Average training time: {np.mean(train_times):.1f}s (±{np.std(train_times):.1f}s)")
    report.append(f"  Average validation time: {np.mean(val_times):.1f}s (±{np.std(val_times):.1f}s)")
    report.append(f"  Min epoch time:        {np.min(epoch_times):.1f}s")
    report.append(f"  Max epoch time:        {np.max(epoch_times):.1f}s")
    report.append("")
    
    # 性能统计
    batch_speeds = [t['batches_per_second'] for t in time_log['epoch_times']]
    val_speeds = [t['val_samples_per_second'] for t in time_log['epoch_times']]
    
    report.append("THROUGHPUT STATISTICS (RTX 4090):")
    report.append(f"  Training batches/sec:   {np.mean(batch_speeds):.2f} (±{np.std(batch_speeds):.2f})")
    report.append(f"  Validation samples/sec: {np.mean(val_speeds):.2f} (±{np.std(val_speeds):.2f})")
    report.append(f"  Peak training speed:    {np.max(batch_speeds):.2f} batches/sec")
    report.append(f"  Peak validation speed:  {np.max(val_speeds):.2f} samples/sec")
    report.append("")
    
    # 最终结果
    report.append("FINAL RESULTS:")
    report.append(f"  Best validation PCC: {time_log['best_val_pcc']:.4f}")
    report.append(f"  Test PCC:           {time_log['final_test_metrics']['pcc']:.4f}")
    report.append(f"  Test SSIM:          {time_log['final_test_metrics']['ssim']:.4f}")
    report.append(f"  Test MSE:           {time_log['final_test_metrics']['mse']:.4f}")
    
    # 保存报告
    with open(os.path.join(results_dir, 'time_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nDetailed time report saved to: {os.path.join(results_dir, 'time_report.txt')}")


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("EXP2 BASELINE ONLY - RTX 4090 GPU Performance Testing")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Mixed Precision: Supported")
    print()
    
    if not torch.cuda.is_available():
        print("WARNING: GPU not detected, running on CPU")
    
    # 数据文件路径 - 请根据实际情况修改
    speckle_file = os.path.join(BASE_DIR, "original", "speckles6000_og.npy")
    pattern_file = os.path.join(BASE_DIR, "original", "pattern.npy")
    
    # 检查文件是否存在
    if not os.path.exists(speckle_file):
        print(f"ERROR: Speckle file not found: {speckle_file}")
        print("Please check the file path and update BASE_DIR")
        return
    
    if not os.path.exists(pattern_file):
        print(f"ERROR: Pattern file not found: {pattern_file}")
        print("Please check the file path and update BASE_DIR")
        return
    
    # 数据分割
    total_samples = 2000
    train_size = int(0.8 * total_samples)  # 1600
    val_size = int(0.1 * total_samples)    # 200
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_samples))
    
    print(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # 创建数据集 (256x256 for exp2)
    data_load_start = time.time()
    train_dataset = Dataset256(speckle_file, pattern_file, train_indices)
    val_dataset = Dataset256(speckle_file, pattern_file, val_indices)
    test_dataset = Dataset256(speckle_file, pattern_file, test_indices)
    
    # RTX 4090优化：更大的batch size和更多workers
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=12, pin_memory=True)
    
    data_load_time = time.time() - data_load_start
    print(f"Data loading time: {data_load_time:.3f}s\n")
    
    # 运行实验
    total_start = time.time()
    time_log = run_exp2_baseline(train_loader, val_loader, test_loader, device)
    total_time = time.time() - total_start
    
    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETED")
    print(f"{'=' * 80}")
    print(f"Total execution time: {total_time / 3600:.2f}h ({total_time:.1f}s)")
    print(f"GPU utilization test completed successfully!")


if __name__ == "__main__":
    try:
        main()
        print("\n✓ Exp2 baseline completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
