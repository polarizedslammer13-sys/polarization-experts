#!/usr/bin/env python3
"""
1010 Exp2 Baseline Only - 严格原版实现
只修改路径，其他完全按照原版
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

# 只修改这个路径
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


# ==================== Loss Functions (原版) ====================

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.window = self._create_window(window_size)

    def _gaussian(self, window_size, sigma=1.5):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                              for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size):
        _1D_window = self._gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window

    def forward(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


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


class AdvancedLoss(nn.Module):
    def __init__(self, device, w_pcc=0.25, w_ssim=0.25, w_percep=0.35, w_edge=0.15):
        super().__init__()
        self.device = device
        self.w_pcc = w_pcc
        self.w_ssim = w_ssim
        self.w_percep = w_percep
        self.w_edge = w_edge
        
        # 原版的感知损失
        vgg = models.vgg16(weights='DEFAULT').features[:16].to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg_slice1 = vgg[:4]
        self.vgg_slice2 = vgg[4:9]
        self.vgg_slice3 = vgg[9:16]
        self.vgg_slice4 = vgg[16:23] if len(vgg) > 16 else nn.Identity()
        
        self.ssim = SSIMLoss()

    def pcc_loss(self, pred, target):
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        pred_mean = torch.mean(pred_flat, dim=1, keepdim=True)
        target_mean = torch.mean(target_flat, dim=1, keepdim=True)
        
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        numerator = torch.sum(pred_centered * target_centered, dim=1)
        denominator = torch.sqrt(torch.sum(pred_centered ** 2, dim=1) * torch.sum(target_centered ** 2, dim=1))
        
        pcc = numerator / (denominator + 1e-8)
        return 1 - torch.mean(pcc)

    def perceptual_loss(self, pred, target):
        with torch.amp.autocast('cuda', enabled=False):
            pred_fp32 = pred.float()
            target_fp32 = target.float()

            pred_3ch = pred_fp32.repeat(1, 3, 1, 1)
            target_3ch = target_fp32.repeat(1, 3, 1, 1)

            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

            pred_norm = (pred_3ch - mean) / std
            target_norm = (target_3ch - mean) / std

            loss = 0.0
            weights = [0.1, 0.2, 0.3, 0.4]

            for layer, weight in zip([self.vgg_slice1, self.vgg_slice2,
                                      self.vgg_slice3, self.vgg_slice4], weights):
                pred_feat = layer(pred_norm)
                with torch.no_grad():
                    target_feat = layer(target_norm)
                loss += F.l1_loss(pred_feat, target_feat) * weight

            return loss

    def forward(self, pred, target):
        if target.dim() == 3:
            target = target.unsqueeze(1)

        loss_pcc = self.pcc_loss(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_percep = self.perceptual_loss(pred, target)
        
        # Edge loss
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        
        pred_edges_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2)
        
        target_edges_x = F.conv2d(target, sobel_x, padding=1)
        target_edges_y = F.conv2d(target, sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2)
        
        loss_edge = F.mse_loss(pred_edges, target_edges)

        if self.w_percep > 0:
            total = (self.w_pcc * loss_pcc + self.w_ssim * loss_ssim +
                     self.w_percep * loss_percep + self.w_edge * loss_edge)
            return {
                'total_loss': total,
                'pcc': float(loss_pcc.item()),
                'ssim': float(loss_ssim.item()),
                'perceptual': float(loss_percep.item()),
                'edge': float(loss_edge.item())
            }
        else:
            total_w = self.w_pcc + self.w_ssim + self.w_edge
            total = ((self.w_pcc / total_w) * loss_pcc + 
                     (self.w_ssim / total_w) * loss_ssim +
                     (self.w_edge / total_w) * loss_edge)
            return {
                'total_loss': total,
                'pcc': float(loss_pcc.item()),
                'ssim': float(loss_ssim.item()),
                'perceptual': 0.0,
                'edge': float(loss_edge.item())
            }


# ==================== Training & Evaluation ====================

def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_dict = loss_fn(outputs, targets)
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        total_loss += loss_dict['total_loss'].item()
        num_batches += 1
        
        progress_bar.set_postfix({'Loss': f"{loss_dict['total_loss'].item():.4f}"})
    
    return {'total_loss': total_loss / num_batches}


def evaluate(model, val_loader, device, eval_at_64=False):
    model.eval()
    total_pcc = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    num_samples = 0
    
    ssim_fn = SSIMLoss()
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if eval_at_64:
                # Resize to 64x64 for evaluation
                outputs = F.interpolate(outputs, size=(64, 64), mode='bilinear', align_corners=False)
                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)
                targets = F.interpolate(targets, size=(64, 64), mode='bilinear', align_corners=False)
            
            # Calculate metrics
            for i in range(outputs.size(0)):
                pred = outputs[i].squeeze()
                gt = targets[i].squeeze() if targets.dim() == 4 else targets[i]
                
                # PCC
                pred_flat = pred.cpu().numpy().flatten()
                gt_flat = gt.cpu().numpy().flatten()
                pcc_val, _ = pearsonr(pred_flat, gt_flat)
                total_pcc += pcc_val if not np.isnan(pcc_val) else 0
                
                # SSIM
                pred_4d = pred.unsqueeze(0).unsqueeze(0)
                gt_4d = gt.unsqueeze(0).unsqueeze(0)
                total_ssim += (1 - ssim_fn(pred_4d, gt_4d)).item()
                
                # MSE
                total_mse += F.mse_loss(pred, gt).item()
                
                num_samples += 1
    
    return {
        'pcc': total_pcc / num_samples,
        'ssim': total_ssim / num_samples,
        'mse': total_mse / num_samples,
    }


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("EXP2 BASELINE - 严格原版实现")
    print("=" * 80)
    print(f"Device: {device}\n")
    
    # Data files
    speckle_file = os.path.join(BASE_DIR, "original", "speckles6000_og.npy")
    pattern_file = os.path.join(BASE_DIR, "original", "pattern.npy")
    
    # Data split
    total_samples = 2000
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_samples))
    
    print(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}\n")
    
    # Dataset256 for exp2
    train_dataset = Dataset256(speckle_file, pattern_file, train_indices)
    val_dataset = Dataset256(speckle_file, pattern_file, val_indices)
    test_dataset = Dataset256(speckle_file, pattern_file, test_indices)
    
    # 原版batch size和workers
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)
    
    # Model
    model = UNetPro256(base=48).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=60)
    
    # 原版exp2 loss weights
    loss_fn = AdvancedLoss(
        device,
        w_pcc=0.25,
        w_ssim=0.25,
        w_percep=0.35,
        w_edge=0.15
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Loss weights: PCC=0.25, SSIM=0.25, Percep=0.35, Edge=0.15\n")
    
    best_pcc = -1.0
    patience_counter = 0
    
    # Training
    for epoch in range(60):
        current_lr = scheduler.step(epoch)
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        val_metrics = evaluate(model, val_loader, device, eval_at_64=True)  # KEY: eval_at_64=True
        
        if val_metrics['pcc'] > best_pcc:
            best_pcc = val_metrics['pcc']
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or epoch == 59:
            print(f"Epoch {epoch + 1:3d}/60: "
                  f"Loss={train_metrics['total_loss']:.4f} | "
                  f"Val PCC={val_metrics['pcc']:.4f} (best={best_pcc:.4f}) | "
                  f"SSIM={val_metrics['ssim']:.4f} | LR={current_lr:.6f}")
        
        if patience_counter >= 25:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Test
    test_metrics = evaluate(model, test_loader, device, eval_at_64=True)
    
    print(f"\nFINAL RESULTS:")
    print(f"Best Val PCC: {best_pcc:.4f}")
    print(f"Test PCC:     {test_metrics['pcc']:.4f}")
    print(f"Test SSIM:    {test_metrics['ssim']:.4f}")


if __name__ == "__main__":
    try:
        main()
        print("\n✓ Exp2 baseline completed!")
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
