#!/usr/bin/env python3
"""
Two New Experiments on Exp2 Baseline (256→64)
Baseline (Exp1) already completed, so we run:

Exp2: Input Speckle Mask (based on radial analysis, r=130 on 256×256)
Exp3: Larger Model (base=56)
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
import cv2

BASE_DIR = r"D:\yhh\facedataset_0825"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_speckle_mask(size=256, center=(128, 128), r_effective=108, r_noise=130, soft_edge=10):
    """
    Create mask for 256×256 speckle input based on radial intensity analysis

    Based on analysis:
    - r=108px: 50% brightness → main information region
    - r=126px: 10% brightness → edge starts
    - r≥130px: <6.55 brightness → noise region

    Args:
        r_effective: Main information radius (108px)
        r_noise: Noise threshold radius (130px)
        soft_edge: Soft transition width
    """
    Y, X = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    radius = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)

    mask = np.ones_like(radius, dtype=np.float32)

    # Central region: full weight
    # r < r_effective: weight = 1.0

    # Transition zone: linear decay
    transition = (radius >= r_effective) & (radius < r_noise)
    mask[transition] = 1.0 - (radius[transition] - r_effective) / (r_noise - r_effective)

    # Noise region: soft exponential decay to 0.1
    outer = radius >= r_noise
    soft_decay = np.exp(-((radius[outer] - r_noise) / soft_edge) ** 2)
    mask[outer] = soft_decay * 0.1

    return mask


# ==================== Dataset ====================
class Dataset256(Dataset):
    """256×256 GT (upsampled from 64×64)"""

    def __init__(self, speckles_path, patterns_path, indices,
                 pol_channel=2, color_channel=2, max_value=255,
                 apply_input_mask=False):
        self.speckles_mmap = np.load(speckles_path, mmap_mode='r')
        self.patterns_mmap = np.load(patterns_path, mmap_mode='r')
        self.indices = indices
        self.pol_channel = pol_channel
        self.color_channel = color_channel
        self.max_value = max_value
        self.apply_input_mask = apply_input_mask

        # Create speckle mask if needed
        if self.apply_input_mask:
            self.speckle_mask = create_speckle_mask()
            print(f"  Input mask enabled: r_effective=108px, r_noise=130px")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        speckle_idx = original_idx * 3 + self.color_channel

        speckle = self.speckles_mmap[speckle_idx, self.pol_channel].astype(np.float32).copy()
        pattern = self.patterns_mmap[speckle_idx].astype(np.float32).copy()

        speckle = speckle / 255.0

        # Apply speckle mask if enabled
        if self.apply_input_mask:
            speckle = speckle * self.speckle_mask

        pattern = pattern / float(self.max_value)

        # Upsample to 256×256
        pattern = cv2.resize(pattern, (256, 256), interpolation=cv2.INTER_LINEAR)

        x = torch.from_numpy(speckle).unsqueeze(0).float()
        gt = torch.from_numpy(pattern).float()
        return x, gt


# ==================== Model ====================
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
    """Output 256×256, then downsample to 64×64"""

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

        u5 = self.up5(b)
        d5 = self.dec5(torch.cat([u5, self.att5(u5, e5)], dim=1))

        u4 = self.up4(d5)
        d4 = self.dec4(torch.cat([u4, self.att4(u4, e4)], dim=1))

        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, self.att3(u3, e3)], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, self.att2(u2, e2)], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, self.att1(u1, e1)], dim=1))

        return torch.sigmoid(self.final(d1))


# ==================== Loss ====================
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


class AdvancedLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        try:
            vgg = models.vgg19(weights='IMAGENET1K_V1').features
            self.vgg_slice1 = vgg[:4].to(device).eval()
            self.vgg_slice2 = vgg[:9].to(device).eval()
            self.vgg_slice3 = vgg[:18].to(device).eval()
            self.vgg_slice4 = vgg[:27].to(device).eval()

            for param in [*self.vgg_slice1.parameters(), *self.vgg_slice2.parameters(),
                          *self.vgg_slice3.parameters(), *self.vgg_slice4.parameters()]:
                param.requires_grad = False

            self.use_perceptual = True
        except:
            self.use_perceptual = False

        self.ssim = SSIMLoss()

        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

        self.w_pcc = 0.25
        self.w_ssim = 0.25
        self.w_percep = 0.35
        self.w_edge = 0.15

    def pcc_loss(self, pred, target, eps=1e-8):
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        target_mean = target_flat.mean(dim=1, keepdim=True)
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean

        pred_std = torch.sqrt((pred_centered ** 2).mean(dim=1, keepdim=True) + eps)
        target_std = torch.sqrt((target_centered ** 2).mean(dim=1, keepdim=True) + eps)

        correlation = (pred_centered * target_centered).mean(dim=1, keepdim=True) / (pred_std * target_std + eps)
        return 1 - correlation.mean()

    def edge_loss(self, pred, target):
        sobel_x = self.sobel_x.to(pred.device).type(pred.dtype)
        sobel_y = self.sobel_y.to(pred.device).type(pred.dtype)

        pred_ex = F.conv2d(pred, sobel_x, padding=1)
        pred_ey = F.conv2d(pred, sobel_y, padding=1)
        target_ex = F.conv2d(target, sobel_x, padding=1)
        target_ey = F.conv2d(target, sobel_y, padding=1)
        return F.l1_loss(pred_ex, target_ex) + F.l1_loss(pred_ey, target_ey)

    def perceptual_loss(self, pred, target):
        with torch.cuda.amp.autocast(enabled=False):
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
        """
        pred: 256×256 output from model
        target: 256×256 GT
        Downsample to 64×64 for loss calculation (no output mask)
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Downsample to 64×64 for loss calculation
        pred_64 = F.adaptive_avg_pool2d(pred, (64, 64))
        target_64 = F.adaptive_avg_pool2d(target, (64, 64))

        loss_pcc = self.pcc_loss(pred_64, target_64)
        loss_ssim = self.ssim(pred_64, target_64)
        loss_edge = self.edge_loss(pred_64, target_64)

        if self.use_perceptual:
            loss_percep = self.perceptual_loss(pred_64, target_64)
            total = (self.w_pcc * loss_pcc + self.w_ssim * loss_ssim +
                     self.w_percep * loss_percep + self.w_edge * loss_edge)
            components = {
                'pcc': float(loss_pcc.item()),
                'ssim': float(loss_ssim.item()),
                'perceptual': float(loss_percep.item()),
                'edge': float(loss_edge.item())
            }
        else:
            total = (0.40 * loss_pcc + 0.40 * loss_ssim + 0.20 * loss_edge)
            components = {
                'pcc': float(loss_pcc.item()),
                'ssim': float(loss_ssim.item()),
                'edge': float(loss_edge.item())
            }

        return total, components


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ==================== Training ====================
def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0.0
    components_sum = {}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
    for x, gt in pbar:
        x = x.to(device)
        gt = gt.to(device)
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)

        optimizer.zero_grad()
        pred = model(x)
        loss, components = loss_fn(pred, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in components.items():
            components_sum[k] = components_sum.get(k, 0.0) + v

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    pbar.close()
    num_batches = len(train_loader)
    return {
        'total_loss': total_loss / num_batches,
        'components': {k: v / num_batches for k, v in components_sum.items()}
    }


def evaluate(model, val_loader, device):
    """Evaluate on 64×64"""
    model.eval()
    total_pcc = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    num_samples = 0

    ssim_fn = SSIMLoss()

    with torch.no_grad():
        for x, gt in val_loader:
            x = x.to(device)
            gt = gt.to(device)
            if gt.dim() == 3:
                gt = gt.unsqueeze(1)

            pred = model(x)

            # Downsample to 64×64 for evaluation
            pred_64 = F.adaptive_avg_pool2d(pred, (64, 64))
            gt_64 = F.adaptive_avg_pool2d(gt, (64, 64))

            total_mse += F.mse_loss(pred_64, gt_64).item() * x.size(0)
            total_ssim += (1 - ssim_fn(pred_64, gt_64)).item() * x.size(0)

            for i in range(pred.shape[0]):
                pred_np = pred_64[i, 0].cpu().numpy().flatten()
                gt_np = gt_64[i, 0].cpu().numpy().flatten()

                try:
                    pcc_val, _ = pearsonr(pred_np, gt_np)
                    if not np.isnan(pcc_val):
                        total_pcc += pcc_val
                        num_samples += 1
                except:
                    pass

    return {
        'pcc': total_pcc / max(num_samples, 1),
        'ssim': total_ssim / len(val_loader.dataset),
        'mse': total_mse / len(val_loader.dataset)
    }


# ==================== Single Experiment ====================
def run_experiment(exp_name, exp_config, train_loader, val_loader, test_loader, device):
    """Run a single experiment"""
    print(f"\n{'=' * 80}")
    print(f"{exp_name}: {exp_config['description']}")
    print(f"{'=' * 80}")

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(BASE_DIR, f"{exp_name.lower()}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Create model
    model = UNetPro256(base=exp_config['base']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=60)
    loss_fn = AdvancedLoss(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Base channels: {exp_config['base']}")
    print(f"Input mask: {exp_config['apply_input_mask']}")
    print(f"Training: 256×256 → Eval: 64×64\n")

    best_pcc = -1.0
    best_model_state = None
    patience_counter = 0

    start_time = time.time()

    # Training loop
    for epoch in range(60):
        current_lr = scheduler.step(epoch)
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        val_metrics = evaluate(model, val_loader, device)

        if val_metrics['pcc'] > best_pcc:
            best_pcc = val_metrics['pcc']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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

    # Load best and test
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, os.path.join(exp_dir, "best_model.pth"))

    test_metrics = evaluate(model, test_loader, device)

    elapsed = time.time() - start_time

    # Results
    print(f"\n{exp_name} Results:")
    print(f"  Best Val PCC: {best_pcc:.4f}")
    print(f"  Test PCC:     {test_metrics['pcc']:.4f}")
    print(f"  Test SSIM:    {test_metrics['ssim']:.4f}")
    print(f"  Time:         {elapsed / 3600:.2f}h")

    # Save report
    report = {
        'experiment': exp_name,
        'config': exp_config,
        'timestamp': timestamp,
        'best_val_pcc': float(best_pcc),
        'test_pcc': float(test_metrics['pcc']),
        'test_ssim': float(test_metrics['ssim']),
        'test_mse': float(test_metrics['mse']),
        'time_hours': float(elapsed / 3600),
        'target_achieved': bool(test_metrics['pcc'] >= 0.900)
    }

    with open(os.path.join(exp_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    return report


# ==================== Main ====================
def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("Two New Experiments on Exp2 Baseline")
    print("Baseline already completed - running Exp2 & Exp3")
    print("=" * 80)
    print(f"Device: {device}\n")

    if not torch.cuda.is_available():
        print("ERROR: GPU not detected")
        return

    # Data files
    speckle_file = os.path.join(BASE_DIR, "original", "speckles6000_og.npy")
    pattern_file = os.path.join(BASE_DIR, "original", "pattern.npy")

    if not os.path.exists(speckle_file) or not os.path.exists(pattern_file):
        print("ERROR: Data files not found")
        return

    # Data split
    total_samples = 2000
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_samples))

    print(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}\n")

    # Experiment configurations
    experiments = [
        {
            'name': 'Exp2',
            'description': 'Input Speckle Mask (r=130, base=48)',
            'base': 48,
            'apply_input_mask': True
        },
        {
            'name': 'Exp3',
            'description': 'Larger Model (base=56, no mask)',
            'base': 56,
            'apply_input_mask': False
        }
    ]

    # Run all experiments
    all_results = []
    start_total = time.time()

    for exp in experiments:
        # Create datasets with appropriate masking
        train_dataset = Dataset256(speckle_file, pattern_file, train_indices,
                                   apply_input_mask=exp['apply_input_mask'])
        val_dataset = Dataset256(speckle_file, pattern_file, val_indices,
                                 apply_input_mask=exp['apply_input_mask'])
        test_dataset = Dataset256(speckle_file, pattern_file, test_indices,
                                  apply_input_mask=exp['apply_input_mask'])

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=6, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)

        result = run_experiment(
            exp['name'],
            exp,
            train_loader,
            val_loader,
            test_loader,
            device
        )
        all_results.append(result)

    total_time = time.time() - start_total

    # Final summary
    print(f"\n{'=' * 80}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'=' * 80}")
    print(f"Total time: {total_time / 3600:.2f}h\n")

    # Add baseline result for comparison (from 1010 experiment)
    print(f"{'Experiment':<15} {'Test PCC':<12} {'Test SSIM':<12} {'Time (h)':<10} {'Target'}")
    print("-" * 80)
    print(f"{'Exp1 (Baseline)':<15} {'0.8964':<12} {'0.6341':<12} {'2.75':<10} {'✗'}")

    for result in all_results:
        target_mark = "✓" if result['target_achieved'] else "✗"
        print(f"{result['experiment']:<15} "
              f"{result['test_pcc']:<12.4f} "
              f"{result['test_ssim']:<12.4f} "
              f"{result['time_hours']:<10.2f} "
              f"{target_mark}")

    # Save summary
    summary = {
        'total_time_hours': float(total_time / 3600),
        'baseline_pcc': 0.8964,
        'experiments': all_results,
        'note': 'Exp1 (baseline) from 1010 experiment: PCC=0.8964, SSIM=0.6341'
    }

    summary_path = os.path.join(BASE_DIR, f"exp2_variants_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    try:
        main()
        print("\n✓ All experiments completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()