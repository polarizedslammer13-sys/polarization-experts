#!/usr/bin/env python3
"""
Pattern Resolution Ablation Study - Complete Version
验证"高分辨率学习，低分辨率应用"的有效性

实验矩阵（9个实验）：
                Speckle分辨率
            64      128     256
Model   64  E1      E9      E3
Output 128  E2      E4      E5
       256  E8      E6      E7(baseline)

所有实验最终在64×64 pattern上评估
"""

import os
import sys
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import traceback
import subprocess

# ==================== Configuration ====================
AUTO_SHUTDOWN = True  # 设置为False可以禁用自动关机
BASE_DIR = "/root/autodl-tmp/facedataset_0825"

# Email Configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.qq.com',
    'smtp_port': 465,
    'sender': '1309992979@qq.com',
    'password': 'ruemdlkqminjjehb',
    'receiver': '1309992979@qq.com'
}

# SSH Info for SCP command
SSH_INFO = {
    'host': 'connect.westb.seetacloud.com',
    'port': 48053,
    'user': 'root'
}


# ==================== Email Functions ====================
def send_email(subject, body, attachments=None):
    """发送邮件，失败不影响训练"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['sender']
        msg['To'] = EMAIL_CONFIG['receiver']
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 添加附件
        if attachments:
            for file_path in attachments:
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition', 
                                      f'attachment; filename={os.path.basename(file_path)}')
                        msg.attach(part)
        
        # 发送邮件
        with smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_server'], 
                             EMAIL_CONFIG['smtp_port']) as server:
            server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
            server.send_message(msg)
        
        print(f"✓ Email sent: {subject}")
        return True
        
    except Exception as e:
        print(f"✗ Email failed: {e}")
        return False


def send_start_notification():
    """训练开始通知"""
    subject = "🚀 Training Started - Pattern Resolution Ablation"
    body = f"""
Pattern Resolution Ablation Study has started!

Configuration:
- Total Experiments: 9
- Base Directory: {BASE_DIR}
- Auto Shutdown: {AUTO_SHUTDOWN}
- Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Experiment Matrix:
E1: S64_M64    E9: S128_M64   E3: S256_M64
E2: S64_M128   E4: S128_M128  E5: S256_M128
E8: S64_M256   E6: S128_M256  E7: S256_M256 (baseline)

Estimated Time: ~20 hours
"""
    send_email(subject, body)


def send_experiment_notification(exp_name, exp_config):
    """单个实验开始通知"""
    subject = f"🔬 Experiment Started: {exp_name}"
    body = f"""
Starting {exp_name}

Configuration:
- Speckle Resolution: {exp_config['speckle_size']}×{exp_config['speckle_size']}
- Model Output: {exp_config['model_output']}×{exp_config['model_output']}
- Description: {exp_config['description']}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    send_email(subject, body)


def send_completion_notification(all_results, total_time, log_file):
    """训练完成通知"""
    subject = "✅ Training Completed - Pattern Resolution Ablation"
    
    # 构建结果摘要
    results_summary = "\n" + "="*60 + "\n"
    results_summary += "EXPERIMENT RESULTS SUMMARY\n"
    results_summary += "="*60 + "\n\n"
    
    results_summary += f"{'Experiment':<15} {'Speckle':<10} {'Model':<10} {'PCC':<10} {'SSIM':<10}\n"
    results_summary += "-"*60 + "\n"
    
    for result in all_results:
        exp_name = result['experiment']
        config = result['config']
        results_summary += (f"{exp_name:<15} "
                          f"{config['speckle_size']}×{config['speckle_size']:<10} "
                          f"{config['model_output']}×{config['model_output']:<10} "
                          f"{result['test_pcc']:<10.4f} "
                          f"{result['test_ssim']:<10.4f}\n")
    
    # 找出最优配置
    best_result = max(all_results, key=lambda x: x['test_pcc'])
    
    results_summary += "\n" + "="*60 + "\n"
    results_summary += f"Best Configuration: {best_result['experiment']}\n"
    results_summary += f"  PCC: {best_result['test_pcc']:.4f}\n"
    results_summary += f"  SSIM: {best_result['test_ssim']:.4f}\n"
    
    # 计算总时间和费用
    hours = total_time / 3600
    cost_estimate = hours * 1.99  # 假设每小时1.99元
    
    body = f"""
Pattern Resolution Ablation Study Completed!

{results_summary}

Training Statistics:
- Total Time: {hours:.2f} hours
- Estimated Cost: ¥{cost_estimate:.2f}
- Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Download Results:
scp -P {SSH_INFO['port']} -r {SSH_INFO['user']}@{SSH_INFO['host']}:{BASE_DIR}/*.json ./
scp -P {SSH_INFO['port']} {SSH_INFO['user']}@{SSH_INFO['host']}:{log_file} ./

{'⚠️  System will shutdown in 5 minutes!' if AUTO_SHUTDOWN else 'ℹ️  System will NOT shutdown (AUTO_SHUTDOWN=False)'}

All result files are attached.
"""
    
    # 收集所有报告文件
    attachments = [log_file] if os.path.exists(log_file) else []
    for result in all_results:
        report_path = os.path.join(BASE_DIR, f"{result['experiment'].lower()}_{result['timestamp']}", 'report.json')
        if os.path.exists(report_path):
            attachments.append(report_path)
    
    send_email(subject, body, attachments)


def send_error_notification(error_msg, error_traceback):
    """错误通知"""
    subject = "❌ Training Error - Pattern Resolution Ablation"
    body = f"""
Training encountered an error!

Error Message:
{error_msg}

Traceback:
{error_traceback}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check the logs and restart training.
"""
    send_email(subject, body)


# ==================== Utilities ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_speckle_mask(size=256, r_noise=130):
    """创建散斑mask"""
    Y, X = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    center = (size//2, size//2)
    radius = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    
    mask = np.ones_like(radius, dtype=np.float32)
    r_effective = int(r_noise * 108/130)  # 按比例缩放
    
    transition = (radius >= r_effective) & (radius < r_noise)
    mask[transition] = 1.0 - (radius[transition] - r_effective) / (r_noise - r_effective)
    
    outer = radius >= r_noise
    soft_decay = np.exp(-((radius[outer] - r_noise) / 10)**2)
    mask[outer] = soft_decay * 0.1
    
    return mask


# ==================== Dataset ====================
class FlexibleDataset(Dataset):
    """支持可变散斑输入分辨率的数据集"""
    
    def __init__(self, speckles_path, patterns_path, indices,
                 speckle_size=256, pol_channel=2, color_channel=2, 
                 max_value=255, apply_mask=True):
        self.speckles_mmap = np.load(speckles_path, mmap_mode='r')
        self.patterns_mmap = np.load(patterns_path, mmap_mode='r')
        self.indices = indices
        self.speckle_size = speckle_size
        self.pol_channel = pol_channel
        self.color_channel = color_channel
        self.max_value = max_value
        self.apply_mask = apply_mask
        
        # 创建mask（如果需要）
        if self.apply_mask:
            # 原始是256×256，需要调整mask大小
            if speckle_size == 256:
                r_noise = 130
            elif speckle_size == 128:
                r_noise = 65
            else:  # 64
                r_noise = 32
            self.mask = create_speckle_mask(size=speckle_size, r_noise=r_noise)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        speckle_idx = original_idx * 3 + self.color_channel
        
        # 读取256×256散斑
        speckle_256 = self.speckles_mmap[speckle_idx, self.pol_channel].astype(np.float32).copy()
        speckle_256 = speckle_256 / 255.0
        
        # 下采样散斑（如果需要）
        if self.speckle_size == 256:
            speckle = speckle_256
        elif self.speckle_size == 128:
            speckle = cv2.resize(speckle_256, (128, 128), interpolation=cv2.INTER_AREA)
        else:  # 64
            speckle = cv2.resize(speckle_256, (64, 64), interpolation=cv2.INTER_AREA)
        
        # 应用mask
        if self.apply_mask:
            speckle = speckle * self.mask
        
        # 读取64×64 pattern（GT固定）
        pattern = self.patterns_mmap[speckle_idx].astype(np.float32).copy()
        pattern = pattern / float(self.max_value)
        
        x = torch.from_numpy(speckle).unsqueeze(0).float()
        gt = torch.from_numpy(pattern).float()
        
        return x, gt


# ==================== Model Components ====================
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


# ==================== Flexible UNet ====================
class FlexibleUNet(nn.Module):
    """
    灵活的UNet，支持不同的输入和输出分辨率
    
    输入: speckle_size × speckle_size
    输出: output_size × output_size
    评估: 统一在64×64
    """
    
    def __init__(self, in_channels=1, base=48, speckle_size=256, output_size=256):
        super().__init__()
        self.speckle_size = speckle_size
        self.output_size = output_size
        
        # 计算encoder需要几层pooling
        # 目标：下采样到8×8
        self.num_encoder_pools = int(np.log2(speckle_size / 8))
        
        # 计算decoder需要几层upsampling
        self.num_decoder_ups = int(np.log2(output_size / 8))
        
        # 构建encoder
        self._build_encoder(base)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            DoubleConv(self.bottleneck_in_channels, base * 32, residual=True),
            ResidualBlock(base * 32)
        )
        
        # 构建decoder
        self._build_decoder(base)
        
    def _build_encoder(self, base):
        """根据speckle_size构建encoder"""
        # 始终构建最多5层encoder，但可能不全用
        self.enc1 = DoubleConv(1, base, residual=False)
        self.pool1 = nn.MaxPool2d(2)
        
        if self.num_encoder_pools >= 2:
            self.enc2 = DoubleConv(base, base*2, residual=True)
            self.pool2 = nn.MaxPool2d(2)
        
        if self.num_encoder_pools >= 3:
            self.enc3 = DoubleConv(base*2, base*4, residual=True)
            self.pool3 = nn.MaxPool2d(2)
        
        if self.num_encoder_pools >= 4:
            self.enc4 = DoubleConv(base*4, base*8, residual=True)
            self.pool4 = nn.MaxPool2d(2)
        
        if self.num_encoder_pools >= 5:
            self.enc5 = DoubleConv(base*8, base*16, residual=True)
            self.pool5 = nn.MaxPool2d(2)
        
        # 确定bottleneck的输入通道数
        if self.num_encoder_pools == 3:
            self.bottleneck_in_channels = base * 4
        elif self.num_encoder_pools == 4:
            self.bottleneck_in_channels = base * 8
        else:  # 5
            self.bottleneck_in_channels = base * 16
    
    def _build_decoder(self, base):
        """根据output_size构建decoder"""
        # bottleneck输出是base*32
        
        if self.num_decoder_ups >= 5:
            self.up5 = nn.ConvTranspose2d(base*32, base*16, 2, stride=2)
            self.att5 = AttentionGate(F_g=base*16, F_l=base*16, F_int=base*8)
            self.dec5 = DoubleConv(base*32, base*16, residual=True)
        
        if self.num_decoder_ups >= 4:
            up4_in = base*16 if self.num_decoder_ups >= 5 else base*32
            self.up4 = nn.ConvTranspose2d(up4_in, base*8, 2, stride=2)
            self.att4 = AttentionGate(F_g=base*8, F_l=base*8, F_int=base*4)
            self.dec4 = DoubleConv(base*16, base*8, residual=True)
        
        if self.num_decoder_ups >= 3:
            up3_in = base*8 if self.num_decoder_ups >= 4 else base*32
            self.up3 = nn.ConvTranspose2d(up3_in, base*4, 2, stride=2)
            self.att3 = AttentionGate(F_g=base*4, F_l=base*4, F_int=base*2)
            self.dec3 = DoubleConv(base*8, base*4, residual=True)
        
        if self.num_decoder_ups >= 2:
            up2_in = base*4 if self.num_decoder_ups >= 3 else base*32
            self.up2 = nn.ConvTranspose2d(up2_in, base*2, 2, stride=2)
            self.att2 = AttentionGate(F_g=base*2, F_l=base*2, F_int=base)
            self.dec2 = DoubleConv(base*4, base*2, residual=True)
        
        # 最后一层decoder
        up1_in = base*2 if self.num_decoder_ups >= 2 else base*32
        self.up1 = nn.ConvTranspose2d(up1_in, base, 2, stride=2)
        self.att1 = AttentionGate(F_g=base, F_l=base, F_int=base//2)
        self.dec1 = DoubleConv(base*2, base, residual=False)
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(base, base//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base//2, 1, 1)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        
        if self.num_encoder_pools >= 2:
            e2 = self.enc2(self.pool1(e1))
        
        if self.num_encoder_pools >= 3:
            e3 = self.enc3(self.pool2(e2))
        
        if self.num_encoder_pools >= 4:
            e4 = self.enc4(self.pool3(e3))
        
        if self.num_encoder_pools >= 5:
            e5 = self.enc5(self.pool4(e4))
            bottleneck_in = self.pool5(e5)
        elif self.num_encoder_pools == 4:
            bottleneck_in = self.pool4(e4)
        else:  # 3
            bottleneck_in = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(bottleneck_in)
        
        # Decoder
        if self.num_decoder_ups >= 5:
            d5 = self.dec5(torch.cat([self.up5(b), self.att5(self.up5(b), e5)], dim=1))
            dec_out = d5
        else:
            dec_out = b
        
        if self.num_decoder_ups >= 4:
            skip = e4 if self.num_encoder_pools >= 4 else e3
            d4 = self.dec4(torch.cat([self.up4(dec_out), self.att4(self.up4(dec_out), skip)], dim=1))
            dec_out = d4
        
        if self.num_decoder_ups >= 3:
            skip = e3 if self.num_encoder_pools >= 3 else e2
            d3 = self.dec3(torch.cat([self.up3(dec_out), self.att3(self.up3(dec_out), skip)], dim=1))
            dec_out = d3
        
        if self.num_decoder_ups >= 2:
            skip = e2 if self.num_encoder_pools >= 2 else e1
            d2 = self.dec2(torch.cat([self.up2(dec_out), self.att2(self.up2(dec_out), skip)], dim=1))
            dec_out = d2
        
        # Final layer
        d1 = self.dec1(torch.cat([self.up1(dec_out), self.att1(self.up1(dec_out), e1)], dim=1))
        
        return torch.sigmoid(self.final(d1))


# ==================== Loss Function ====================
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.window = self._create_window(window_size)
    
    def _gaussian(self, window_size, sigma=1.5):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2*sigma**2))
                             for x in range(window_size)])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size):
        _1D_window = self._gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window
    
    def forward(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
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
        
        pred_std = torch.sqrt((pred_centered**2).mean(dim=1, keepdim=True) + eps)
        target_std = torch.sqrt((target_centered**2).mean(dim=1, keepdim=True) + eps)
        
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
        pred: 任意分辨率输出
        target: 64×64 GT
        统一下采样到64×64计算loss
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # 下采样到64×64
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


# ==================== Training Functions ====================
def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0.0
    components_sum = {}
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
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
    """评估在64×64"""
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
            
            # 下采样到64×64评估
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
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"{exp_name}: {exp_config['description']}")
    print(f"{'='*80}")
    
    # 发送实验开始通知
    send_experiment_notification(exp_name, exp_config)
    
    # 创建目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(BASE_DIR, f"{exp_name.lower()}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # 创建模型
    model = FlexibleUNet(
        base=48,
        speckle_size=exp_config['speckle_size'],
        output_size=exp_config['model_output']
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, 
                                  weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=60)
    loss_fn = AdvancedLoss(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Speckle: {exp_config['speckle_size']}×{exp_config['speckle_size']}")
    print(f"Model Output: {exp_config['model_output']}×{exp_config['model_output']}")
    print(f"Eval: 64×64\n")
    
    best_pcc = -1.0
    best_model_state = None
    patience_counter = 0
    
    start_time = time.time()
    
    # 训练循环
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
            print(f"Epoch {epoch+1:3d}/60: "
                 f"Loss={train_metrics['total_loss']:.4f} | "
                 f"Val PCC={val_metrics['pcc']:.4f} (best={best_pcc:.4f}) | "
                 f"SSIM={val_metrics['ssim']:.4f} | LR={current_lr:.6f}")
        
        if patience_counter >= 25:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 加载最佳模型并测试
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, os.path.join(exp_dir, "best_model.pth"))
    
    test_metrics = evaluate(model, test_loader, device)
    
    elapsed = time.time() - start_time
    
    # 结果
    print(f"\n{exp_name} Results:")
    print(f"  Best Val PCC: {best_pcc:.4f}")
    print(f"  Test PCC:     {test_metrics['pcc']:.4f}")
    print(f"  Test SSIM:    {test_metrics['ssim']:.4f}")
    print(f"  Time:         {elapsed/3600:.2f}h")
    
    # 保存报告
    report = {
        'experiment': exp_name,
        'config': exp_config,
        'timestamp': timestamp,
        'best_val_pcc': float(best_pcc),
        'test_pcc': float(test_metrics['pcc']),
        'test_ssim': float(test_metrics['ssim']),
        'test_mse': float(test_metrics['mse']),
        'time_hours': float(elapsed / 3600),
        'parameters_M': float(sum(p.numel() for p in model.parameters())/1e6)
    }
    
    with open(os.path.join(exp_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


# ==================== Main ====================
def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("Pattern Resolution Ablation Study - Complete 9 Experiments")
    print("="*80)
    print(f"Device: {device}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Auto Shutdown: {AUTO_SHUTDOWN}\n")
    
    if not torch.cuda.is_available():
        print("ERROR: GPU not detected")
        send_error_notification("GPU not detected", "CUDA not available")
        return
    
    # 发送开始通知
    send_start_notification()
    
    # 数据文件
    speckle_file = os.path.join(BASE_DIR, "original", "speckles6000_og.npy")
    pattern_file = os.path.join(BASE_DIR, "original", "pattern.npy")
    
    if not os.path.exists(speckle_file) or not os.path.exists(pattern_file):
        error_msg = "Data files not found"
        print(f"ERROR: {error_msg}")
        send_error_notification(error_msg, f"Files: {speckle_file}, {pattern_file}")
        return
    
    # 数据划分
    total_samples = 2000
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_samples))
    
    print(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}\n")
    
    # 实验配置（9个实验）
    experiments = [
        {
            'name': 'E1',
            'description': 'S64_M64: Minimal matched config',
            'speckle_size': 64,
            'model_output': 64
        },
        {
            'name': 'E2',
            'description': 'S64_M128: Small speckle, medium output',
            'speckle_size': 64,
            'model_output': 128
        },
        {
            'name': 'E3',
            'description': 'S256_M64: Full speckle, direct output',
            'speckle_size': 256,
            'model_output': 64
        },
        {
            'name': 'E4',
            'description': 'S128_M128: Medium matched config',
            'speckle_size': 128,
            'model_output': 128
        },
        {
            'name': 'E5',
            'description': 'S256_M128: Full speckle, medium output',
            'speckle_size': 256,
            'model_output': 128
        },
        {
            'name': 'E6',
            'description': 'S128_M256: Medium speckle, high output',
            'speckle_size': 128,
            'model_output': 256
        },
        {
            'name': 'E7',
            'description': 'S256_M256: BASELINE - Full matched config',
            'speckle_size': 256,
            'model_output': 256
        },
        {
            'name': 'E8',
            'description': 'S64_M256: Minimal speckle, high output',
            'speckle_size': 64,
            'model_output': 256
        },
        {
            'name': 'E9',
            'description': 'S128_M64: Medium speckle, direct output',
            'speckle_size': 128,
            'model_output': 64
        }
    ]
    
    # 运行所有实验
    all_results = []
    start_total = time.time()
    
    for exp in experiments:
        try:
            # 创建对应分辨率的数据集
            train_dataset = FlexibleDataset(
                speckle_file, pattern_file, train_indices,
                speckle_size=exp['speckle_size'], apply_mask=True
            )
            val_dataset = FlexibleDataset(
                speckle_file, pattern_file, val_indices,
                speckle_size=exp['speckle_size'], apply_mask=True
            )
            test_dataset = FlexibleDataset(
                speckle_file, pattern_file, test_indices,
                speckle_size=exp['speckle_size'], apply_mask=True
            )
            
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, 
                                    num_workers=6, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, 
                                  num_workers=6, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, 
                                   num_workers=6, pin_memory=True)
            
            result = run_experiment(
                exp['name'],
                exp,
                train_loader,
                val_loader,
                test_loader,
                device
            )
            all_results.append(result)
            
        except Exception as e:
            error_msg = f"Error in {exp['name']}: {str(e)}"
            error_trace = traceback.format_exc()
            print(f"\n{error_msg}\n{error_trace}")
            send_error_notification(error_msg, error_trace)
            continue
    
    total_time = time.time() - start_total
    
    # 最终总结
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f}h\n")
    
    print(f"{'Exp':<8} {'Speckle':<10} {'Model':<10} {'PCC':<10} {'SSIM':<10} {'Time(h)':<10}")
    print("-"*80)
    
    for result in all_results:
        exp = result['experiment']
        cfg = result['config']
        print(f"{exp:<8} "
             f"{cfg['speckle_size']}×{cfg['speckle_size']:<10} "
             f"{cfg['model_output']}×{cfg['model_output']:<10} "
             f"{result['test_pcc']:<10.4f} "
             f"{result['test_ssim']:<10.4f} "
             f"{result['time_hours']:<10.2f}")
    
    # 保存汇总
    summary = {
        'total_time_hours': float(total_time / 3600),
        'experiments': all_results,
        'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    summary_path = os.path.join(BASE_DIR, f"pattern_resolution_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved: {summary_path}")
    
    # 发送完成通知
    log_file = os.path.join(BASE_DIR, "training.log")
    send_completion_notification(all_results, total_time, log_file)
    
    # 自动关机
    if AUTO_SHUTDOWN:
        print("\n" + "="*80)
        print("⚠️  System will shutdown in 5 minutes...")
        print("="*80)
        time.sleep(300)  # 等待5分钟
        subprocess.run(['shutdown', '-h', 'now'])


if __name__ == "__main__":
    try:
        # 重定向输出到日志文件
        log_file = os.path.join(BASE_DIR, "training.log")
        os.makedirs(BASE_DIR, exist_ok=True)
        
        class Logger:
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, 'w', encoding='utf-8')
            
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                self.log.flush()
            
            def flush(self):
                self.terminal.flush()
                self.log.flush()
        
        sys.stdout = Logger(log_file)
        sys.stderr = sys.stdout
        
        main()
        print("\n✓ All experiments completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted by user")
        send_error_notification("Training interrupted by user", "KeyboardInterrupt")
        
    except Exception as e:
        error_msg = f"Fatal error: {str(e)}"
        error_trace = traceback.format_exc()
        print(f"\n✗ {error_msg}\n{error_trace}")
        send_error_notification(error_msg, error_trace)
        
    finally:
        if hasattr(sys.stdout, 'log'):
            sys.stdout.log.close()
