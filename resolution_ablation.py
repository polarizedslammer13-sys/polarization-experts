#!/usr/bin/env python3
"""Advanced resolution ablation study for MMF face reconstruction."""

import argparse
import csv
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils import checkpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


BASE_DIR = "/root/autodl-tmp/facedataset_0825"
SPECKLE_FILE = os.path.join(BASE_DIR, "original", "speckles6000_og.npy")
PATTERN_FILE = os.path.join(BASE_DIR, "original", "pattern.npy")

TOTAL_SAMPLES = 2000
TRAIN_SPLIT = int(0.8 * TOTAL_SAMPLES)
VAL_SPLIT = int(0.1 * TOTAL_SAMPLES)
TEST_SPLIT = TOTAL_SAMPLES - TRAIN_SPLIT - VAL_SPLIT

TRAIN_EPOCHS = 50
WARMUP_EPOCHS = 7
EARLY_STOP_PATIENCE = None  # run full schedule by default

RESOLUTION_EXPERIMENTS = [
    {
        "name": "res_64_core",
        "speckle_input": 64,
        "model_output": 64,
        "eval_resolution": 64,
        "description": "64→64→64 (低带宽baseline，训练最快)",
        "expected_time": "≈30-40min",
        "physics_valid": True,
        "for_final_table": True,
    },
    {
        "name": "res_256_to_64",
        "speckle_input": 256,
        "model_output": 64,
        "eval_resolution": 64,
        "description": "256→64→64 (高分辨率speckle输入，低分辨率输出头)",
        "expected_time": "≈2.5h",
        "physics_valid": True,
        "for_final_table": True,
    },
    {
        "name": "res_256_native_stage",
        "speckle_input": 256,
        "model_output": 256,
        "eval_resolution": 256,
        "description": "256→256→256 (高分辨率重建，仅用于过程与可视化，不计入主表)",
        "expected_time": "≈2.5h+",
        "physics_valid": True,
        "for_final_table": False,
    },
    {
        "name": "res_256_native_to64eval",
        "speckle_input": 256,
        "model_output": 256,
        "eval_resolution": 64,
        "description": "256→256 (高分辨率中间表征)→64评估，验证更大输出头是否提升64×64指标",
        "expected_time": "≈3h",
        "physics_valid": True,
        "for_final_table": True,
    },
    {
        "name": "res_512_upscaled_probe",
        "speckle_input": 512,
        "model_output": 256,
        "eval_resolution": 64,
        "description": "512输入为256上采样，仅探究正则化与归纳偏置效应，非物理有效",
        "expected_time": "6-8h",
        "physics_valid": False,
        "for_final_table": False,
    },
]

BATCH_SIZE_RULES = {
    64: 16,
    256: 4,
    512: 2,
}


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "resolution_experiment_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_physics_valid(base_resolution: int, target_resolution: int, physics_valid: bool) -> None:
    if physics_valid and target_resolution > base_resolution:
        raise ValueError(
            "Physics-valid experiment cannot upsample speckle data (target resolution exceeds base data)."
        )


def quantize_and_normalize(pattern: np.ndarray, max_value: float = 255.0) -> np.ndarray:
    pattern = np.clip(pattern.astype(np.float32), 0, max_value)
    return pattern / max_value


class ConfigurableDataset(Dataset):
    """Dataset that adapts speckle/pattern resolution per experiment."""

    def __init__(
        self,
        speckles_path: str,
        patterns_path: str,
        indices: Sequence[int],
        *,
        speckle_input: int,
        model_output: int,
        physics_valid: bool,
        color_channel: int = 2,
        pol_channel: int = 2,
    ) -> None:
        self.speckles = np.load(speckles_path, mmap_mode="r")
        self.patterns = np.load(patterns_path, mmap_mode="r")
        self.indices = list(indices)
        self.speckle_input = speckle_input
        self.model_output = model_output
        self.color_channel = color_channel
        self.pol_channel = pol_channel

        self.base_speckle_res = self.speckles.shape[-1]
        self.base_pattern_res = self.patterns.shape[-1]

        ensure_physics_valid(self.base_speckle_res, self.speckle_input, physics_valid)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        original_idx = self.indices[idx]
        speckle_idx = original_idx * 3 + self.color_channel

        speckle = self.speckles[speckle_idx, self.pol_channel].astype(np.float32).copy()
        pattern = self.patterns[speckle_idx].astype(np.float32).copy()

        if self.speckle_input != self.base_speckle_res:
            speckle = cv2.resize(
                speckle,
                (self.speckle_input, self.speckle_input),
                interpolation=cv2.INTER_AREA if self.speckle_input < self.base_speckle_res else cv2.INTER_CUBIC,
            )

        target = pattern
        if self.model_output != self.base_pattern_res:
            target = cv2.resize(
                target,
                (self.model_output, self.model_output),
                interpolation=cv2.INTER_LINEAR,
            )

        speckle = speckle / 255.0
        target = quantize_and_normalize(target)

        speckle_tensor = torch.from_numpy(speckle).unsqueeze(0).float()
        target_tensor = torch.from_numpy(target).unsqueeze(0).float()
        return speckle_tensor, target_tensor


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual, inplace=True)


class DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, residual: bool = False) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.residual:
            out = self.res_block(out)
        return out


class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1, inplace=True)
        psi = self.psi(psi)
        return x * psi


class AdaptiveUNet(nn.Module):
    """UNet variant with dynamic output heads and gradient checkpointing."""

    def __init__(self, input_resolution: int, output_resolution: int, base: int = 48) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.use_checkpoint = output_resolution >= 256

        self.enc1 = DoubleConv(1, base, residual=False)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2, residual=True)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4, residual=True)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base * 4, base * 8, residual=True)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = DoubleConv(base * 8, base * 16, residual=True)
        self.pool5 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(DoubleConv(base * 16, base * 32, residual=True), ResidualBlock(base * 32))

        self.up5 = nn.ConvTranspose2d(base * 32, base * 16, 2, stride=2)
        self.att5 = AttentionGate(base * 16, base * 16, base * 8)
        self.dec5 = DoubleConv(base * 32, base * 16, residual=True)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.att4 = AttentionGate(base * 8, base * 8, base * 4)
        self.dec4 = DoubleConv(base * 16, base * 8, residual=True)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.att3 = AttentionGate(base * 4, base * 4, base * 2)
        self.dec3 = DoubleConv(base * 8, base * 4, residual=True)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.dec2 = DoubleConv(base * 4, base * 2, residual=True)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att1 = AttentionGate(base, base, base // 2)
        self.dec1 = DoubleConv(base * 2, base, residual=False)

        self.final_256 = nn.Sequential(nn.Conv2d(base, base // 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base // 2, 1, 1))
        self.final_64 = nn.Sequential(nn.Conv2d(base, base // 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base // 2, 1, 1))

    def _maybe_checkpoint(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(module, x)
        return module(x)

    def _maybe_checkpoint_with_cat(self, module: nn.Module, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([x, skip], dim=1)
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(module, merged)
        return module(merged)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self._maybe_checkpoint(self.enc1, x)
        e2 = self._maybe_checkpoint(self.enc2, self.pool1(e1))
        e3 = self._maybe_checkpoint(self.enc3, self.pool2(e2))
        e4 = self._maybe_checkpoint(self.enc4, self.pool3(e3))
        e5 = self._maybe_checkpoint(self.enc5, self.pool4(e4))

        b = self._maybe_checkpoint(self.bottleneck, self.pool5(e5))

        d5 = self.up5(b)
        e5_att = self.att5(d5, e5)
        d5 = self._maybe_checkpoint_with_cat(self.dec5, d5, e5_att)

        d4 = self.up4(d5)
        e4_att = self.att4(d4, e4)
        d4 = self._maybe_checkpoint_with_cat(self.dec4, d4, e4_att)

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self._maybe_checkpoint_with_cat(self.dec3, d3, e3_att)

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self._maybe_checkpoint_with_cat(self.dec2, d2, e2_att)

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self._maybe_checkpoint_with_cat(self.dec1, d1, e1_att)

        if self.output_resolution == 64:
            out = self.final_64(d1)
            out = F.interpolate(out, size=(64, 64), mode="bilinear", align_corners=False)
        elif self.output_resolution == 256:
            out = self.final_256(d1)
            out = F.interpolate(out, size=(256, 256), mode="bilinear", align_corners=False)
        else:
            out = self.final_256(d1)
            out = F.interpolate(out, size=(self.output_resolution, self.output_resolution), mode="bilinear", align_corners=False)

        return torch.sigmoid(out)


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._create_window(window_size), persistent=False)

    def _gaussian(self, window_size: int, sigma: float = 1.5) -> torch.Tensor:
        gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / (2 * sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size: int) -> torch.Tensor:
        _1d = self._gaussian(window_size).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
        return _2d

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
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

        c1 = 0.01**2
        c2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1 - ssim_map.mean()


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class AdvancedLoss(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.w_pcc = 0.25
        self.w_ssim = 0.25
        self.w_percep = 0.35
        self.w_edge = 0.15

        try:
            vgg = models.vgg19(weights="IMAGENET1K_V1").features
            self.vgg_slice1 = vgg[:4].to(device).eval()
            self.vgg_slice2 = vgg[:9].to(device).eval()
            self.vgg_slice3 = vgg[:18].to(device).eval()
            self.vgg_slice4 = vgg[:27].to(device).eval()
            for module in [self.vgg_slice1, self.vgg_slice2, self.vgg_slice3, self.vgg_slice4]:
                for param in module.parameters():
                    param.requires_grad = False
            self.use_percep = True
        except Exception:
            logging.getLogger(self.__class__.__name__).warning("VGG19 weights not available; disabling perceptual loss")
            self.use_percep = False
            self.vgg_slice1 = self.vgg_slice2 = self.vgg_slice3 = self.vgg_slice4 = None

        self.ssim = SSIMLoss()
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
            persistent=False,
        )

    def pcc_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
        pred_std = torch.sqrt((pred_centered**2).mean(dim=1, keepdim=True) + eps)
        target_std = torch.sqrt((target_centered**2).mean(dim=1, keepdim=True) + eps)
        correlation = (pred_centered * target_centered).mean(dim=1, keepdim=True) / (pred_std * target_std + eps)
        return 1 - correlation.mean()

    def edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sobel_x = self.sobel_x.to(pred.device)
        sobel_y = self.sobel_y.to(pred.device)
        pred_edges = torch.sqrt(F.conv2d(pred, sobel_x, padding=1) ** 2 + F.conv2d(pred, sobel_y, padding=1) ** 2)
        target_edges = torch.sqrt(F.conv2d(target, sobel_x, padding=1) ** 2 + F.conv2d(target, sobel_y, padding=1) ** 2)
        return F.mse_loss(pred_edges, target_edges)

    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.use_percep:
            return pred.new_tensor(0.0)

        with torch.amp.autocast("cuda", enabled=False):
            pred_fp32 = pred.float()
            target_fp32 = target.float()
            pred_3ch = pred_fp32.repeat(1, 3, 1, 1)
            target_3ch = target_fp32.repeat(1, 3, 1, 1)
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            pred_norm = (pred_3ch - mean) / std
            target_norm = (target_3ch - mean) / std
            slices = [self.vgg_slice1, self.vgg_slice2, self.vgg_slice3, self.vgg_slice4]
            weights = [0.1, 0.2, 0.3, 0.4]
            loss = 0.0
            for layer, weight in zip(slices, weights):
                if layer is None:
                    continue
                pred_feat = layer(pred_norm)
                with torch.no_grad():
                    target_feat = layer(target_norm)
                loss = loss + F.l1_loss(pred_feat, target_feat) * weight
            return loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        loss_pcc = self.pcc_loss(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_percep = self.perceptual_loss(pred, target)
        loss_edge = self.edge_loss(pred, target)

        total = (
            self.w_pcc * loss_pcc
            + self.w_ssim * loss_ssim
            + (self.w_percep * loss_percep if self.use_percep else 0.0)
            + self.w_edge * loss_edge
        )

        return {
            "total_loss": total,
            "pcc": loss_pcc.detach(),
            "ssim": loss_ssim.detach(),
            "perceptual": loss_percep.detach(),
            "edge": loss_edge.detach(),
        }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: AdvancedLoss,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    logger: logging.Logger,
    description: str,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0
    progress = tqdm(dataloader, desc=f"{description} | Epoch {epoch + 1}/{total_epochs}", leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_dict = loss_fn(outputs, targets)
        loss = loss_dict["total_loss"]
        loss.backward()
        optimizer.step()
        batch_loss = float(loss.item())
        running_loss += batch_loss
        num_batches += 1
        progress.set_postfix({"loss": f"{batch_loss:.4f}"})
    avg_loss = running_loss / max(1, num_batches)
    logger.debug("%s epoch %d avg_loss=%.4f", description, epoch + 1, avg_loss)
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    eval_resolution: int,
) -> Dict[str, float]:
    model.eval()
    total_pcc = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    total_samples = 0
    ssim_fn = SSIMLoss().to(device)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            if outputs.size(-1) != eval_resolution:
                outputs = F.interpolate(outputs, size=(eval_resolution, eval_resolution), mode="bilinear", align_corners=False)
            if targets.size(-1) != eval_resolution:
                targets = F.interpolate(targets, size=(eval_resolution, eval_resolution), mode="bilinear", align_corners=False)

            for i in range(outputs.size(0)):
                pred = outputs[i].squeeze()
                gt = targets[i].squeeze()
                pred_flat = pred.detach().cpu().numpy().flatten()
                gt_flat = gt.detach().cpu().numpy().flatten()
                pcc_val, _ = pearsonr(pred_flat, gt_flat)
                total_pcc += 0.0 if np.isnan(pcc_val) else pcc_val
                total_ssim += float((1 - ssim_fn(pred.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0))).item())
                total_mse += float(F.mse_loss(pred, gt).item())
                total_samples += 1

    return {
        "pcc": total_pcc / max(1, total_samples),
        "ssim": total_ssim / max(1, total_samples),
        "mse": total_mse / max(1, total_samples),
    }


def format_minutes(seconds: float) -> float:
    return seconds / 60.0


def get_memory_usage_mb(device: torch.device) -> float:
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)
    return float("nan")


def reset_memory_stats(device: torch.device) -> None:
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


@dataclass
class ExperimentConfig:
    name: str
    speckle_input: int
    model_output: int
    eval_resolution: int
    description: str
    expected_time: str
    physics_valid: bool
    for_final_table: bool


class ExperimentRunner:
    def __init__(self, configs: Sequence[ExperimentConfig], output_dir: str, device: torch.device) -> None:
        self.configs = list(configs)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        self.memory_log_path = os.path.join(self.output_dir, "memory_usage_report.txt")
        with open(self.memory_log_path, "w") as mem_file:
            mem_file.write("experiment_name,peak_memory_mb,device\n")

    def _build_dataloaders(self, config: ExperimentConfig) -> Dict[str, DataLoader]:
        batch_size = BATCH_SIZE_RULES.get(config.speckle_input)
        if batch_size is None:
            raise ValueError(f"No batch size rule defined for input resolution {config.speckle_input}")

        train_indices = list(range(0, TRAIN_SPLIT))
        val_indices = list(range(TRAIN_SPLIT, TRAIN_SPLIT + VAL_SPLIT))
        test_indices = list(range(TRAIN_SPLIT + VAL_SPLIT, TOTAL_SAMPLES))

        dataset_kwargs = dict(
            speckles_path=SPECKLE_FILE,
            patterns_path=PATTERN_FILE,
            speckle_input=config.speckle_input,
            model_output=config.model_output,
            physics_valid=config.physics_valid,
        )

        train_dataset = ConfigurableDataset(indices=train_indices, **dataset_kwargs)
        val_dataset = ConfigurableDataset(indices=val_indices, **dataset_kwargs)
        test_dataset = ConfigurableDataset(indices=test_indices, **dataset_kwargs)

        pin_memory = self.device.type == "cuda"
        num_workers = 4 if pin_memory else 0

        loaders = {
            "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
            "val": DataLoader(val_dataset, batch_size=max(1, batch_size // 2), shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
            "test": DataLoader(test_dataset, batch_size=max(1, batch_size // 2), shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        }
        return loaders

    def _log_memory_usage(self, config: ExperimentConfig, device: torch.device) -> None:
        peak_memory = get_memory_usage_mb(device)
        with open(self.memory_log_path, "a") as mem_file:
            mem_file.write(f"{config.name},{peak_memory:.2f},{device.type}\n")

    def _cleanup(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_all(self) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []
        for cfg_dict in RESOLUTION_EXPERIMENTS:
            config = ExperimentConfig(**cfg_dict)
            self.logger.info("Starting experiment: %s (%s)", config.name, config.description)
            try:
                exp_result = self._run_single(config, self.device)
            except RuntimeError as err:
                if "out of memory" in str(err).lower() and self.device.type == "cuda":
                    self.logger.error("OOM on GPU for %s; retrying on CPU", config.name)
                    self._cleanup()
                    fallback_device = torch.device("cpu")
                    exp_result = self._run_single(config, fallback_device)
                else:
                    raise
            results.append(exp_result)
            self.logger.info(
                "%s results: PCC=%.4f | SSIM=%.4f | Time=%.2f min",
                config.name,
                exp_result["pcc"],
                exp_result["ssim"],
                exp_result["training_time_minutes"],
            )
            self._cleanup()
        return results

    def _run_single(self, config: ExperimentConfig, device: torch.device) -> Dict[str, float]:
        loaders = self._build_dataloaders(config)
        model = AdaptiveUNet(config.speckle_input, config.model_output).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5, betas=(0.9, 0.999))
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=TRAIN_EPOCHS)
        loss_fn = AdvancedLoss(device)

        reset_memory_stats(device)

        epoch_durations: List[float] = []
        start_time = time.perf_counter()
        best_val_pcc = -float("inf")
        patience_counter = 0

        for epoch in range(TRAIN_EPOCHS):
            epoch_start = time.perf_counter()
            scheduler.step(epoch)
            train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                loss_fn,
                device,
                epoch,
                TRAIN_EPOCHS,
                self.logger,
                config.name,
            )
            val_metrics = evaluate(model, loaders["val"], device, config.eval_resolution)
            epoch_durations.append(time.perf_counter() - epoch_start)

            if val_metrics["pcc"] > best_val_pcc:
                best_val_pcc = val_metrics["pcc"]
                patience_counter = 0
            else:
                patience_counter += 1

            if EARLY_STOP_PATIENCE is not None and patience_counter >= EARLY_STOP_PATIENCE:
                self.logger.info("Early stopping %s at epoch %d", config.name, epoch + 1)
                break

        total_time = time.perf_counter() - start_time
        test_metrics = evaluate(model, loaders["test"], device, config.eval_resolution)

        self._log_memory_usage(config, device)

        result = {
            "experiment_name": config.name,
            "speckle_input": config.speckle_input,
            "model_output": config.model_output,
            "eval_resolution": config.eval_resolution,
            "pcc": test_metrics["pcc"],
            "ssim": test_metrics["ssim"],
            "mse": test_metrics["mse"],
            "training_time_minutes": format_minutes(total_time),
            "physics_valid": config.physics_valid,
            "for_final_table": config.for_final_table,
            "efficiency_ratio": test_metrics["pcc"] / max(1e-6, format_minutes(total_time)),
            "avg_epoch_time_minutes": format_minutes(float(np.mean(epoch_durations))) if epoch_durations else 0.0,
            "device": device.type,
        }
        return result


def export_results(results: Sequence[Dict[str, float]], output_dir: str) -> None:
    main_path = os.path.join(output_dir, "main_comparison_table.csv")
    full_path = os.path.join(output_dir, "comprehensive_results.csv")

    fields = [
        "experiment_name",
        "speckle_input",
        "model_output",
        "eval_resolution",
        "pcc",
        "ssim",
        "mse",
        "training_time_minutes",
        "physics_valid",
        "efficiency_ratio",
        "device",
    ]

    with open(full_path, "w", newline="") as full_file:
        writer = csv.DictWriter(full_file, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fields})

    main_rows = [row for row in results if row["for_final_table"]]
    with open(main_path, "w", newline="") as main_file:
        writer = csv.DictWriter(main_file, fieldnames=fields)
        writer.writeheader()
        for row in main_rows:
            writer.writerow({k: row.get(k) for k in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resolution ablation experiments")
    parser.add_argument("--output-dir", default="resolution_outputs", help="Directory for experiment outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.output_dir)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.getLogger(__name__).info("Running resolution ablation on device: %s", device)

    runner = ExperimentRunner(
        configs=[ExperimentConfig(**cfg) for cfg in RESOLUTION_EXPERIMENTS],
        output_dir=args.output_dir,
        device=device,
    )

    results = runner.run_all()
    export_results(results, args.output_dir)
    logging.getLogger(__name__).info("All experiments completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user")
    except Exception:
        logging.getLogger(__name__).exception("Unhandled error during resolution ablation")
