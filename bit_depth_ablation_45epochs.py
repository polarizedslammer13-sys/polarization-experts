#!/usr/bin/env python3
"""Bit-depth ablation study for MMF face reconstruction baseline (45-epoch variant)."""

import argparse
import csv
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


BASE_DIR = "/root/autodl-tmp/facedataset_0825"
TOTAL_SAMPLES = 2000
TRAIN_SPLIT = int(0.8 * TOTAL_SAMPLES)
VAL_SPLIT = int(0.1 * TOTAL_SAMPLES)
TEST_SPLIT = TOTAL_SAMPLES - TRAIN_SPLIT - VAL_SPLIT


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def quantize_pattern(pattern: np.ndarray, max_value: int) -> np.ndarray:
    """Quantize a uint8 pattern image to the target bit-depth range."""
    scaled = np.round(pattern.astype(np.float32) / 255.0 * max_value)
    scaled = np.clip(scaled, 0, max_value)
    return scaled.astype(np.float32)


class BitDepthDataset(Dataset):
    """Speckle-pattern pairs with configurable quantization."""

    def __init__(
        self,
        speckles_path: str,
        patterns_path: str,
        indices: Sequence[int],
        *,
        pol_channel: int,
        color_channel: int,
        max_value: int,
        validate: bool = True,
    ) -> None:
        self.speckles_mmap = np.load(speckles_path, mmap_mode="r")
        self.patterns_mmap = np.load(patterns_path, mmap_mode="r")
        self.indices = list(indices)
        self.pol_channel = pol_channel
        self.color_channel = color_channel
        self.max_value = max_value
        if validate:
            self._validate_quantization()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        original_idx = self.indices[idx]
        speckle_idx = original_idx * 3 + self.color_channel

        speckle = self.speckles_mmap[speckle_idx, self.pol_channel].astype(np.float32).copy()
        pattern = self.patterns_mmap[speckle_idx].astype(np.float32).copy()

        speckle = speckle / 255.0
        quantized_pattern = quantize_pattern(pattern, self.max_value)
        quantized_pattern /= float(self.max_value)

        # Upsample to 256x256
        quantized_pattern = cv2.resize(quantized_pattern, (256, 256), interpolation=cv2.INTER_LINEAR)

        x = torch.from_numpy(speckle).unsqueeze(0).float()
        gt = torch.from_numpy(quantized_pattern).float()
        return x, gt

    def _validate_quantization(self) -> None:
        if self.max_value <= 0:
            raise ValueError("max_value must be positive")
        if not self.indices:
            return
        sample_indices = random.sample(self.indices, min(5, len(self.indices)))
        for idx in sample_indices:
            speckle_idx = idx * 3 + self.color_channel
            pattern = self.patterns_mmap[speckle_idx]
            quantized = quantize_pattern(pattern, self.max_value)
            if quantized.min() < 0 or quantized.max() > self.max_value:
                raise ValueError("Quantized pattern out of expected range")


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


class UNetPro256(nn.Module):
    def __init__(self, in_channels: int = 1, base: int = 48) -> None:
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

        self.bottleneck = nn.Sequential(DoubleConv(base * 16, base * 32, residual=True), ResidualBlock(base * 32))

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

        self.final = nn.Sequential(nn.Conv2d(base, base // 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base // 2, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._create_window(window_size), persistent=False)

    def _gaussian(self, window_size: int, sigma: float = 1.5) -> torch.Tensor:
        gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma**2)) for x in range(window_size)])
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
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class AdvancedLoss(nn.Module):
    def __init__(self, device: torch.device, w_pcc: float, w_ssim: float, w_percep: float, w_edge: float) -> None:
        super().__init__()
        self.device = device
        self.w_pcc = w_pcc
        self.w_ssim = w_ssim
        self.w_percep = w_percep
        self.w_edge = w_edge

        try:
            vgg = models.vgg19(weights="IMAGENET1K_V1").features
            self.vgg_slice1 = vgg[:4].to(device).eval()
            self.vgg_slice2 = vgg[:9].to(device).eval()
            self.vgg_slice3 = vgg[:18].to(device).eval()
            self.vgg_slice4 = vgg[:27].to(device).eval()

            for module in [self.vgg_slice1, self.vgg_slice2, self.vgg_slice3, self.vgg_slice4]:
                for param in module.parameters():
                    param.requires_grad = False

            self.use_perceptual = True
        except Exception:
            logging.getLogger(self.__class__.__name__).warning(
                "VGG19 weights not available; perceptual loss disabled"
            )
            self.use_perceptual = False
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

        pred_edges_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2)

        target_edges_x = F.conv2d(target, sobel_x, padding=1)
        target_edges_y = F.conv2d(target, sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2)

        return F.mse_loss(pred_edges, target_edges)

    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.use_perceptual:
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

            weights = [0.1, 0.2, 0.3, 0.4]
            loss = 0.0
            for layer, weight in zip([self.vgg_slice1, self.vgg_slice2, self.vgg_slice3, self.vgg_slice4], weights):
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
            + (self.w_percep * loss_percep if self.use_perceptual else 0.0)
            + self.w_edge * loss_edge
        )

        return {
            "total_loss": total,
            "pcc": loss_pcc.detach().cpu(),
            "ssim": loss_ssim.detach().cpu(),
            "perceptual": loss_percep.detach().cpu(),
            "edge": loss_edge.detach().cpu(),
        }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: AdvancedLoss,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    description: str,
) -> Dict[str, float]:
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
        loss_dict["total_loss"].backward()
        optimizer.step()

        batch_loss = float(loss_dict["total_loss"].item())
        running_loss += batch_loss
        num_batches += 1
        progress.set_postfix({"loss": f"{batch_loss:.4f}"})

    avg_loss = running_loss / max(1, num_batches)
    return {"loss": avg_loss}


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, eval_at_64: bool = True) -> Dict[str, float]:
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

            if eval_at_64:
                outputs = F.interpolate(outputs, size=(64, 64), mode="bilinear", align_corners=False)
                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)
                targets = F.interpolate(targets, size=(64, 64), mode="bilinear", align_corners=False)

            for i in range(outputs.size(0)):
                pred = outputs[i].squeeze()
                gt = targets[i].squeeze() if targets.dim() == 4 else targets[i]

                pred_flat = pred.detach().cpu().numpy().flatten()
                gt_flat = gt.detach().cpu().numpy().flatten()
                pcc_val, _ = pearsonr(pred_flat, gt_flat)
                total_pcc += 0.0 if np.isnan(pcc_val) else pcc_val

                pred_4d = pred.unsqueeze(0).unsqueeze(0)
                gt_4d = gt.unsqueeze(0).unsqueeze(0)
                total_ssim += float((1 - ssim_fn(pred_4d, gt_4d)).item())

                total_mse += float(F.mse_loss(pred, gt).item())
                total_samples += 1

    return {
        "pcc": total_pcc / max(1, total_samples),
        "ssim": total_ssim / max(1, total_samples),
        "mse": total_mse / max(1, total_samples),
    }


def format_seconds(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def get_memory_snapshot(device: torch.device) -> float:
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        return peak_bytes / (1024**2)
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**2)
    return float("nan")


def reset_memory_stats(device: torch.device) -> None:
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


@dataclass
class ExperimentConfig:
    bit_depth: int
    max_value: int
    seed: int = 42
    epochs: int = 45
    batch_size: int = 4
    val_batch_size: int = 8
    num_workers: int = 6


class AblationRunner:
    def __init__(self, configs: Sequence[ExperimentConfig], device: torch.device, output_dir: str) -> None:
        self.configs = list(configs)
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []
        for config in self.configs:
            self.logger.info("Starting experiment for %d-bit (max=%d)", config.bit_depth, config.max_value)
            experiment_result = self._run_single_experiment(config)
            results.append(experiment_result)
            self.logger.info(
                "Completed %d-bit: PCC=%.4f | SSIM=%.4f | Time=%s",
                config.bit_depth,
                experiment_result["test_pcc"],
                experiment_result["test_ssim"],
                experiment_result["train_time_str"],
            )
        return results

    def _run_single_experiment(self, config: ExperimentConfig) -> Dict[str, float]:
        set_seed(config.seed)

        train_indices = list(range(0, TRAIN_SPLIT))
        val_indices = list(range(TRAIN_SPLIT, TRAIN_SPLIT + VAL_SPLIT))
        test_indices = list(range(TRAIN_SPLIT + VAL_SPLIT, TOTAL_SAMPLES))

        train_dataset = BitDepthDataset(
            os.path.join(BASE_DIR, "original", "speckles6000_og.npy"),
            os.path.join(BASE_DIR, "original", "pattern.npy"),
            train_indices,
            pol_channel=2,
            color_channel=2,
            max_value=config.max_value,
        )
        val_dataset = BitDepthDataset(
            os.path.join(BASE_DIR, "original", "speckles6000_og.npy"),
            os.path.join(BASE_DIR, "original", "pattern.npy"),
            val_indices,
            pol_channel=2,
            color_channel=2,
            max_value=config.max_value,
        )
        test_dataset = BitDepthDataset(
            os.path.join(BASE_DIR, "original", "speckles6000_og.npy"),
            os.path.join(BASE_DIR, "original", "pattern.npy"),
            test_indices,
            pol_channel=2,
            color_channel=2,
            max_value=config.max_value,
        )

        effective_workers = config.num_workers
        pin_memory = self.device.type == "cuda"
        if effective_workers > 0 and not pin_memory:
            self.logger.warning(
                "%d-bit running without CUDA; forcing num_workers=0 due to limited multiprocessing support",
                config.bit_depth,
            )
            effective_workers = 0

        loader_kwargs = {
            "pin_memory": pin_memory,
            "num_workers": effective_workers,
        }

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            **loader_kwargs,
        )

        model = UNetPro256(base=48).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5, betas=(0.9, 0.999))
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=7, total_epochs=config.epochs)
        loss_fn = AdvancedLoss(
            self.device,
            w_pcc=0.25,
            w_ssim=0.25,
            w_percep=0.35,
            w_edge=0.15,
        )

        reset_memory_stats(self.device)
        epoch_durations: List[float] = []
        start_time = time.perf_counter()
        best_val_pcc = -float("inf")
        patience_counter = 0
        patience = 10

        for epoch in range(config.epochs):
            epoch_start = time.perf_counter()
            current_lr = scheduler.step(epoch)
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                self.device,
                epoch,
                config.epochs,
                description=f"{config.bit_depth}-bit",
            )
            val_metrics = evaluate(model, val_loader, self.device, eval_at_64=True)
            epoch_duration = time.perf_counter() - epoch_start
            epoch_durations.append(epoch_duration)

            if val_metrics["pcc"] > best_val_pcc:
                best_val_pcc = val_metrics["pcc"]
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == config.epochs - 1:
                self.logger.info(
                    "%d-bit epoch %d/%d | loss=%.4f | val_pcc=%.4f (best=%.4f) | val_ssim=%.4f | lr=%.6f | epoch_time=%s",
                    config.bit_depth,
                    epoch + 1,
                    config.epochs,
                    train_metrics["loss"],
                    val_metrics["pcc"],
                    best_val_pcc,
                    val_metrics["ssim"],
                    current_lr,
                    format_seconds(epoch_duration),
                )

            if patience_counter >= patience:
                self.logger.info("%d-bit early stopping triggered at epoch %d", config.bit_depth, epoch + 1)
                break

        total_time = time.perf_counter() - start_time
        test_metrics = evaluate(model, test_loader, self.device, eval_at_64=True)
        peak_memory = get_memory_snapshot(self.device)

        result = {
            "bit_depth": config.bit_depth,
            "max_value": config.max_value,
            "best_val_pcc": best_val_pcc,
            "test_pcc": test_metrics["pcc"],
            "test_ssim": test_metrics["ssim"],
            "test_mse": test_metrics["mse"],
            "train_time_sec": total_time,
            "train_time_str": format_seconds(total_time),
            "avg_epoch_time_sec": float(np.mean(epoch_durations)) if epoch_durations else 0.0,
            "avg_epoch_time_str": format_seconds(float(np.mean(epoch_durations))) if epoch_durations else "00:00:00.00",
            "num_epochs_completed": len(epoch_durations),
            "peak_memory_mb": peak_memory,
        }

        return result


def export_results(results: Sequence[Dict[str, float]], output_dir: str) -> None:
    csv_path = os.path.join(output_dir, "bit_depth_results.csv")
    json_path = os.path.join(output_dir, "bit_depth_results.json")

    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        with open(json_path, "w") as json_file:
            json.dump(results, json_file, indent=2)


def plot_results(results: Sequence[Dict[str, float]], output_dir: str) -> None:
    bit_depths = [r["bit_depth"] for r in results]
    pcc_scores = [r["test_pcc"] for r in results]
    ssim_scores = [r["test_ssim"] for r in results]
    train_times = [r["train_time_sec"] for r in results]
    memory_usage = [r["peak_memory_mb"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    axes[0].plot(bit_depths, pcc_scores, marker="o")
    axes[0].set_title("Bit Depth vs PCC")
    axes[0].set_xlabel("Bit Depth")
    axes[0].set_ylabel("Test PCC")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(bit_depths, ssim_scores, marker="s", color="tab:orange")
    axes[1].set_title("Bit Depth vs SSIM")
    axes[1].set_xlabel("Bit Depth")
    axes[1].set_ylabel("Test SSIM")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].bar(bit_depths, [t / 60 for t in train_times], color="tab:green")
    axes[2].set_title("Training Time")
    axes[2].set_xlabel("Bit Depth")
    axes[2].set_ylabel("Minutes")

    axes[3].bar(bit_depths, memory_usage, color="tab:red")
    axes[3].set_title("Peak Memory Usage")
    axes[3].set_xlabel("Bit Depth")
    axes[3].set_ylabel("MB")

    plt.tight_layout()
    figure_path = os.path.join(output_dir, "bit_depth_comparison.png")
    plt.savefig(figure_path, dpi=200)
    plt.close(fig)


def summarize_results(results: Sequence[Dict[str, float]]) -> str:
    if not results:
        return "No results recorded."
    headers = ["Bit", "PCC", "SSIM", "MSE", "Time", "Epochs", "Memory(MB)"]
    rows = []
    for r in results:
        rows.append(
            [
                f"{r['bit_depth']} bit",
                f"{r['test_pcc']:.4f}",
                f"{r['test_ssim']:.4f}",
                f"{r['test_mse']:.6f}",
                r["train_time_str"],
                r["num_epochs_completed"],
                f"{r['peak_memory_mb']:.1f}" if not math.isnan(r["peak_memory_mb"]) else "N/A",
            ]
        )

    line_lengths = [max(len(header), *(len(row[idx]) for row in rows)) for idx, header in enumerate(headers)]
    formatted_lines = []
    header_line = " | ".join(header.ljust(lengths) for header, lengths in zip(headers, line_lengths))
    formatted_lines.append(header_line)
    formatted_lines.append("-" * len(header_line))
    for row in rows:
        formatted_lines.append(" | ".join(cell.ljust(lengths) for cell, lengths in zip(row, line_lengths)))
    return "\n".join(formatted_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bit-depth ablation study runner")
    parser.add_argument("--output-dir", default="ablation_outputs", help="Directory to store logs, tables, and plots")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.getLogger(__name__).info("Running on %s", device)

    bit_depth_settings = {
        3: 7,
        4: 15,
        5: 31,
        6: 63,
        7: 127,
        8: 255,
    }

    configs = [
        ExperimentConfig(bit_depth=bit_depth, max_value=max_val, seed=42 + idx)
        for idx, (bit_depth, max_val) in enumerate(sorted(bit_depth_settings.items()))
    ]

    runner = AblationRunner(configs=configs, device=device, output_dir=args.output_dir)
    results = runner.run()

    export_results(results, args.output_dir)
    plot_results(results, args.output_dir)

    summary = summarize_results(results)
    logging.getLogger(__name__).info("\n%s", summary)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user")
    except Exception:
        logging.getLogger(__name__).exception("Unhandled error during ablation")
