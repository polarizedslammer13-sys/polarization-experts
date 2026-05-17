#!/usr/bin/env python3
"""
Early-Stack 偏振融合策略
=========================
将 4 路偏振散斑（0°/45°/90°/135°，pol_channel=1/2/3/4）
在输入端直接拼接为 [B, 4, 256, 256]，送入 UNetPro256（首层改为 in_channels=4）。

固定追踪样本（永久固定）：
  指定锚点：1898, 1924, 1930
  随机补充：1800, 1821, 1844, 1923, 1938, 1945, 1968  (seed=2025)
  全部10个：[1800, 1821, 1844, 1898, 1923, 1924, 1930, 1938, 1945, 1968]

用法：
    python3 run_early_stack.py
    python3 run_early_stack.py --color 0 --epochs 60
"""

import os, math, random, time, json, argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm
from scipy.stats import pearsonr
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/tmp/NotoSansCJKsc.otf')
matplotlib.rcParams['font.family'] = 'Noto Sans CJK SC'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR   = "/root/autodl-tmp/facedataset_0825"
TRAIN_IDX  = list(range(0, 1600))
VAL_IDX    = list(range(1600, 1800))
TEST_IDX   = list(range(1800, 2000))

# 永久固定的10个追踪样本
VIZ_IDX    = [1800, 1821, 1844, 1898, 1923, 1924, 1930, 1938, 1945, 1968]
VIZ_ANCHORS = {1898, 1924, 1930}

# Early-Stack 使用 pol=1/2/3/4（0°/45°/90°/135°），排除OG
POL_CHANNELS = [1, 2, 3, 4]
POL_LABELS   = ['0°', '45°', '90°', '135°']
CH_NAMES     = {0: 'B', 1: 'G', 2: 'R'}


# ── GPU ───────────────────────────────────────────────────────────────
def resolve_device():
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception as e:
        cuda_ok = False; err = str(e)
    else:
        err = None
    if cuda_ok:
        try:
            dev = torch.device('cuda:0')
            torch.empty(1, device=dev)
            return dev, "torch.cuda.is_available() = True"
        except Exception as e:
            first_err = str(e)
    else:
        first_err = err or "is_available() = False"
    try:
        dev = torch.device('cuda:0')
        torch.empty(1, device=dev)
        return dev, "Explicit CUDA probe succeeded"
    except Exception as e:
        return torch.device('cpu'), f"CUDA failed ({first_err}; {e})"


# ── Dataset：4路拼接输入 ──────────────────────────────────────────────
class EarlyStackDataset(Dataset):
    """
    输入：将 pol=1/2/3/4 的散斑在 channel 维度拼接 → [4, 256, 256]
    标签：对应 color_channel 的 pattern，上采样至 256×256
    """
    def __init__(self, spf, paf, indices, color_channel=2):
        self.sp  = np.load(spf, mmap_mode='r')
        self.pat = np.load(paf, mmap_mode='r')
        self.idx = indices
        self.cc  = color_channel

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        oi = self.idx[i]
        si = oi * 3 + self.cc

        # 4路偏振拼接
        channels = []
        for pc in POL_CHANNELS:
            sp = self.sp[si, pc].astype(np.float32).copy() / 255.0
            channels.append(sp)
        x = torch.from_numpy(np.stack(channels, axis=0)).float()  # [4, 256, 256]

        pat = self.pat[si].astype(np.float32).copy() / 255.0
        pat = cv2.resize(pat, (256, 256), interpolation=cv2.INTER_LINEAR)
        gt  = torch.from_numpy(pat).float()
        return x, gt


# ── Model：UNetPro256，首层改为 in_channels=4 ─────────────────────────
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(
            nn.Linear(c, c//r, bias=False), nn.ReLU(True),
            nn.Linear(c//r, c, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        return x * self.ex(self.sq(x).view(b, c)).view(b, c, 1, 1)

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
        self.se = SEBlock(c)
    def forward(self, x):
        return F.relu(self.se(self.net(x)) + x, True)

class DoubleConv(nn.Module):
    def __init__(self, i, o, res=False):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
        self.r = ResidualBlock(o) if res else None
    def forward(self, x):
        return self.r(self.c(x)) if self.r else self.c(x)

class AG(nn.Module):
    def __init__(self, g, l, i):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(g, i, 1, bias=False), nn.BatchNorm2d(i))
        self.Wx = nn.Sequential(nn.Conv2d(l, i, 1, bias=False), nn.BatchNorm2d(i))
        self.ps = nn.Sequential(nn.Conv2d(i, 1, 1), nn.Sigmoid())
    def forward(self, g, x):
        return x * self.ps(F.relu(self.Wg(g) + self.Wx(x), True))

class UNetEarlyStack(nn.Module):
    """
    标准 UNetPro256，唯一改动：in_channels=4（接受4路拼接输入）
    """
    def __init__(self, in_channels=4, base=48):
        super().__init__()
        B = base
        self.e1=DoubleConv(in_channels,B);    self.p1=nn.MaxPool2d(2)
        self.e2=DoubleConv(B,   B*2,  True);  self.p2=nn.MaxPool2d(2)
        self.e3=DoubleConv(B*2, B*4,  True);  self.p3=nn.MaxPool2d(2)
        self.e4=DoubleConv(B*4, B*8,  True);  self.p4=nn.MaxPool2d(2)
        self.e5=DoubleConv(B*8, B*16, True);  self.p5=nn.MaxPool2d(2)
        self.bot = nn.Sequential(DoubleConv(B*16, B*32, True), ResidualBlock(B*32))
        self.u5=nn.ConvTranspose2d(B*32,B*16,2,2); self.a5=AG(B*16,B*16,B*8);  self.d5=DoubleConv(B*32,B*16,True)
        self.u4=nn.ConvTranspose2d(B*16,B*8, 2,2); self.a4=AG(B*8, B*8, B*4);  self.d4=DoubleConv(B*16,B*8, True)
        self.u3=nn.ConvTranspose2d(B*8, B*4, 2,2); self.a3=AG(B*4, B*4, B*2);  self.d3=DoubleConv(B*8, B*4, True)
        self.u2=nn.ConvTranspose2d(B*4, B*2, 2,2); self.a2=AG(B*2, B*2, B);    self.d2=DoubleConv(B*4, B*2, True)
        self.u1=nn.ConvTranspose2d(B*2, B,   2,2); self.a1=AG(B,   B,   B//2); self.d1=DoubleConv(B*2, B)
        self.final = nn.Sequential(
            nn.Conv2d(B, B//2, 3, padding=1), nn.ReLU(True), nn.Conv2d(B//2, 1, 1))

    def forward(self, x):
        e1=self.e1(x); e2=self.e2(self.p1(e1)); e3=self.e3(self.p2(e2))
        e4=self.e4(self.p3(e3)); e5=self.e5(self.p4(e4)); b=self.bot(self.p5(e5))
        def up(u, a, d, e, prev): t=u(prev); return d(torch.cat([t, a(t,e)], 1))
        d5=up(self.u5,self.a5,self.d5,e5,b); d4=up(self.u4,self.a4,self.d4,e4,d5)
        d3=up(self.u3,self.a3,self.d3,e3,d4); d2=up(self.u2,self.a2,self.d2,e2,d3)
        d1=up(self.u1,self.a1,self.d1,e1,d2)
        return torch.sigmoid(self.final(d1))


# ── Loss（与 run_exp2.py 完全一致）──────────────────────────────────
class SSIMLoss(nn.Module):
    def __init__(self, ws=11):
        super().__init__()
        self.ws = ws
        g = torch.Tensor([math.exp(-(x-ws//2)**2/(2*1.5**2)) for x in range(ws)])
        g /= g.sum()
        self.window = g.unsqueeze(1).mm(g.unsqueeze(0)).float().unsqueeze(0).unsqueeze(0)
    def forward(self, a, b):
        if self.window.device != a.device:
            self.window = self.window.to(a.device)
        w=self.window; p=self.ws//2
        m1=F.conv2d(a,w,padding=p); m2=F.conv2d(b,w,padding=p)
        s1=F.conv2d(a*a,w,padding=p)-m1**2; s2=F.conv2d(b*b,w,padding=p)-m2**2
        s12=F.conv2d(a*b,w,padding=p)-m1*m2; C1,C2=0.01**2,0.03**2
        return 1-((2*m1*m2+C1)*(2*s12+C2)/((m1**2+m2**2+C1)*(s1+s2+C2))).mean()

def pcc_loss_fn(pred, target, eps=1e-8):
    pf=pred.view(pred.size(0),-1); tf=target.view(target.size(0),-1)
    pc=pf-pf.mean(1,keepdim=True); tc=tf-tf.mean(1,keepdim=True)
    return 1-((pc*tc).mean(1,keepdim=True)/(
        torch.sqrt((pc**2).mean(1,keepdim=True)+eps)*
        torch.sqrt((tc**2).mean(1,keepdim=True)+eps)+eps)).mean()

class AdvancedLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.ssim = SSIMLoss()
        self.use_vgg = False
        try:
            vgg = models.vgg19(weights='IMAGENET1K_V1').features
            self.vgg_s = nn.ModuleList([
                vgg[:4].to(device).eval(),  vgg[:9].to(device).eval(),
                vgg[:18].to(device).eval(), vgg[:27].to(device).eval()])
            for p in self.vgg_s.parameters(): p.requires_grad = False
            self.use_vgg = True
        except Exception as e:
            print(f"[WARN] VGG: {e}")
        self.register_buffer('sx',
            torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('sy',
            torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3))

    def _percep(self, p, t):
        with torch.amp.autocast('cuda', enabled=False):
            p3=p.float().repeat(1,3,1,1); t3=t.float().repeat(1,3,1,1)
            mn=torch.tensor([0.485,0.456,0.406],device=self.device).view(1,3,1,1)
            st=torch.tensor([0.229,0.224,0.225],device=self.device).view(1,3,1,1)
            p3=(p3-mn)/st; t3=(t3-mn)/st; loss=0.0
            for layer,w in zip(self.vgg_s,[0.1,0.2,0.3,0.4]):
                pf=layer(p3)
                with torch.no_grad(): tf=layer(t3)
                loss += F.l1_loss(pf, tf)*w
            return loss

    def _edge(self, p, t):
        sx=self.sx.to(p.device).type(p.dtype); sy=self.sy.to(p.device).type(p.dtype)
        return (F.l1_loss(F.conv2d(p,sx,padding=1), F.conv2d(t,sx,padding=1)) +
                F.l1_loss(F.conv2d(p,sy,padding=1), F.conv2d(t,sy,padding=1)))

    def forward(self, pred, gt):
        if gt.dim()==3: gt=gt.unsqueeze(1)
        p64=F.adaptive_avg_pool2d(pred,(64,64)); g64=F.adaptive_avg_pool2d(gt,(64,64))
        lp  = pcc_loss_fn(p64, g64)
        ls  = self.ssim(p64, g64)
        le  = self._edge(p64, g64)
        lpe = self._percep(p64, g64) if self.use_vgg else torch.zeros(1,device=self.device)
        total = 0.25*lp + 0.25*ls + 0.35*lpe + 0.15*le
        return total, {'pcc':float(lp.item()), 'ssim':float(ls.item()),
                       'percep':float(lpe.item()), 'edge':float(le.item())}

class WarmupCosine:
    def __init__(self, opt, warmup=10, total=60, eta_min=1e-6):
        self.opt=opt; self.w=warmup; self.T=total; self.eta=eta_min
        self.base=opt.param_groups[0]['lr']
    def step(self, e):
        lr = (self.base*(e+1)/self.w if e<self.w else
              self.eta+(self.base-self.eta)*0.5*(1+math.cos(math.pi*(e-self.w)/(self.T-self.w))))
        for pg in self.opt.param_groups: pg['lr']=lr
        return lr


# ── Evaluate ─────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval(); tp=0.; ts=0.; tm=0.; n=0
    ssim_fn = SSIMLoss()
    with torch.no_grad():
        for x, gt in loader:
            x=x.to(device); gt=gt.to(device)
            if gt.dim()==3: gt=gt.unsqueeze(1)
            pred = model(x)
            p64  = F.adaptive_avg_pool2d(pred,(64,64))
            g64  = F.adaptive_avg_pool2d(gt,  (64,64))
            tm  += F.mse_loss(p64,g64).item()*x.size(0)
            ts  += (1-ssim_fn(p64,g64)).item()*x.size(0)
            for i in range(p64.shape[0]):
                try:
                    r,_ = pearsonr(p64[i,0].cpu().numpy().flatten(),
                                   g64[i,0].cpu().numpy().flatten())
                    if not np.isnan(r): tp+=r; n+=1
                except: pass
    N = len(loader.dataset)
    return {'pcc':tp/max(n,1), 'ssim':ts/N, 'mse':tm/N}


# ── Visualisation ─────────────────────────────────────────────────────
def collect_viz(model, spf, paf, color_ch, device):
    ds = EarlyStackDataset(spf, paf, VIZ_IDX, color_ch)
    ld = DataLoader(ds, batch_size=len(VIZ_IDX), shuffle=False, num_workers=2)
    model.eval()
    with torch.no_grad():
        x, gt = next(iter(ld))
        x=x.to(device); gt=gt.to(device)
        if gt.dim()==3: gt=gt.unsqueeze(1)
        pred = model(x)
        p64  = F.adaptive_avg_pool2d(pred,(64,64))
        g64  = F.adaptive_avg_pool2d(gt,  (64,64))
    # 用第3路（90°，pol=3）的散斑作为代表性输入展示
    speckles = [x[i,2].cpu().numpy() for i in range(len(VIZ_IDX))]
    preds    = [p64[i,0].cpu().numpy() for i in range(len(VIZ_IDX))]
    gts      = [g64[i,0].cpu().numpy() for i in range(len(VIZ_IDX))]
    return speckles, preds, gts


def make_figure(speckles, preds, gts, val_curve,
                test_metrics, color_ch, best_epoch, save_path):
    n = len(VIZ_IDX)
    fig = plt.figure(figsize=(16, 14), facecolor='white')

    outer = gridspec.GridSpec(2, 1, figure=fig,
                               height_ratios=[2.2, n*1.6], hspace=0.12)

    # ── 学习曲线 ──────────────────────────────────────────────────────
    ax = fig.add_subplot(outer[0])
    ax.set_facecolor('#F8F9FA')
    epochs_x = list(range(2, 2*len(val_curve)+1, 2))
    ax.plot(epochs_x, [v['pcc']  for v in val_curve],
            color='#1565C0', lw=2.0, label='验证集 PCC')
    ax.plot(epochs_x, [v['ssim'] for v in val_curve],
            color='#2E7D32', lw=1.6, ls='--', label='验证集 SSIM')
    ax.axvline(best_epoch, color='#E53935', lw=1.0, ls=':', alpha=0.7)
    ax.text(best_epoch+0.5, min(v['pcc'] for v in val_curve),
            f'最优 Ep={best_epoch}', fontsize=8, color='#E53935', va='bottom')
    ax.text(0.98, 0.08,
            f"测试集  PCC={test_metrics['pcc']:.4f}   SSIM={test_metrics['ssim']:.4f}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color='#333',
            bbox=dict(facecolor='white', edgecolor='#CCC',
                      boxstyle='round,pad=0.4'))
    ax.set_xlabel('训练轮次', fontsize=10)
    ax.set_ylabel('指标值', fontsize=10)
    ax.set_title(
        f'输入端叠加融合（Early-Stack）·  4路偏振拼接  ·  '
        f'{CH_NAMES[color_ch]} 通道',
        fontsize=11, fontweight='bold', pad=8)
    ax.legend(fontsize=9, framealpha=0.9, loc='lower right')
    ax.grid(color='#E0E0E0', lw=0.5)
    for sp in ax.spines.values(): sp.set_edgecolor('#CCCCCC')

    # ── 重建网格 ──────────────────────────────────────────────────────
    inner = gridspec.GridSpecFromSubplotSpec(
        n, 3, subplot_spec=outer[1], hspace=0.06, wspace=0.04)

    col_titles = ['散斑输入（90°代表）', '重建结果', '真实标签']
    col_cmaps  = ['hot', 'gray', 'gray']

    for col_i, (title, cmap) in enumerate(zip(col_titles, col_cmaps)):
        for row_i in range(n):
            ax2 = fig.add_subplot(inner[row_i, col_i])
            ax2.set_facecolor('white')
            img = [speckles, preds, gts][col_i][row_i]
            ax2.imshow(img, cmap=cmap, vmin=img.min(), vmax=img.max(),
                        interpolation='nearest')
            if row_i == 0:
                ax2.set_title(title, fontsize=9, pad=4,
                               fontweight='bold', color='#333')
            if col_i == 0:
                ax2.set_ylabel(f'#{VIZ_IDX[row_i]}', fontsize=7.5,
                                color='#666', rotation=0,
                                labelpad=32, va='center')
            if col_i == 1:
                r_val, _ = pearsonr(img.flatten(), gts[row_i].flatten())
                ax2.text(0.97, 0.04, f'r={r_val:.3f}',
                          transform=ax2.transAxes, color='white',
                          fontsize=6.5, ha='right', va='bottom',
                          bbox=dict(facecolor='#333', edgecolor='none',
                                    boxstyle='round,pad=0.25', alpha=0.75))
            for sp in ax2.spines.values():
                sp.set_edgecolor('#DDDDDD'); sp.set_linewidth(0.5)
            ax2.set_xticks([]); ax2.set_yticks([])

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  图像已保存 → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description='Early-Stack 偏振融合')
    parser.add_argument('--color',  type=int, default=2, choices=[0,1,2],
                         help='颜色通道 B=0 G=1 R=2')
    parser.add_argument('--epochs', type=int, default=60)
    args = parser.parse_args()

    set_seed(42)
    device, reason = resolve_device()

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    ch_name = CH_NAMES[args.color]
    exp_dir = os.path.join(BASE_DIR, f"early_stack_{ch_name}_{ts}")
    os.makedirs(exp_dir, exist_ok=True)

    print("="*60)
    print("Early-Stack 输入端叠加融合")
    print("="*60)
    print(f"设备           : {device}  ({reason})")
    print(f"偏振通道       : 0°/45°/90°/135°（pol 1/2/3/4 拼接）")
    print(f"颜色通道       : {ch_name}")
    print(f"训练轮次       : {args.epochs}")
    print(f"输出目录       : {exp_dir}")
    print(f"追踪样本       : {VIZ_IDX}")
    print(f"锚点样本       : {sorted(VIZ_ANCHORS)}\n")

    spf = os.path.join(BASE_DIR, "original", "speckles6000_og.npy")
    paf = os.path.join(BASE_DIR, "original", "pattern.npy")

    kw  = dict(num_workers=6, pin_memory=True)
    trl = DataLoader(EarlyStackDataset(spf,paf,TRAIN_IDX,args.color),
                      batch_size=4, shuffle=True, **kw)
    vll = DataLoader(EarlyStackDataset(spf,paf,VAL_IDX,  args.color),
                      batch_size=8, shuffle=False, **kw)
    tel = DataLoader(EarlyStackDataset(spf,paf,TEST_IDX, args.color),
                      batch_size=8, shuffle=False, **kw)

    model   = UNetEarlyStack(in_channels=4, base=48).to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=2e-4,
                                  weight_decay=1e-5, betas=(0.9,0.999))
    sched   = WarmupCosine(opt, warmup=10, total=args.epochs)
    loss_fn = AdvancedLoss(device)

    print(f"参数量：{sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")

    best_pcc     = -1.0
    best_weights = None
    best_epoch   = 0
    patience     = 0
    val_curve    = []
    t0           = time.time()

    for ep in range(args.epochs):
        lr = sched.step(ep)
        model.train()
        for x, gt in tqdm(trl, desc=f"Ep{ep+1:2d}", leave=False):
            x=x.to(device); gt=gt.to(device)
            if gt.dim()==3: gt=gt.unsqueeze(1)
            opt.zero_grad()
            loss, _ = loss_fn(model(x), gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        ep_num = ep + 1
        if ep_num % 2 == 0 or ep_num == args.epochs:
            vm = evaluate(model, vll, device)
            val_curve.append(vm)

            improved = vm['pcc'] > best_pcc
            if improved:
                best_pcc     = vm['pcc']
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch   = ep_num
                patience     = 0
            else:
                patience += 1

            flag = ' ←' if improved else ''
            print(f"  Ep {ep_num:3d}/{args.epochs} | "
                  f"Val PCC={vm['pcc']:.4f} (best={best_pcc:.4f}) | "
                  f"SSIM={vm['ssim']:.4f} | LR={lr:.6f}{flag}")

            if patience >= 4:
                print(f"  早停于 Ep {ep_num}")
                break

    if best_weights:
        model.load_state_dict(best_weights)

    # 保存模型
    model_path = os.path.join(exp_dir, "best_model.pth")
    torch.save({
        'model_state_dict': best_weights,
        'color_channel':  args.color,
        'best_epoch':     best_epoch,
        'best_val_pcc':   best_pcc,
        'viz_idx':        VIZ_IDX,
        'fusion_strategy':'early_stack',
        'pol_channels':   POL_CHANNELS,
    }, model_path)
    print(f"\n  模型已保存 → {model_path}")

    test_m  = evaluate(model, tel, device)
    elapsed = time.time() - t0

    print(f"\n  最优验证 PCC : {best_pcc:.4f}  (Ep {best_epoch})")
    print(f"  测试集 PCC   : {test_m['pcc']:.4f}")
    print(f"  测试集 SSIM  : {test_m['ssim']:.4f}")
    print(f"  测试集 MSE   : {test_m['mse']:.6f}")
    print(f"  总耗时       : {elapsed/3600:.2f}h")

    speckles, preds, gts = collect_viz(
        model, spf, paf, args.color, device)
    fig_path = os.path.join(exp_dir, f"reconstruction_early_stack_{ch_name}_{ts}.png")
    make_figure(speckles, preds, gts, val_curve,
                test_m, args.color, best_epoch, fig_path)

    report = {
        'timestamp':      ts,
        'fusion_strategy':'early_stack',
        'pol_channels':   POL_CHANNELS,
        'color_channel':  args.color,
        'best_epoch':     best_epoch,
        'best_val_pcc':   float(best_pcc),
        'test_pcc':       float(test_m['pcc']),
        'test_ssim':      float(test_m['ssim']),
        'test_mse':       float(test_m['mse']),
        'time_hours':     float(elapsed/3600),
        'viz_idx':        VIZ_IDX,
        'anchors':        sorted(VIZ_ANCHORS),
    }
    with open(os.path.join(exp_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  所有输出 → {exp_dir}/")

if __name__ == "__main__":
    try:
        main(); print("\n✓ 完成！")
    except KeyboardInterrupt: print("\n⚠ 已中断")
    except Exception as e:
        import traceback; print(f"\n✗ {e}"); traceback.print_exc()
