import os
import random
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import json
from datasets import make_dataloaders
import argparse
from models.generator import ResNetGenerator
from models.discriminator import PatchDiscriminator
from losses import GANLoss, VGGPerceptualLoss


def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def save_ckpt(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_cfg(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train(cfg_path: str = 'training/configs/deblurganv2.yaml'):
    cfg = load_cfg(cfg_path)
    set_seed(cfg.get('seed', 42))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader = make_dataloaders(cfg)

    G = ResNetGenerator(
        in_ch=cfg['model']['generator']['in_ch'],
        base_ch=cfg['model']['generator']['base_ch'],
        blocks=cfg['model']['generator']['blocks'],
    ).to(device)
    D = PatchDiscriminator(
        in_ch=cfg['model']['discriminator']['in_ch'],
        base_ch=cfg['model']['discriminator']['base_ch'],
    ).to(device)

    l1 = nn.L1Loss()
    vgg = VGGPerceptualLoss().to(device)
    gan_loss = GANLoss(use_lsgan=True)

    g_opt = optim.Adam(G.parameters(), lr=cfg['optimizer']['g_lr'], betas=tuple(cfg['optimizer']['betas']))
    d_opt = optim.Adam(D.parameters(), lr=cfg['optimizer']['d_lr'], betas=tuple(cfg['optimizer']['betas']))

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get('amp', True) and device == 'cuda')

    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    max_epochs = cfg['max_epochs']
    log_interval = cfg['log_interval']

    global_step = 0
    for epoch in range(1, max_epochs + 1):
        G.train(); D.train()
        for i, batch in enumerate(train_loader, 1):
            blur = batch['blur'].to(device)
            sharp = batch['sharp'].to(device)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred = G(blur)
                # D loss
                d_real = D(sharp)
                d_fake = D(pred.detach())
                d_loss = (gan_loss(d_real, True) + gan_loss(d_fake, False)) * 0.5

            d_opt.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            scaler.step(d_opt)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                # G loss
                d_fake_for_g = D(pred)
                loss_gan = gan_loss(d_fake_for_g, True) * cfg['lambda_gan']
                loss_l1 = l1(pred, sharp) * cfg['lambda_l1']
                loss_perc = vgg(pred, sharp) * cfg['lambda_perc']
                g_loss = loss_gan + loss_l1 + loss_perc

            g_opt.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.step(g_opt)
            scaler.update()

            if global_step % log_interval == 0:
                print(f"epoch {epoch} step {global_step} | D {d_loss.item():.3f} | G {g_loss.item():.3f} (GAN {loss_gan.item():.3f} L1 {loss_l1.item():.3f} P {loss_perc.item():.3f})")

            global_step += 1

        # Validation, checkpoint, and early stop
        G.eval()
        psnr_vals, ssim_vals = [], []
        with torch.no_grad():
            for batch in val_loader:
                blur = batch['blur'].to(device)
                sharp = batch['sharp'].to(device)
                pred = G(blur)
                # Save a small grid preview from first batch
                if len(psnr_vals) == 0:
                    grid = torch.cat([blur[:2], pred[:2], sharp[:2]], dim=0)
                    save_image(grid, os.path.join(save_dir, f'epoch_{epoch:03d}.png'), nrow=2)

                # Convert to uint8 HWC
                def t2img8(t):
                    t = (t.clamp(0,1) * 255.0).round().byte().cpu().numpy()
                    return t.transpose(0,2,3,1)  # NCHW->NHWC

                p = t2img8(pred)
                g = t2img8(sharp)
                for i in range(p.shape[0]):
                    psnr_vals.append(psnr(g[i], p[i], data_range=255))
                    ssim_vals.append(ssim(g[i], p[i], channel_axis=2, data_range=255))

        import numpy as _np
        psnr_mean = float(_np.mean(_np.array(psnr_vals))) if psnr_vals else 0.0
        ssim_mean = float(_np.mean(_np.array(ssim_vals))) if ssim_vals else 0.0
        psnr_std = float(_np.std(_np.array(psnr_vals))) if psnr_vals else 0.0
        ssim_std = float(_np.std(_np.array(ssim_vals))) if ssim_vals else 0.0

        metrics = {
            'epoch': epoch,
            'psnr_mean': psnr_mean,
            'psnr_std': psnr_std,
            'ssim_mean': ssim_mean,
            'ssim_std': ssim_std,
        }

        # Save metrics JSON (append latest)
        metrics_path = os.path.join(save_dir, 'metrics_history.json')
        hist = []
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    hist = json.load(f)
            except Exception:
                hist = []
        hist.append(metrics)
        with open(metrics_path, 'w') as f:
            json.dump(hist, f, indent=2)

        print(f"val epoch {epoch}: PSNR {metrics['psnr_mean']:.3f}±{metrics.get('psnr_std',0):.3f} | SSIM {metrics['ssim_mean']:.3f}±{metrics.get('ssim_std',0):.3f}")

        # Save checkpoint
        save_ckpt({'G': G.state_dict(), 'D': D.state_dict(), 'epoch': epoch}, os.path.join(save_dir, f'ckpt_{epoch:03d}.pt'))

        # Early stop when SSIM >= 0.88
        if metrics['ssim_mean'] >= 0.88:
            print(f"Early stopping: SSIM reached {metrics['ssim_mean']:.3f} at epoch {epoch}")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeblurGANv2 baseline (lightweight).')
    parser.add_argument('--config', type=str, default='training/configs/deblurganv2.yaml', help='Path to YAML config')
    args = parser.parse_args()
    train(args.config)
