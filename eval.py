import os
import json
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import PairedImageFolder
from models.generator import ResNetGenerator
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np


def load_cfg(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def tensor_to_img8(t):
    t = (t.clamp(0, 1) * 255.0).round().byte().cpu().numpy()
    # CHW -> HWC
    return np.transpose(t, (1, 2, 0))


def main():
    parser = argparse.ArgumentParser(description='Evaluate generator on val set (PSNR/SSIM).')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset (val)
    val_ds = PairedImageFolder(
        root=cfg['val']['data_root'],
        sharp_dir=cfg['val']['sharp_dir'],
        blur_dir=cfg['val']['blur_dir'],
        crop_size=cfg['train']['crop_size'],
        train=False,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Model
    G = ResNetGenerator(
        in_ch=cfg['model']['generator']['in_ch'],
        base_ch=cfg['model']['generator']['base_ch'],
        blocks=cfg['model']['generator']['blocks'],
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    G.load_state_dict(ckpt['G'])
    G.eval()

    psnr_vals, ssim_vals = [], []
    with torch.no_grad():
        for batch in val_loader:
            blur = batch['blur'].to(device)
            sharp = batch['sharp'].to(device)
            pred = G(blur)
            # To uint8 for metrics
            p = tensor_to_img8(pred[0])
            gt = tensor_to_img8(sharp[0])
            # skimage expects grayscale or multichannel flag for SSIM
            psnr_vals.append(psnr(gt, p, data_range=255))
            ssim_vals.append(ssim(gt, p, channel_axis=2, data_range=255))

    metrics = {
        'n': len(psnr_vals),
        'psnr_mean': float(np.mean(psnr_vals)) if psnr_vals else 0.0,
        'psnr_std': float(np.std(psnr_vals)) if psnr_vals else 0.0,
        'ssim_mean': float(np.mean(ssim_vals)) if ssim_vals else 0.0,
        'ssim_std': float(np.std(ssim_vals)) if ssim_vals else 0.0,
    }

    out_path = args.out
    if out_path is None:
        # default next to checkpoint
        base = os.path.dirname(args.ckpt)
        out_path = os.path.join(base, 'metrics_val.json')
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
