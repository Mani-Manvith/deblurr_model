import os
from typing import Tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class PairedImageFolder(Dataset):
    def __init__(self, root: str, sharp_dir: str, blur_dir: str, crop_size: int = 256, train: bool = True):
        self.sharp_root = os.path.join(root, sharp_dir)
        self.blur_root = os.path.join(root, blur_dir)
        self.paths = self._paired_paths(self.sharp_root, self.blur_root)
        self.train = train
        self.crop_size = crop_size
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((crop_size, crop_size))
        if train:
            self.aug = T.Compose([
                T.RandomHorizontalFlip(),
                # Avoid hue on PIL path due to uint8 overflow in some torchvision versions
                T.ColorJitter(0.1, 0.1, 0.1, 0.0),
            ])
        else:
            self.aug = T.Compose([])

    def _paired_paths(self, sharp_root: str, blur_root: str):
        files = []
        for dirpath, _, filenames in os.walk(sharp_root):
            for f in filenames:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    sharp_p = os.path.join(dirpath, f)
                    rel = os.path.relpath(sharp_p, sharp_root)
                    blur_p = os.path.join(blur_root, rel)
                    if os.path.exists(blur_p):
                        files.append((sharp_p, blur_p))
        return sorted(files)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        sharp_p, blur_p = self.paths[idx]
        with Image.open(sharp_p) as s:
            s = s.convert("RGB")
        with Image.open(blur_p) as b:
            b = b.convert("RGB")
        # simple center resize (replace with random crop for better training)
        s = self.resize(self.aug(s))
        b = self.resize(self.aug(b))
        return {
            "sharp": self.to_tensor(s),
            "blur": self.to_tensor(b),
            "name": os.path.basename(sharp_p),
        }


def make_dataloaders(cfg):
    train_ds = PairedImageFolder(
        root=cfg["train"]["data_root"],
        sharp_dir=cfg["train"]["sharp_dir"],
        blur_dir=cfg["train"]["blur_dir"],
        crop_size=cfg["train"]["crop_size"],
        train=True,
    )
    val_ds = PairedImageFolder(
        root=cfg["val"]["data_root"],
        sharp_dir=cfg["val"]["sharp_dir"],
        blur_dir=cfg["val"]["blur_dir"],
        crop_size=cfg["train"]["crop_size"],
        train=False,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["val"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)
    return train_loader, val_loader
