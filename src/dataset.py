import os
import torch
import numpy as np
from PIL import Image
from skimage import color
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class ColorizationDataset(Dataset):
    """
    Custom PyTorch Dataset for Image Colorization in CIE Lab space.
    - Input: L channel (Lightness/grayscale, normalized to [-1, 1], shape: 1x256x256)
    - Target: ab channels (Color hues, normalized to [-1, 1], shape: 2x256x256)
    """
    def __init__(self, paths, is_train=True):
        self.paths = paths
        self.is_train = is_train
        
        # Data augmentation for training: resize to 256x256 and random flip
        if self.is_train:
            self.transforms = T.Compose([
                T.Resize((256, 256), Image.BICUBIC),
                T.RandomHorizontalFlip(p=0.5)
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((256, 256), Image.BICUBIC)
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 1. Load image and ensure RGB mode
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        
        # 2. Convert to CIE Lab space
        img_np = np.array(img)
        lab = color.rgb2lab(img_np).astype(np.float32)
        
        # 3. Normalize channels to [-1, 1]
        # L in [0, 100] -> (L / 50.0) - 1.0 in [-1, 1]
        L = (lab[:, :, 0] / 50.0) - 1.0
        # ab in [-128, 127] -> ab / 128.0 in [-1, 1]
        ab = lab[:, :, 1:] / 128.0
        
        # 4. Convert to PyTorch tensors (C, H, W)
        L_tensor = torch.from_numpy(L).unsqueeze(0)             # (1, 256, 256)
        ab_tensor = torch.from_numpy(ab).permute(2, 0, 1)        # (2, 256, 256)
        
        return {"L": L_tensor, "ab": ab_tensor}

def lab_to_rgb(L_tensor, ab_tensor):
    """
    Converts normalized L ([-1, 1]) and ab ([-1, 1]) back to RGB numpy array [0, 1].
    Supports both batch tensors (B, C, H, W) and single tensors (C, H, W).
    """
    if L_tensor.ndim == 4:
        # Batch: (B, 1, H, W) -> list of RGBs
        rgbs = []
        for l, ab in zip(L_tensor, ab_tensor):
            rgbs.append(lab_to_rgb(l, ab))
        return np.array(rgbs)

    # Single image: L is (1, H, W), ab is (2, H, W)
    if isinstance(L_tensor, torch.Tensor):
        L_np = L_tensor.squeeze().detach().cpu().numpy()
        ab_np = ab_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    else:
        L_np = L_tensor.squeeze()
        ab_np = ab_tensor.transpose(1, 2, 0) if ab_tensor.shape[0] == 2 else ab_tensor

    L_denorm = (L_np + 1.0) * 50.0
    ab_denorm = ab_np * 128.0
    
    H, W = L_np.shape
    lab = np.zeros((H, W, 3), dtype=np.float64)
    lab[:, :, 0] = L_denorm
    lab[:, :, 1:] = ab_denorm
    
    rgb = color.lab2rgb(lab)
    return np.clip(rgb, 0.0, 1.0)

def make_dataloaders(train_paths, val_paths, batch_size=16, num_workers=0):
    train_dataset = ColorizationDataset(train_paths, is_train=True)
    val_dataset = ColorizationDataset(val_paths, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader
