"""
Model Evaluation Script for Image Colorization
Randomly picks an image, runs inference with your trained model (including vibrant saturation boost)
and the reference model, computes PSNR and SSIM metrics, and displays a side-by-side comparison.

Usage:
    python evaluate.py                          # Evaluates a random validation image
    python evaluate.py --split train            # Evaluates a random training image
    python evaluate.py --image path/to.jpg      # Evaluates a specific custom image
    python evaluate.py --saturation 1.35        # Boosts color vibrancy by 35%
"""

import os
import sys
import random
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Ensure local imports work cleanly
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "colorization"))

from src.model import ResNetColorizationNet
from src.dataset import lab_to_rgb
import colorizers


def get_random_image(split="val"):
    split_dir = os.path.join(ROOT_DIR, "data", "raw", split)
    if not os.path.exists(split_dir) or len(os.listdir(split_dir)) == 0:
        split_dir = os.path.join(ROOT_DIR, "data", "raw", "train")
    
    files = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not files:
        raise FileNotFoundError(f"No images found in {split_dir}!")
    return random.choice(files)


def evaluate(image_path=None, split="val", saturation=1.25, show_window=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print(f"[*] Image Colorization Model Evaluation on {device.type.upper()}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    # 1. Select image
    if image_path is None or not os.path.exists(image_path):
        image_path = get_random_image(split)
    
    image_name = os.path.basename(image_path)
    print(f"\n[1/4] Selected Image: {image_name}")
    print(f"      Path: {image_path}")

    # 2. Load Your Trained Model (v2 perceptual if available, else v1)
    ckpt_v2 = os.path.join(ROOT_DIR, "checkpoints", "best_colorizer_v2.pth")
    ckpt_v1 = os.path.join(ROOT_DIR, "checkpoints", "best_colorizer.pth")
    checkpoint_path = ckpt_v2 if os.path.exists(ckpt_v2) else ckpt_v1

    if not os.path.exists(checkpoint_path):
        print(f"\n[!] Error: Checkpoint not found at {checkpoint_path}")
        print("   Please train your model first using notebooks/04_training.ipynb!")
        return

    model_version = "V2 (Perceptual)" if "v2" in checkpoint_path else "V1 (L1 Loss)"
    print(f"\n[2/4] Loading models ({model_version})...")
    my_model = ResNetColorizationNet(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    my_model.load_state_dict(ckpt["model_state_dict"])
    my_model.eval()
    print(f"      [+] Loaded Your Model (Trained for {ckpt.get('epoch', '?')} epochs, Val Loss: {ckpt.get('val_loss', 0):.4f})")

    # Load Zhang ECCV16 reference model
    ref_eccv16 = colorizers.eccv16(pretrained=True).eval().to(device)
    print("      [+] Loaded Zhang ECCV16 Reference Model")

    # 3. Preprocess Image
    print("\n[3/4] Running inference & color reconstruction...")
    pil_img = Image.open(image_path).convert("RGB").resize((256, 256), Image.BICUBIC)
    rgb_gt = np.array(pil_img) / 255.0  # Ground truth [0, 1]

    # Convert to CIE Lab
    lab = color.rgb2lab(rgb_gt).astype(np.float32)
    L_orig = lab[:, :, 0]  # Lightness [0, 100]

    # Normalize L for our model: [-1, 1]
    L_norm = (L_orig / 50.0) - 1.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 256, 256)

    # 1. Run Your Model (Standard)
    t0 = time.time()
    with torch.no_grad():
        pred_ab = my_model(L_tensor).squeeze(0)  # (2, 256, 256)
    my_time = (time.time() - t0) * 1000
    my_rgb = lab_to_rgb(L_tensor.squeeze(0), pred_ab)

    # 2. Run Your Model (Vibrant Saturation Boosted)
    pred_ab_boosted = torch.clamp(pred_ab * saturation, -1.0, 1.0)
    my_rgb_boosted = lab_to_rgb(L_tensor.squeeze(0), pred_ab_boosted)

    # 3. Run Reference Model (Zhang ECCV16)
    t0 = time.time()
    tens_l_orig = torch.Tensor(L_orig)[None, None, :, :].to(device)
    with torch.no_grad():
        eccv_out = ref_eccv16(tens_l_orig).cpu()
    ref_time = (time.time() - t0) * 1000

    ref_lab = np.zeros((256, 256, 3))
    ref_lab[:, :, 0] = L_orig
    ref_lab[:, :, 1:] = eccv_out[0].numpy().transpose(1, 2, 0)
    ref_rgb = np.clip(color.lab2rgb(ref_lab), 0.0, 1.0)

    # 4. Compute Quantitative Evaluation Metrics
    # PSNR (higher is better)
    my_psnr = psnr(rgb_gt.astype(np.float32), my_rgb.astype(np.float32), data_range=1.0)
    my_boosted_psnr = psnr(rgb_gt.astype(np.float32), my_rgb_boosted.astype(np.float32), data_range=1.0)
    ref_psnr = psnr(rgb_gt.astype(np.float32), ref_rgb.astype(np.float32), data_range=1.0)

    # SSIM (0 to 1, higher is better)
    my_ssim = ssim(rgb_gt.astype(np.float32), my_rgb.astype(np.float32), channel_axis=2, data_range=1.0)
    my_boosted_ssim = ssim(rgb_gt.astype(np.float32), my_rgb_boosted.astype(np.float32), channel_axis=2, data_range=1.0)
    ref_ssim = ssim(rgb_gt.astype(np.float32), ref_rgb.astype(np.float32), channel_axis=2, data_range=1.0)

    # MAE (lower is better)
    my_mae = np.mean(np.abs(rgb_gt - my_rgb))
    my_boosted_mae = np.mean(np.abs(rgb_gt - my_rgb_boosted))
    ref_mae = np.mean(np.abs(rgb_gt - ref_rgb))

    # Print Formatted Evaluation Table
    print("\n" + "=" * 70)
    print("EVALUATION METRICS REPORT")
    print("=" * 70)
    print(f"{'Metric':<18} | {'Your Model':<15} | {f'Boosted ({saturation}x)':<15} | {'Zhang ECCV16':<15}")
    print("-" * 70)
    print(f"{'PSNR (dB)':<18} | {my_psnr:<15.2f} | {my_boosted_psnr:<15.2f} | {ref_psnr:<15.2f}")
    print(f"{'SSIM (0 - 1)':<18} | {my_ssim:<15.4f} | {my_boosted_ssim:<15.4f} | {ref_ssim:<15.4f}")
    print(f"{'MAE (Lower=Better)':<18} | {my_mae:<15.4f} | {my_boosted_mae:<15.4f} | {ref_mae:<15.4f}")
    print(f"{'Inference Speed':<18} | {f'{my_time:.1f} ms':<15} | {f'{my_time:.1f} ms':<15} | {f'{ref_time:.1f} ms':<15}")
    print("=" * 70)

    # 5. Plot 5-Panel Side-by-Side Comparison
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    # Panel 1: Grayscale Input
    axes[0].imshow(L_orig, cmap="gray")
    axes[0].set_title("Input (Grayscale L)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Your Model (Standard)
    axes[1].imshow(my_rgb)
    axes[1].set_title(
        f"Your Model (Standard)\nPSNR: {my_psnr:.2f} dB | SSIM: {my_ssim:.3f}",
        fontsize=12,
        fontweight="bold",
        color="darkblue"
    )
    axes[1].axis("off")

    # Panel 3: Your Model (Vibrant Boosted)
    axes[2].imshow(my_rgb_boosted)
    axes[2].set_title(
        f"Your Model (Vibrant {saturation}x)\nPSNR: {my_boosted_psnr:.2f} dB | SSIM: {my_boosted_ssim:.3f}",
        fontsize=12,
        fontweight="bold",
        color="#8A2BE2"
    )
    axes[2].axis("off")

    # Panel 4: Reference Model
    axes[3].imshow(ref_rgb)
    axes[3].set_title(
        f"Zhang ECCV16\nPSNR: {ref_psnr:.2f} dB | SSIM: {ref_ssim:.3f}",
        fontsize=12,
        fontweight="bold",
        color="#8B4513"
    )
    axes[3].axis("off")

    # Panel 5: Ground Truth
    axes[4].imshow(rgb_gt)
    axes[4].set_title("Ground Truth (Target)", fontsize=12, fontweight="bold", color="darkgreen")
    axes[4].axis("off")

    plt.suptitle(f"Image Colorization Evaluation: {image_name}", fontsize=15, y=0.98)
    plt.tight_layout()

    # Save to outputs directory
    out_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    clean_name = os.path.splitext(image_name)[0]
    out_path = os.path.join(out_dir, f"eval_{clean_name}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n[4/4] Saved 5-panel comparison plot to:")
    print(f"      -> {out_path}\n")

    if show_window:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Colorization Models")
    parser.add_argument("--image", type=str, default=None, help="Path to a specific image to evaluate")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"], help="Data split to pick randomly from")
    parser.add_argument("--saturation", type=float, default=1.25, help="Color saturation boost factor (e.g., 1.25 for +25%)")
    parser.add_argument("--no-show", action="store_true", help="Do not open GUI window (save to file only)")
    args = parser.parse_args()

    evaluate(image_path=args.image, split=args.split, saturation=args.saturation, show_window=not args.no_show)
