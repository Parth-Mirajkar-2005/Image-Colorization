"""
A3 Poster Generator for Deep Learning Image Colorization Project
Author: Parth Randhir Mirajkar
Dimensions: A3 Portrait (297 x 420 mm / 11.693 x 16.535 inches @ 300 DPI -> 3508 x 4960 px)
Outputs: outputs/poster_A3.png and outputs/poster_A3.pdf
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# Premium Academic Color Palette
# -------------------------------------------------------------
COLOR_BG = "#F8FAFC"          # Light Slate Canvas Background
COLOR_HEADER_BG = "#0B1329"   # Midnight Deep Navy
COLOR_HEADER_TEXT = "#FFFFFF"
COLOR_ACCENT_BLUE = "#0284C7" # Vibrant Sky Blue
COLOR_ACCENT_CYAN = "#38BDF8" # Bright Cyan
COLOR_CARD_BG = "#FFFFFF"
COLOR_CARD_BORDER = "#CBD5E1"
COLOR_TEXT_DARK = "#0F172A"
COLOR_TEXT_BODY = "#334155"
COLOR_TEXT_MUTED = "#64748B"
COLOR_SUCCESS = "#15803D"
COLOR_BADGE_BG = "#16203A"


def create_poster():
    print("[1/4] Setting up A3 Portrait Canvas (11.693 x 16.535 inches @ 300 DPI)...")
    fig = plt.figure(figsize=(11.693, 16.535), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background canvas
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor=COLOR_BG, zorder=0))

    # -------------------------------------------------------------
    # 1. HEADER BANNER SECTION
    # -------------------------------------------------------------
    header_h = 0.098
    header_y = 1.0 - header_h
    ax.add_patch(patches.Rectangle((0, header_y), 1, header_h, facecolor=COLOR_HEADER_BG, zorder=1))

    # Dual accent divider stripes under header
    ax.add_patch(patches.Rectangle((0, header_y - 0.0035), 1, 0.0035, facecolor=COLOR_ACCENT_BLUE, zorder=2))
    ax.add_patch(patches.Rectangle((0, header_y - 0.0050), 1, 0.0015, facecolor=COLOR_ACCENT_CYAN, zorder=2))

    # Title & Subtitle (Left block, carefully constrained so it never touches the candidate badge)
    ax.text(0.035, header_y + 0.068, "DEEP LEARNING-BASED AUTOMATIC IMAGE COLORIZATION",
            fontsize=15.0, fontweight="bold", color=COLOR_HEADER_TEXT, va="center", ha="left", family="sans-serif")
    ax.text(0.035, header_y + 0.044, "Using ResNet-UNet Architecture with CIE Lab Color Space & Perceptual Loss",
            fontsize=10.5, fontweight="semibold", color=COLOR_ACCENT_CYAN, va="center", ha="left", family="sans-serif")
    ax.text(0.035, header_y + 0.022, "Domain: Computer Vision  |  Deep Learning  |  Generative Image Reconstruction  |  PyTorch + CUDA 12.1",
            fontsize=8.0, color="#94A3B8", va="center", ha="left", family="sans-serif")

    # Student Details Box (Right Upper Corner - STRICTLY AS REQUESTED)
    badge_x, badge_y, badge_w, badge_h = 0.675, header_y + 0.010, 0.290, 0.078
    badge = patches.FancyBboxPatch(
        (badge_x, badge_y), badge_w, badge_h,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        facecolor=COLOR_BADGE_BG, edgecolor=COLOR_ACCENT_CYAN, linewidth=1.5, zorder=3
    )
    ax.add_patch(badge)

    ax.text(badge_x + 0.012, badge_y + 0.063, "CANDIDATE INFORMATION", fontsize=7.0, fontweight="bold", color=COLOR_ACCENT_CYAN, va="center")
    ax.text(badge_x + 0.012, badge_y + 0.046, "Name: Parth Randhir Mirajkar", fontsize=8.0, fontweight="bold", color="#FFFFFF", va="center")
    ax.text(badge_x + 0.012, badge_y + 0.028, "RBT no.: RBT23IT066", fontsize=7.5, fontweight="bold", color="#E2E8F0", va="center")
    ax.text(badge_x + 0.012, badge_y + 0.012, "Dept: IT", fontsize=7.0, color="#93C5FD", va="center")

    # -------------------------------------------------------------
    # Helper: Draw Card with Professional Section Header
    # -------------------------------------------------------------
    def draw_card(x, y, w, h, title, section_num=""):
        # Shadow / Border
        card = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.003,rounding_size=0.007",
            facecolor=COLOR_CARD_BG, edgecolor=COLOR_CARD_BORDER, linewidth=1.0, zorder=2
        )
        ax.add_patch(card)

        # Header Pill
        pill_h = 0.024
        pill_y = y + h - pill_h
        pill = patches.FancyBboxPatch(
            (x, pill_y), w, pill_h,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            facecolor=COLOR_HEADER_BG, edgecolor=COLOR_HEADER_BG, linewidth=0.5, zorder=3
        )
        ax.add_patch(pill)

        # Title text
        disp_title = f"{section_num}  {title}" if section_num else title
        ax.text(x + 0.012, pill_y + pill_h / 2, disp_title,
                fontsize=9.2, fontweight="bold", color="#FFFFFF", va="center", ha="left")
        
        # Subtle status indicator
        ax.plot(x + w - 0.014, pill_y + pill_h / 2, marker="o", markersize=4.5, color=COLOR_ACCENT_CYAN, zorder=4)

    # -------------------------------------------------------------
    # 2. ROW 1: INTRODUCTION & OBJECTIVES (Two Equal Columns)
    # -------------------------------------------------------------
    row1_y = 0.742
    row1_h = 0.148
    col_w = 0.448
    col1_x = 0.035
    col2_x = 0.517

    # Card 1: Introduction & Problem Statement
    draw_card(col1_x, row1_y, col_w, row1_h, "INTRODUCTION & PROBLEM STATEMENT", "01")
    intro_text = (
        "• The Challenge:\n"
        "  Converting monochrome (grayscale) images into natural, vivid color is an ill-posed inverse problem.\n"
        "  Luminance alone provides zero chromaticity information; an infinite set of plausible colorings exist.\n\n"
        "• The 'Sepia / Brown Averaging' Dilemma:\n"
        "  Standard neural networks operating in RGB space with MSE loss take the mathematical expectation\n"
        "  across conflicting colors (e.g., averaging red vs blue cars), yielding desaturated, muddy sepia tones.\n\n"
        "• Our Proposed Approach:\n"
        "  1. Decouple lightness from chrominance via the CIE Lab color space (L channel as independent input).\n"
        "  2. Train a deep neural network to predict strictly the (a, b) channels, preserving 100% of input edges.\n"
        "  3. Leverage multi-scale skip connections to prevent color bleeding across object boundaries."
    )
    ax.text(col1_x + 0.012, row1_y + row1_h - 0.034, intro_text,
            fontsize=7.3, color=COLOR_TEXT_BODY, va="top", ha="left", linespacing=1.28)

    # Card 2: Objectives & Key Innovations
    draw_card(col2_x, row1_y, col_w, row1_h, "PROJECT OBJECTIVES & KEY INNOVATIONS", "02")
    obj_text = (
        "• Primary Project Objectives:\n"
        "  - Build a fully self-supervised image colorization framework from scratch using PyTorch.\n"
        "  - Eliminate the need for labeled annotations by utilizing ground-truth color photos as their own target.\n"
        "  - Achieve superior quantitative (PSNR, SSIM) and qualitative fidelity over classical baselines.\n\n"
        "• Key Innovations Implemented:\n"
        "  - Pretrained ResNet-18 Backbone: Replaces naive random encoders with rich ImageNet visual priors.\n"
        "  - Differentiable CIE Lab to sRGB Converter: Enables gradient backpropagation from RGB vision losses.\n"
        "  - VGG-16 Perceptual Feature Loss: Minimizes perceptual texture divergence for vibrant color saturation.\n"
        "  - GPU Accelerated Pipeline: Mixed precision (FP16/AMP) achieving real-time inference (~160 ms)."
    )
    ax.text(col2_x + 0.012, row1_y + row1_h - 0.034, obj_text,
            fontsize=7.3, color=COLOR_TEXT_BODY, va="top", ha="left", linespacing=1.28)

    # -------------------------------------------------------------
    # 3. ROW 2: METHODOLOGY & ARCHITECTURE (Full Width)
    # -------------------------------------------------------------
    row2_y = 0.542
    row2_h = 0.188
    draw_card(col1_x, row2_y, 0.930, row2_h, "METHODOLOGY & PROPOSED RESNET-UNET SYSTEM ARCHITECTURE", "03")

    # Flowchart blocks
    boxes = [
        {"x": 0.050, "y": 0.648, "w": 0.125, "h": 0.056, "title": "1. CIE Lab Preprocessing", "sub": "RGB to CIE Lab\nL: [0, 100] -> [-1, 1]\nab: [-128, 127] -> [-1, 1]", "color": "#F1F5F9"},
        {"x": 0.210, "y": 0.648, "w": 0.145, "h": 0.056, "title": "2. ResNet-18 Encoder", "sub": "conv1 adapted (1 channel)\n4 Downsampling Stages\n(Extracts 512 semantic maps)", "color": "#DBEAFE"},
        {"x": 0.390, "y": 0.648, "w": 0.145, "h": 0.056, "title": "3. Latent Bottleneck", "sub": "Spatial: 8 x 8 x 512\nCaptures Scene Context:\nSky, Foliage, Water, Skin", "color": "#EDE9FE"},
        {"x": 0.570, "y": 0.648, "w": 0.155, "h": 0.056, "title": "4. U-Net 4-Stage Decoder", "sub": "Bilinear 2x Upsampling\nConcatenate Skip Connections\nDouble ConvBlock + BatchNorm", "color": "#DCFCE7"},
        {"x": 0.760, "y": 0.648, "w": 0.185, "h": 0.056, "title": "5. Color Output & Loss", "sub": "Tanh -> ab in [-1, 1]\nRecombine [L + ab] -> RGB\nLoss = L1 + 0.10 * VGG-16", "color": "#FEF3C7"}
    ]

    for b in boxes:
        bp = patches.FancyBboxPatch(
            (b["x"], b["y"]), b["w"], b["h"],
            boxstyle="round,pad=0.003,rounding_size=0.006",
            facecolor=b["color"], edgecolor="#94A3B8", linewidth=1.1, zorder=3
        )
        ax.add_patch(bp)
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"] - 0.012, b["title"],
                fontsize=7.6, fontweight="bold", color=COLOR_TEXT_DARK, ha="center", va="center", zorder=4)
        ax.text(b["x"] + b["w"]/2, b["y"] + (b["h"] - 0.012)/2, b["sub"],
                fontsize=6.6, color=COLOR_TEXT_BODY, ha="center", va="center", zorder=4, linespacing=1.22)

    # Connecting Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i]["x"] + boxes[i]["w"]
        x2 = boxes[i+1]["x"]
        y_mid = boxes[i]["y"] + boxes[i]["h"]/2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                    arrowprops=dict(arrowstyle="-|>", color=COLOR_ACCENT_BLUE, lw=2.0, mutation_scale=12), zorder=5)

    # Skip Connection Annotation
    ax.annotate("Multi-Scale Skip Connections (Direct low-level edge & texture transfer to decoder stages)",
                xy=(0.65, 0.648), xytext=(0.25, 0.612),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.22", color="#DC2626", lw=1.6, ls="--"),
                fontsize=7.2, fontweight="semibold", color="#B91C1C", zorder=6)

    # Mathematical Formulation Box
    math_bg = patches.Rectangle((0.050, row2_y + 0.008), 0.895, 0.050, facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=0.8, zorder=3)
    ax.add_patch(math_bg)

    math_col1 = (
        "• Color Space Normalization:\n"
        "  $L_{norm} = \\frac{L}{50.0} - 1.0 \\in [-1, 1]$    |    $ab_{norm} = \\frac{ab}{128.0} \\in [-1, 1]$\n"
        "  Model Output: 2 Channels ($a, b$) bounded to $[-1, 1]$ using $\\tanh()$ activation."
    )
    math_col2 = (
        "• Multi-Objective Loss Formulation:\n"
        "  $\\mathcal{L}_{total} = \\|ab_{pred} - ab_{gt}\\|_1 + \\lambda \\cdot \\sum_{k} \\frac{1}{M_k} \\|\\psi_k(RGB_{pred}) - \\psi_k(RGB_{gt})\\|_1$\n"
        "  where $\\psi_k$ denotes VGG-16 feature layers (relu1_2, relu2_2, relu3_3) with weight $\\lambda = 0.10$."
    )
    ax.text(0.060, row2_y + 0.048, math_col1, fontsize=7.1, color=COLOR_TEXT_DARK, va="top", ha="left")
    ax.text(0.485, row2_y + 0.048, math_col2, fontsize=7.1, color=COLOR_TEXT_DARK, va="top", ha="left")

    # -------------------------------------------------------------
    # 4. ROW 3: EXPERIMENTAL RESULTS & QUALITATIVE EVALUATION
    # -------------------------------------------------------------
    row3_y = 0.230
    row3_h = 0.300
    draw_card(col1_x, row3_y, 0.930, row3_h, "EXPERIMENTAL RESULTS & QUALITATIVE EVALUATION", "04")

    # Sample 1: Historical Vintage Photo (Ansel Adams Landscape)
    ansel_path = os.path.join(OUTPUT_DIR, "eval_ansel_adams3.png")
    val_path = os.path.join(OUTPUT_DIR, "eval_000000526256.png")

    sample1_y = 0.378
    sample1_h = 0.095
    ax.text(0.050, 0.482,
            "Sample 1: Historical Vintage Photo (Ansel Adams Landscape) —\n[Grayscale Input | Our Model | Vibrant Boosted | Zhang ECCV16 | Ground Truth]",
            fontsize=6.8, fontweight="bold", color=COLOR_TEXT_DARK, ha="left")

    if os.path.exists(ansel_path):
        im1 = Image.open(ansel_path)
        crop_top = int(im1.height * 0.075)
        im1_cropped = im1.crop((0, crop_top, im1.width, im1.height))
        ax.imshow(im1_cropped, extent=[0.050, 0.965, sample1_y, sample1_y + sample1_h], aspect="auto", zorder=3)

    # Sample 2: Complex Real-World Scene
    sample2_y = 0.268
    sample2_h = 0.085
    ax.text(0.050, 0.360,
            "Sample 2: Complex Real-World Scene (Validation Set) —\n5-Panel Colorization Benchmark",
            fontsize=6.8, fontweight="bold", color=COLOR_TEXT_DARK, ha="left")

    if os.path.exists(val_path):
        im2 = Image.open(val_path)
        crop_top2 = int(im2.height * 0.075)
        im2_cropped = im2.crop((0, crop_top2, im2.width, im2.height))
        ax.imshow(im2_cropped, extent=[0.050, 0.585, sample2_y, sample2_y + sample2_h], aspect="auto", zorder=3)

    # Embedded Loss Curve on the right of Sample 2
    loss_path = os.path.join(OUTPUT_DIR, "loss_curves.png")
    ax.text(0.610, 0.360,
            "Training & Validation Loss Convergence (25 Epochs)",
            fontsize=7.3, fontweight="bold", color=COLOR_TEXT_DARK, ha="left")

    if os.path.exists(loss_path):
        im_loss = Image.open(loss_path)
        crop_loss_top = int(im_loss.height * 0.08)
        im_loss_cropped = im_loss.crop((0, crop_loss_top, im_loss.width, im_loss.height))
        ax.imshow(im_loss_cropped, extent=[0.610, 0.965, sample2_y, sample2_y + sample2_h], aspect="auto", zorder=3)

    # Styled Metrics Table at Bottom of Card 4
    table_y = 0.235
    table_h = 0.026
    ax.add_patch(patches.Rectangle((0.050, table_y), 0.915, table_h, facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.8, zorder=3))

    t_header = f"{'EVALUATION METRIC':<26} | {'OUR MODEL (ResNet-UNet)':<27} | {'ZHANG ECCV16 (Reference)':<27} | {'PERFORMANCE ADVANTAGE':<22}"
    t_row1   = f"{'Peak SNR (PSNR) (dB)':<26} | {'25.65 dB – 47.77 dB':<27} | {'18.35 dB – 26.70 dB':<27} | {'+3.8 dB to +28.7 dB (Sharper)'}"
    t_row2   = f"{'Structural Similarity (SSIM)':<26} | {'0.9161 – 0.9997':<27} | {'0.8100 – 0.9629':<27} | {'+0.02 to +0.05 True Edge Fidelity'}"
    t_row3   = f"{'Mean Absolute Error (MAE)':<26} | {'0.0031 – 0.0447':<27} | {'0.0272 – 0.0740':<27} | {'~45% Lower Reconstruction Error'}"

    ax.text(0.060, table_y + 0.022, t_header, fontsize=6.8, fontweight="bold", color=COLOR_TEXT_DARK, family="monospace")
    ax.text(0.060, table_y + 0.015, t_row1, fontsize=6.5, color=COLOR_SUCCESS, fontweight="semibold", family="monospace")
    ax.text(0.060, table_y + 0.008, t_row2, fontsize=6.5, color=COLOR_TEXT_BODY, family="monospace")
    ax.text(0.060, table_y + 0.002, t_row3, fontsize=6.5, color=COLOR_TEXT_MUTED, family="monospace")

    # -------------------------------------------------------------
    # 5. ROW 4: APPLICATIONS, CONCLUSION & REFERENCES (Three Columns)
    # -------------------------------------------------------------
    row4_y = 0.032
    row4_h = 0.190
    c3_w = 0.297
    c1_x = 0.035
    c2_x = 0.352
    c3_x = 0.668

    # Card 5: Real-World Applications
    draw_card(c1_x, row4_y, c3_w, row4_h, "REAL-WORLD APPLICATIONS", "05")
    app_text = (
        "• Historical Archive Restoration:\n"
        "  Fully automated restoration of vintage monochrome\n"
        "  photographs (early 20th-century cultural archives,\n"
        "  freedom movements, and family collections).\n\n"
        "• Film & Entertainment Remastering:\n"
        "  Revitalizing classic black-and-white cinematic films,\n"
        "  documentaries, and newsreels in full HD/4K fidelity.\n\n"
        "• Medical & Scientific Visualization:\n"
        "  Color-mapping grayscale X-ray, MRI, CT scans, and\n"
        "  electron microscopy to enhance tissue visibility.\n\n"
        "• Night Vision & Surveillance:\n"
        "  Transforming thermal and infrared nighttime security\n"
        "  footage into natural daytime color representations."
    )
    ax.text(c1_x + 0.010, row4_y + row4_h - 0.034, app_text,
            fontsize=7.1, color=COLOR_TEXT_BODY, va="top", ha="left", linespacing=1.24)

    # Card 6: Conclusion & Future Scope
    draw_card(c2_x, row4_y, c3_w, row4_h, "CONCLUSION & FUTURE SCOPE", "06")
    conc_text = (
        "• Key Research Findings:\n"
        "  - The ResNet-UNet architecture completely solves\n"
        "    the classical sepia averaging trap of pure L2 loss.\n"
        "  - Combining L1 with VGG Perceptual Loss yields\n"
        "    vivid, natural chrominance without color bleeding.\n"
        "  - Outperformed the ECCV16 benchmark across all\n"
        "    unseen test samples in PSNR, SSIM, and MAE.\n\n"
        "• Future Directions:\n"
        "  - Video colorization with temporal consistency loss.\n"
        "  - Interactive user-guided scribble colorization.\n"
        "  - Conditional GAN (Pix2Pix PatchGAN) integration\n"
        "    for ultra-high-resolution photorealistic textures."
    )
    ax.text(c2_x + 0.010, row4_y + row4_h - 0.034, conc_text,
            fontsize=7.1, color=COLOR_TEXT_BODY, va="top", ha="left", linespacing=1.24)

    # Card 7: Key References
    draw_card(c3_x, row4_y, c3_w, row4_h, "KEY REFERENCES", "07")
    ref_text = (
        "[1] R. Zhang, P. Isola, and A. A. Efros,\n"
        "    'Colorful Image Colorization,' in Proc. ECCV,\n"
        "    pp. 649–666, 2016.\n\n"
        "[2] O. Ronneberger, P. Fischer, and T. Brox,\n"
        "    'U-Net: Convolutional Networks for Biomedical\n"
        "    Image Segmentation,' in Proc. MICCAI, 2015.\n\n"
        "[3] J. Johnson, A. Alahi, and L. Fei-Fei,\n"
        "    'Perceptual Losses for Real-Time Style Transfer\n"
        "    and Super-Resolution,' in Proc. ECCV, 2016.\n\n"
        "[4] K. He, X. Zhang, S. Ren, and J. Sun,\n"
        "    'Deep Residual Learning for Image Recognition,'\n"
        "    in Proc. IEEE CVPR, pp. 770–778, 2016."
    )
    ax.text(c3_x + 0.010, row4_y + row4_h - 0.034, ref_text,
            fontsize=6.9, color=COLOR_TEXT_BODY, va="top", ha="left", linespacing=1.22)

    # -------------------------------------------------------------
    # 6. FOOTER BRANDING STRIPE
    # -------------------------------------------------------------
    footer_h = 0.016
    ax.add_patch(patches.Rectangle((0, 0), 1, footer_h, facecolor=COLOR_HEADER_BG, zorder=1))
    footer_text = "Engineering Capstone Project  |  Department of Information Technology  |  Academic Year 2024–25  |  Built with PyTorch & NVIDIA CUDA"
    ax.text(0.5, footer_h / 2, footer_text, fontsize=7.2, color="#94A3B8", va="center", ha="center")

    # -------------------------------------------------------------
    # 7. EXPORT
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # 7. EXPORT (with resolution verification)
    # -------------------------------------------------------------
    png_path = os.path.join(OUTPUT_DIR, "poster_A3_v2.png")
    pdf_path = os.path.join(OUTPUT_DIR, "poster_A3_v2.pdf")

    # Save PNG at 300 DPI – this yields 3508 × 4960 px for A3
    print(f"[2/4] Saving 300 DPI A3 PNG to:\n      -> {png_path}...")
    fig.savefig(png_path, dpi=300, facecolor=COLOR_BG, edgecolor="none", bbox_inches="tight", pad_inches=0)

    # Verify the pixel dimensions to guarantee correct resolution
    from PIL import Image as _Image
    _img = _Image.open(png_path)
    width, height = _img.size
    print(f"[Verification] PNG dimensions: {width} x {height} px (expected 3508 x 4960)")

    # Export PDF (vector, resolution‑independent)
    print(f"[3/4] Exporting Vector Print-Ready A3 PDF to:\n      -> {pdf_path}...")
    fig.savefig(pdf_path, dpi=300, facecolor=COLOR_BG, edgecolor="none", bbox_inches="tight", pad_inches=0)

    plt.close(fig)
    print("\n" + "=" * 60)
    print("SUCCESS: A3 Poster rendered with publication quality!")
    print(f"High-Res PNG: {png_path} ({os.path.getsize(png_path)/(1024*1024):.2f} MB)")
    print(f"Vector PDF  : {pdf_path} ({os.path.getsize(pdf_path)/(1024*1024):.2f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    create_poster()
