"""
Coral Reef Image Quality Pipeline
===================================
Evaluates underwater image quality using:
  - UCIQE  (Yang et al., 2015) — chroma variance, luminance contrast, mean saturation
  - RGB channel analysis — mean & std per channel, supporting colour visibility claim

Processes all JPG/PNG images in the same folder as this script.
Outputs:
  - quality_results.csv      : per-image metrics table
  - quality_barchart.png     : UCIQE bar chart figure
  - rgb_channel_analysis.png : RGB mean bars + per-channel histograms
"""

import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ──────────────────────────────────────────────
# UCIQE  (kept faithful to original paper impl)
# ──────────────────────────────────────────────
def uciqe(nargin, loc):
    img_bgr = cv2.imread(loc)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {loc}")
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    if nargin == 1:
        coe_metric = [0.4680, 0.2745, 0.2576]

    img_lum = img_lab[..., 0] / 255
    img_a   = img_lab[..., 1] / 255
    img_b   = img_lab[..., 2] / 255

    img_chr  = np.sqrt(np.square(img_a) + np.square(img_b))
    img_sat  = img_chr / np.sqrt(np.square(img_chr) + np.square(img_lum))
    aver_sat = np.mean(img_sat)
    aver_chr = np.mean(img_chr)
    var_chr  = np.sqrt(np.mean(abs(1 - np.square(aver_chr / img_chr))))

    dtype = img_lum.dtype
    nbins = 256 if dtype == 'uint8' else 65536

    hist, _ = np.histogram(img_lum, nbins)
    cdf = np.cumsum(hist) / np.sum(hist)

    ilow  = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol   = [(ilow[0][0]  - 1) / (nbins - 1),
             (ihigh[0][0] - 1) / (nbins - 1)]
    con_lum = tol[1] - tol[0]

    quality_val = (coe_metric[0] * var_chr +
                   coe_metric[1] * con_lum +
                   coe_metric[2] * aver_sat)
    return quality_val


# ──────────────────────────────────────────────
# RGB channel analysis
# ──────────────────────────────────────────────
def rgb_channel_analysis(loc):
    """
    Computes per-channel mean and std for R, G, B.
    Supports the claim that the camera retains colour information
    across all three channels underwater.
    Returns dict with stats and raw channel arrays for histogram plotting.
    """
    img_bgr = cv2.imread(loc)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {loc}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    ch_r = img_rgb[..., 0]
    ch_g = img_rgb[..., 1]
    ch_b = img_rgb[..., 2]

    return {
        "mean_r"   : float(np.mean(ch_r)),
        "std_r"    : float(np.std(ch_r)),
        "mean_g"   : float(np.mean(ch_g)),
        "std_g"    : float(np.std(ch_g)),
        "mean_b"   : float(np.mean(ch_b)),
        "std_b"    : float(np.std(ch_b)),
        "_channels": (ch_r, ch_g, ch_b),   # kept for histogram plotting; not written to CSV
    }


# ──────────────────────────────────────────────
# RGB figure
# ──────────────────────────────────────────────
def plot_rgb_analysis(results, script_dir):
    """
    Two-row figure:
      Row 1 — grouped R/G/B mean bars (± 1 std) for each image
      Row 2 — overlapping per-channel intensity histograms
    """
    n      = len(results)
    labels = [r["filename"] if len(r["filename"]) <= 20
              else r["filename"][:17] + "..."
              for r in results]

    means_r = [r["mean_r"] for r in results]
    means_g = [r["mean_g"] for r in results]
    means_b = [r["mean_b"] for r in results]
    stds_r  = [r["std_r"]  for r in results]
    stds_g  = [r["std_g"]  for r in results]
    stds_b  = [r["std_b"]  for r in results]

    width  = 0.25
    colors = {"R": "#e74c3c", "G": "#2ecc71", "B": "#3498db"}

    fig, axes = plt.subplots(2, n, figsize=(max(8, n * 3.5), 9),
                             gridspec_kw={"height_ratios": [1.4, 1]},
                             squeeze=False)

    for i in range(n):
        ax = axes[0, i]
        bars_r = ax.bar(0,        means_r[i], width, yerr=stds_r[i],
                        color=colors["R"], label="R", capsize=4,
                        error_kw={"elinewidth": 1.2})
        bars_g = ax.bar(width,    means_g[i], width, yerr=stds_g[i],
                        color=colors["G"], label="G", capsize=4,
                        error_kw={"elinewidth": 1.2})
        bars_b = ax.bar(width*2,  means_b[i], width, yerr=stds_b[i],
                        color=colors["B"], label="B", capsize=4,
                        error_kw={"elinewidth": 1.2})

        ax.set_xticks([width])
        ax.set_xticklabels([labels[i]], fontsize=8)
        ax.set_ylim(0, 280)
        ax.set_ylabel("Mean pixel intensity (0–255)", fontsize=8)
        ax.set_title(f"RGB Means\n{labels[i]}", fontsize=8, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.axhline(128, color="grey", linewidth=0.8, linestyle=":",
                   label="Mid-grey (128)")
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

        for bar_container, val, std in zip(
                [bars_r, bars_g, bars_b],
                [means_r[i], means_g[i], means_b[i]],
                [stds_r[i],  stds_g[i],  stds_b[i]]):
            ax.text(bar_container[0].get_x() + bar_container[0].get_width() / 2,
                    val + std + 4,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    for i, r in enumerate(results):
        ax = axes[1, i]
        ch_r, ch_g, ch_b = r["_channels"]
        bins = np.linspace(0, 255, 64)
        ax.hist(ch_r.ravel(), bins=bins, color=colors["R"], alpha=0.55,
                label="R", density=True)
        ax.hist(ch_g.ravel(), bins=bins, color=colors["G"], alpha=0.55,
                label="G", density=True)
        ax.hist(ch_b.ravel(), bins=bins, color=colors["B"], alpha=0.55,
                label="B", density=True)
        ax.set_xlabel("Pixel intensity", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"Channel Histograms\n{labels[i]}", fontsize=8,
                     fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("RGB Channel Analysis — Coral Reef Images",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(script_dir, "rgb_channel_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"RGB chart saved → {out_path}")


# ──────────────────────────────────────────────
# Find images in the same folder as this script
# ──────────────────────────────────────────────
def find_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    return sorted([
        f for f in os.listdir(folder)
        if os.path.splitext(f.lower())[1] in exts
    ])


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def run_pipeline():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images = find_images(script_dir)

    if not images:
        print("No images found in script folder. Place JPG/PNG files alongside this script.")
        return

    print(f"Found {len(images)} image(s). Processing...\n")

    results = []

    for fname in images:
        fpath = os.path.join(script_dir, fname)
        try:
            score = uciqe(1, fpath)
            rgb   = rgb_channel_analysis(fpath)

            results.append({
                "filename" : fname,
                "uciqe"    : round(score, 4),
                "mean_r"   : round(rgb["mean_r"], 2),
                "std_r"    : round(rgb["std_r"],  2),
                "mean_g"   : round(rgb["mean_g"], 2),
                "std_g"    : round(rgb["std_g"],  2),
                "mean_b"   : round(rgb["mean_b"], 2),
                "std_b"    : round(rgb["std_b"],  2),
                "_channels": rgb["_channels"],   # plotting only; excluded from CSV
            })

            print(f"  {fname}")
            print(f"    UCIQE      : {score:.4f}")
            print(f"    Mean R/G/B : {rgb['mean_r']:.1f} / {rgb['mean_g']:.1f} / {rgb['mean_b']:.1f}")
            print(f"    Std  R/G/B : {rgb['std_r']:.1f} / {rgb['std_g']:.1f} / {rgb['std_b']:.1f}\n")

        except Exception as e:
            print(f"  [ERROR] {fname}: {e}\n")

    if not results:
        print("No results to save.")
        return

    # ── CSV output ──────────────────────────────────────────
    csv_path   = os.path.join(script_dir, "quality_results.csv")
    fieldnames = ["filename", "uciqe",
                  "mean_r", "std_r", "mean_g", "std_g", "mean_b", "std_b"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved → {csv_path}")

    # ── UCIQE bar chart ─────────────────────────────────────
    names  = [r["filename"] for r in results]
    scores = [r["uciqe"]    for r in results]
    labels = [n if len(n) <= 25 else n[:22] + "..." for n in names]

    fig, ax = plt.subplots(figsize=(max(7, len(results) * 1.4), 5))
    bars = ax.bar(labels, scores, color="#2196a8", edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, color="#333333")

    ax.set_xlabel("Image", fontsize=11)
    ax.set_ylabel("UCIQE Score", fontsize=11)
    ax.set_title("UCIQE Image Quality Scores — Coral Reef Images",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(scores) * 1.15)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.axhspan(0.55, 0.65, alpha=0.08, color="green",
               label="Good quality range (0.55–0.65)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(script_dir, "quality_barchart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"UCIQE chart saved → {chart_path}")

    # ── RGB figure ───────────────────────────────────────────
    plot_rgb_analysis(results, script_dir)

    # ── Summary stats ────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────")
    print(f"  Images processed : {len(results)}")
    print(f"  UCIQE  mean      : {np.mean(scores):.4f}")
    print(f"  UCIQE  min/max   : {np.min(scores):.4f} / {np.max(scores):.4f}")
    print(f"  Mean R (avg)     : {np.mean([r['mean_r'] for r in results]):.1f}")
    print(f"  Mean G (avg)     : {np.mean([r['mean_g'] for r in results]):.1f}")
    print(f"  Mean B (avg)     : {np.mean([r['mean_b'] for r in results]):.1f}")
    print("─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    run_pipeline()