"""
=============================================================================
JELLYBOT FYP — SCENARIO 1: Dome Optics on Land (Air Baseline)
Analysis 
=============================================================================

FOLDER STRUCTURE EXPECTED:
    images/
        no_dome/    ← your Session A images (JPG or PNG)
        dome/       ← your Session B images (JPG or PNG)

CHECKERBOARD SPEC:
    - 9x6 inner corners
    - 30mm square size
=============================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# =============================================================================
# CONFIGURATION 
# =============================================================================

CHECKERBOARD = (9, 6)       # (columns, rows)
SQUARE_SIZE_MM = 30.0       # physical size of each square in millimetres
IMAGE_DIR_NO_DOME = "no_dome/*.jpg"
IMAGE_DIR_DOME    = "dome/*.jpg"
OUTPUT_DIR        = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Build the real-world 3D coordinates of the checkerboard corners
# =============================================================================
# We define the board as lying flat on the Z=0 plane


def make_object_points():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    # mgrid produces a grid of (col, row) indices
    objp[:, :2] = np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ].T.reshape(-1, 2) * SQUARE_SIZE_MM
    return objp


# =============================================================================
# Detect corners in all images for one session
# =============================================================================

def detect_corners(image_glob):
    """
    For each image:
      - Converts to greyscale (corner detection works on intensity, not colour)
      - Finds checkerboard corners with findChessboardCorners()
      - Refines to sub-pixel accuracy with cornerSubPix()
    Returns lists of real-world points and corresponding image points.
    """
    objp = make_object_points()
    all_obj_points = []   # 3D real-world points (same for every image)
    all_img_points = []   # 2D image points (different per image)
    valid_images   = []   # track which images were usable

    # Termination criteria for sub-pixel refinement:
    # stop after 30 iterations OR when accuracy reaches 0.001 pixels
    subpix_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
    )

    image_paths = sorted(glob.glob(image_glob))
    if not image_paths:
        print(f"  [WARNING] No images found at: {image_glob}")
        return [], [], []

    print(f"  Processing {len(image_paths)} images...")

    for path in image_paths:
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # COLOR_BGR2GRAY: OpenCV loads images as BGR (not RGB) because corner detection uses intensity gradients
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
                criteria=subpix_criteria
            )
            all_obj_points.append(objp)
            all_img_points.append(corners_refined)
            valid_images.append(path)
            print(f"    ✓ {os.path.basename(path)}")
        else:
            print(f"    ✗ {os.path.basename(path)} — corners not found, skipping")

    print(f"  {len(valid_images)}/{len(image_paths)} images usable.\n")
    return all_obj_points, all_img_points, valid_images


# =============================================================================
# Run camera calibration
# =============================================================================

def calibrate(obj_points, img_points, image_size):
    """
    calibrateCamera: the core function.
    It takes all the real-world↔image point pairs and solves for:
      - camera_matrix: focal length (fx, fy) and optical centre (cx, cy)
      - dist_coeffs:   [k1, k2, p1, p2, k3]
          k1, k2, k3 = radial distortion (barrel/pincushion)
          p1, p2     = tangential distortion (lens tilt)
      - rms: reprojection error in pixels (how well the model fits)
    """
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    return rms, camera_matrix, dist_coeffs, rvecs, tvecs


# =============================================================================
# Print and compare results
# =============================================================================

def print_results(label, rms, camera_matrix, dist_coeffs):
    print(f"{'='*60}")
    print(f"  SESSION: {label}")
    print(f"{'='*60}")
    print(f"  RMS Reprojection Error : {rms:.4f} px")
    print(f"  (Target: <0.5 no-dome, <1.0 dome)")
    print()
    print(f"  Camera Matrix:")
    print(f"    fx = {camera_matrix[0,0]:.2f} px   fy = {camera_matrix[1,1]:.2f} px")
    print(f"    cx = {camera_matrix[0,2]:.2f} px   cy = {camera_matrix[1,2]:.2f} px")
    print()

    k1, k2, p1, p2, k3 = dist_coeffs.ravel()
    print(f"  Distortion Coefficients:")
    print(f"    k1 = {k1:.6f}  ← primary radial (negative=barrel, positive=pincushion)")
    print(f"    k2 = {k2:.6f}  ← secondary radial")
    print(f"    k3 = {k3:.6f}  ← tertiary radial (usually small)")
    print(f"    p1 = {p1:.6f}  ← tangential")
    print(f"    p2 = {p2:.6f}  ← tangential")
    print()


# =============================================================================
# Undistort a sample image and save it
# =============================================================================

def save_undistorted_sample(image_path, camera_matrix, dist_coeffs, label):
    """
    undistort(): applies the inverse of the distortion model to produce
    a corrected image. Straight lines that were curved will become straight.
    """
    img = cv2.imread(image_path)
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)

    # Save side-by-side comparison
    comparison = np.hstack([img, undistorted])
    out_path = os.path.join(OUTPUT_DIR, f"{label}_undistort_comparison.jpg")
    cv2.imwrite(out_path, comparison)
    print(f"  Saved undistortion comparison → {out_path}")


# =============================================================================
# Visualise the distortion vector field
# =============================================================================

def plot_distortion_field(camera_matrix, dist_coeffs, image_size, label):
    """
    initUndistortRectifyMap(): computes, for every pixel, how far it moves
    during undistortion. We sample a grid of points and draw arrows showing
    the displacement — this creates a vector field.
    """
    h, w = image_size
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1
    )

    # Sample a grid of points
    step = 50
    ys, xs = np.mgrid[step:h:step, step:w:step]
    # For each sampled point (x,y), find where it maps to after undistortion
    src_x = xs.astype(np.float32)
    src_y = ys.astype(np.float32)
    dst_x = map1[ys, xs]
    dst_y = map2[ys, xs]
    # Displacement = how far the pixel moved
    dx = dst_x - src_x
    dy = dst_y - src_y

    plt.figure(figsize=(10, 7))
    plt.quiver(src_x, src_y, dx, dy, scale=1, scale_units='xy',
               angles='xy', color='red', alpha=0.7)
    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.title(f"Distortion Vector Field — {label}\n"
              f"(arrows show pixel displacement during undistortion)")
    plt.xlabel("Image X (px)")
    plt.ylabel("Image Y (px)")
    plt.gca().set_aspect('equal')
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"{label}_distortion_field.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved distortion field → {out_path}")


# =============================================================================
# Regional sharpness analysis (Laplacian variance)
# =============================================================================

def measure_sharpness(image_paths, label):
    """
    For each image, we compute Laplacian variance in three regions:
      - Centre (middle 40% of frame)
      - Edge   (outer ring, excluding corners)
      - Corner (top-left and bottom-right 20% blocks)

    Laplacian: measures rate of intensity change. Sharp edges → high values.
    .var(): variance of all Laplacian values → one number per region.
    High variance = sharp. Low variance = blurry.

    We average across all images in the session to get stable estimates.
    """
    centre_scores, edge_scores, corner_scores = [], [], []

    for path in image_paths:
        img  = cv2.imread(path)
        # Convert to greyscale — sharpness is luminance-based, not colour-based
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Compute Laplacian on the full image
        # CV_64F: use 64-bit float so negative edge responses aren't clipped to 0
        lap = cv2.Laplacian(gray, cv2.CV_64F)

        # Define regions
        cy1, cy2 = int(h * 0.3), int(h * 0.7)   # centre rows
        cx1, cx2 = int(w * 0.3), int(w * 0.7)   # centre cols

        centre_region = lap[cy1:cy2, cx1:cx2]
        edge_region   = np.concatenate([
            lap[0:cy1, :].ravel(),
            lap[cy2:,  :].ravel(),
            lap[:, 0:cx1].ravel(),
            lap[:, cx2: ].ravel()
        ])
        corner_size   = int(min(h, w) * 0.2)
        corner_region = np.concatenate([
            lap[0:corner_size, 0:corner_size].ravel(),
            lap[-corner_size:, -corner_size:].ravel()
        ])

        centre_scores.append(centre_region.var())
        edge_scores.append(np.var(edge_region))
        corner_scores.append(np.var(corner_region))

    print(f"  Sharpness (Laplacian variance) — {label}:")
    print(f"    Centre : {np.mean(centre_scores):.1f}")
    print(f"    Edge   : {np.mean(edge_scores):.1f}")
    print(f"    Corner : {np.mean(corner_scores):.1f}")
    print()

    return {
        "centre": np.mean(centre_scores),
        "edge":   np.mean(edge_scores),
        "corner": np.mean(corner_scores)
    }


def plot_sharpness_comparison(sharpness_no_dome, sharpness_dome):
    regions = ["Centre", "Edge", "Corner"]
    nd_vals = [sharpness_no_dome[k] for k in ["centre", "edge", "corner"]]
    d_vals  = [sharpness_dome[k]    for k in ["centre", "edge", "corner"]]

    x = np.arange(len(regions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, nd_vals, width, label="No Dome",  color="#2196F3")
    ax.bar(x + width/2, d_vals,  width, label="With Dome", color="#F44336")
    ax.set_ylabel("Laplacian Variance (higher = sharper)")
    ax.set_title("Sharpness Comparison by Region — Scenario 1")
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "sharpness_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved sharpness comparison chart → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  SCENARIO 1 — Calibrating: NO DOME session")
    print("="*60)
    obj_nd, img_nd, paths_nd = detect_corners(IMAGE_DIR_NO_DOME)

    print("="*60)
    print("  SCENARIO 1 — Calibrating: DOME session")
    print("="*60)
    obj_d, img_d, paths_d = detect_corners(IMAGE_DIR_DOME)

    # Need at least one valid image in each session to proceed
    if not obj_nd or not obj_d:
        print("[ERROR] Not enough valid images in one or both sessions.")
        exit(1)

    # Get image size from first valid image
    sample_img   = cv2.imread(paths_nd[0])
    image_size   = (sample_img.shape[1], sample_img.shape[0])  # (width, height)

    # Calibrate both sessions
    rms_nd, cam_nd, dist_nd, _, _ = calibrate(obj_nd, img_nd, image_size)
    rms_d,  cam_d,  dist_d,  _, _ = calibrate(obj_d,  img_d,  image_size)

    # Print results
    print_results("NO DOME (Baseline)", rms_nd, cam_nd, dist_nd)
    print_results("WITH DOME",          rms_d,  cam_d,  dist_d)

    # Distortion coefficient delta
    k1_nd = dist_nd.ravel()[0]
    k1_d  = dist_d.ravel()[0]
    print(f"  Δk1 (dome − no dome) = {k1_d - k1_nd:.6f}")
    print(f"  → {'Barrel distortion introduced' if k1_d < k1_nd else 'Pincushion or negligible change'}\n")

    # Undistort sample images
    save_undistorted_sample(paths_nd[0], cam_nd, dist_nd, "no_dome")
    save_undistorted_sample(paths_d[0],  cam_d,  dist_d,  "dome")

    # Distortion vector fields
    plot_distortion_field(cam_nd, dist_nd, (sample_img.shape[0], sample_img.shape[1]), "no_dome")
    plot_distortion_field(cam_d,  dist_d,  (sample_img.shape[0], sample_img.shape[1]), "dome")

    # Sharpness analysis
    print("="*60)
    print("  SHARPNESS ANALYSIS")
    print("="*60)
    sharp_nd = measure_sharpness(paths_nd, "No Dome")
    sharp_d  = measure_sharpness(paths_d,  "Dome")
    plot_sharpness_comparison(sharp_nd, sharp_d)

    print("\n✓ All outputs saved to /outputs/")
    print("  Files generated:")
    print("    no_dome_undistort_comparison.jpg")
    print("    dome_undistort_comparison.jpg")
    print("    no_dome_distortion_field.png")
    print("    dome_distortion_field.png")
    print("    sharpness_comparison.png")
