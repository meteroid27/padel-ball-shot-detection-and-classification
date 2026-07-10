import cv2
import numpy as np
from scipy.optimize import minimize_scalar

IMAGE_PATH = ''
EDGE_POINTS_PATH = 'data/calibration/edge_points.npz'

# Which edges are actually curved (real distortion evidence).
# p1_to_p2 is your straight control edge - we'll use it to VALIDATE, not fit.
CURVED_EDGES = ['p2_to_p3', 'p3_to_p4', 'p4_to_p1']
CONTROL_EDGE = 'p1_to_p2'

# --- Load data ---
frame = cv2.imread(IMAGE_PATH)
h, w = frame.shape[:2]
data = np.load(EDGE_POINTS_PATH)

# --- Rough camera matrix (no checkerboard available) ---
cx, cy = w / 2.0, h / 2.0
focal_guess = w  # common heuristic when true focal length is unknown
K = np.array([
    [focal_guess, 0, cx],
    [0, focal_guess, cy],
    [0, 0, 1]
], dtype=np.float32)


def undistort_points(points, k1):
    """
    Takes raw clicked (distorted) pixel points and a candidate k1,
    returns corrected pixel positions using OpenCV's proper inversion.
    """
    dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)  # only solving for k1
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    # P=K tells OpenCV: give me the result back in pixel coordinates (not normalized)
    undistorted = cv2.undistortPoints(pts, K, dist_coeffs, P=K)
    return undistorted.reshape(-1, 2)


def straightness_error(points):
    """
    PCA-based perpendicular-distance error.
    Returns sum of squared distances of points from their own best-fit line.
    Smaller = more collinear = straighter.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    # SVD gives us the principal direction directly (first right-singular vector)
    _, _, vt = np.linalg.svd(centered)
    principal_dir = vt[0]          # direction of maximum spread (the line's direction)
    normal_dir = vt[1]             # perpendicular direction (the "error" direction)
    # Project each centered point onto the normal direction -> perpendicular distance
    perp_distances = centered @ normal_dir
    return np.sum(perp_distances ** 2)


def total_error_for_k1(k1):
    """
    The score for a candidate k1: sum of straightness errors
    across all curved edges after undistorting with this k1.
    """
    total = 0.0
    for edge_name in CURVED_EDGES:
        pts = data[edge_name]
        corrected = undistort_points(pts, k1)
        total += straightness_error(corrected)
    return total


# --- Search for the best k1 ---
result = minimize_scalar(total_error_for_k1, bounds=(-1.0, 1.0), method='bounded')
best_k1 = result.x

print(f"Best k1 found: {best_k1:.6f}")
print(f"Total straightness error at best k1: {result.fun:.4f}")

# --- Validation: check total error at k1=0 (no correction) for comparison ---
baseline_error = total_error_for_k1(0.0)
print(f"Total straightness error at k1=0 (no correction): {baseline_error:.4f}")
print(f"Improvement: {baseline_error - result.fun:.4f}")

# --- Validation using control edge (straight line, should stay straight) ---
control_pts = data[CONTROL_EDGE]
control_error_before = straightness_error(control_pts)
control_corrected = undistort_points(control_pts, best_k1)
control_error_after = straightness_error(control_corrected)
print(f"\nControl edge (p1_to_p2) straightness error:")
print(f"  Before correction: {control_error_before:.4f}")
print(f"  After correction:  {control_error_after:.4f}")
print("  (These should be small and similar - correcting shouldn't distort an already-straight line)")

np.save('k1_coefficient.npy', np.array([best_k1]))
print(f"\nSaved k1 to k1_coefficient.npy")