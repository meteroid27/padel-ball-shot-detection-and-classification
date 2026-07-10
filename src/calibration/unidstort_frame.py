import cv2
import numpy as np

IMAGE_PATH = "first_frame.jpg"
K1_PATH  = "data/calibration/k1_coefficient.npy"
OUTPUT_PATH = "data/processed/first_frame_undistorted.jpg"

frame = cv2.imread(IMAGE_PATH)
h,w = frame.shape[:2]

cx, cy = w/2.0, h/2.0
focal_guess = w

k = np.array([
    [focal_guess , 0, cx],
    [0,focal_guess ,cy],
    [0,0, 1]
])

k1= np.load(K1_PATH)[0]
dist_coeffs = np.array([k1,0,0,0,0], dtype=np.float32)

new_k , roi = cv2.getOptimalNewCameraMatrix(k, dist_coeffs,(w,h), alpha = 0)

undistorted_full = cv2.undistort(frame, k, dist_coeffs, None , new_k)

x,y,roi_w, roi_h = roi
undistorted_cropped = undistorted_full[y:y+roi_h, x:x+roi_w]

cv2.imwrite(OUTPUT_PATH, undistorted_cropped)
print(f"Original size: {w}x{h}")
print(f"Valid ROI after alpha=0: x={x}, y={y}, w={roi_w}, h={roi_h}")
print(f"Saved undistorted (cropped) frame to {OUTPUT_PATH}")