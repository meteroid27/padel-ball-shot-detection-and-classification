import cv2
import numpy as np


class CourtCalibration:
    def __init__(self, k1_path, H_path, raw_frame_shape):

        h, w = raw_frame_shape
        self.orig_size = (w, h)

       
        cx, cy = w / 2.0, h / 2.0
        focal_guess = w
        self.K = np.array([
            [focal_guess, 0, cx],
            [0, focal_guess, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        k1 = np.load(k1_path)[0]
        self.dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)

    
        self.new_K, self.roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist_coeffs, self.orig_size, alpha=0
        )


        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, None, self.new_K,
            self.orig_size, cv2.CV_16SC2
        )

        self.H = np.load(H_path)

    def undistort(self, raw_frame):
        corrected = cv2.remap(raw_frame, self.map1, self.map2, cv2.INTER_LINEAR)
        x, y, roi_w, roi_h = self.roi
        return corrected[y:y+roi_h, x:x+roi_w]

    def pixel_to_court(self, points):
   
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        court_coords = cv2.perspectiveTransform(pts, self.H)
        return court_coords.reshape(-1, 2)


if __name__ == "__main__":

    calib = CourtCalibration(
        k1_path="data/calibration/k1_coefficient.npy",
        H_path="data/callibiration/h_undistort.npy",
        raw_frame_shape=(1080, 1920)
    )

    raw = cv2.imread("data/raw/first_frame.jpg")
    clean = calib.undistort(raw)
    cv2.imwrite("data/processed/test_undistorted.jpg", clean)

    example_point = [(500, 300)]
    court_pos = calib.pixel_to_court(example_point)
    print("Pixel", example_point, "-> Court position (m):", court_pos)