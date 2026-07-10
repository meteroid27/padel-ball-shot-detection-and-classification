import cv2 
import numpy as np 

pixels_points = np.load("data/calibration/court_corners.npy")
print(f"the pixel points are { pixels_points}")

real_points = np.array([[0,0],[10,0],[10,20],[0,20]], dtype=np.float32)


H = cv2.getPerspectiveTransform(pixels_points, real_points)

print(H)



