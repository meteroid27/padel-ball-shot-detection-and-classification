import cv2
import numpy as np

pixel_points = np.load("court_corners.npy")
scale = 20

width , height = 10, 20
s_width , s_height = int(scale*10) , int(scale*20)
scaled_real_points = np.array([[0,0], [s_width , 0], [s_width , s_height],[0,s_height]], dtype= np.float32)

h_display = cv2.getPerspectiveTransform(pixel_points, scaled_real_points)
frame = cv2.imread("first_frame.jpg")

birds_eye = cv2.warpPerspective(frame , h_display , (s_width, s_height))

cv2.imshow("birds_eye", birds_eye)
cv2.waitKey(0)
cv2.destroyAllWindows()





