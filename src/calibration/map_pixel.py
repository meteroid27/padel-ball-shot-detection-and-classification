import cv2 
import numpy as np

IMAGE_PATH = 'data/processed/first_frame_undistorted.jpg'
SAVE_PATH = 'data/calibration/court_corners.npy'

clicked_points = []

def click_handler(event, x,y,flags,param):
    
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x,y))
        print(f"point {len(clicked_points)} selected:  ({x},{y})")
        
        cv2.circle(param , (x,y) , 8, (0,255,0), -1 )
        cv2.putText(param, f"p{len(clicked_points)}",
                    (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)
        cv2.imshow("court calibration", param)
        
        
        if len(clicked_points) == 4:
            print("\nall 4 corners selected!")
            np.save(SAVE_PATH, np.array(clicked_points, dtype= np.float32 ))
            print("done. press any key to exit.")
            
frame = cv2.imread(IMAGE_PATH)
clone = frame.copy()

cv2.namedWindow("court calibration", cv2.WINDOW_NORMAL)
cv2.resizeWindow('court calibration', 1280, 720)
cv2.imshow("court calibration", clone)
cv2.setMouseCallback("court calibration", click_handler, clone)
print("click p1 and rotate clock-wise ")

cv2.waitKey(0)
cv2.destroyAllWindows()
