import cv2 

cap = cv2.VideoCapture("D:\padel_cv\infernce_sample_video.mp4")
ret, frame = cap.read()
cv2.imwrite("first_frame.jpg", frame)
cap.release