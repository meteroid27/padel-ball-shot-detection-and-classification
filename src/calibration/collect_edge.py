import cv2
import numpy as np

IMAGE_PATH = 'first_frame.jpg'
SAVE_PATH = "edge_points.npz"

EDGE_NAMES = ["p1_to_p2","p2_to_p3","p3_to_p4","p4_to_p1"]
EDGE_COLORS = [(0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 0, 255)]

edges = {name: [] for name in EDGE_NAMES}
current_edge_idx = 0

def redraw(frame):
    display = frame.copy()
    for idx, name in enumerate(EDGE_NAMES):
        color = EDGE_COLORS[idx]
        for (x,y) in edges[name]:
            cv2.circle(display, (x,y), 5,color, -1)
        cv2.putText(display, f"Current edge: {EDGE_NAMES[current_edge_idx]} | 'n' = next edge | 'q' = finish",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("edge_point_collection", display)
            
    
def click_handler(event,x,y,flags,params):
    global current_edge_idx 
    frame = params
    if event == cv2.EVENT_LBUTTONDOWN:
        name = EDGE_NAMES[current_edge_idx]
        edges[name].append((x,y))
        print(f"[{name}] point {len(edges[name])} added: ({x}, {y})")
        redraw(frame)
        
frame = cv2.imread(IMAGE_PATH)
clone = frame.copy()

cv2.namedWindow("edge_point_collection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("edge_point_collection", 1280, 720)
cv2.setMouseCallback("edge_point_collection", click_handler, clone)

redraw(clone)

while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('n'):
        if current_edge_idx < len(EDGE_NAMES)-1:
            current_edge_idx += 1
            print(f"nMoving to edge: {EDGE_NAMES[current_edge_idx]}")
            redraw(clone)
        else:
            print("\nAlready on the last edge. Press 'q' to save and quit.")
            
    elif key == ord('q'):
        break
    
cv2.destroyAllWindows()

save_dict  = {name: np.array(edges[name], dtype = np.float32) for name in EDGE_NAMES}
np.savez(SAVE_PATH , **save_dict)

print("\nSaved point counts:")
for name in EDGE_NAMES:
    print(f"  {name}: {len(edges[name])} points")
print(f"\nSaved to {SAVE_PATH}")


