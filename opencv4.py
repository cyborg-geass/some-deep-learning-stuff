from pickle import NONE
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

# s=0
# if len(sys.argv)>1:
#     s=sys.argv[1]

# source = cv2.VideoCapture(s)
# win_name = "camera Preview"
# cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
# while cv2.waitKey(1)!=ord('a'):
#     has_frame, frame = source.read()
#     if not has_frame:
#         break
#     cv2.imshow(win_name, frame)

# source.release()
# cv2.destroyWindow(winname=win_name)
PREVIEW = 0 #preview mode
BLUR = 1 # blurring filter
FEATURES = 2 #corner feature detecter
CANNY = 3 #canny edge detecter

feature_params = dict(maxCorners=500,
                      qualityLevel=0.2,
                      minDistance=15,
                      blockSize=9)
s=0
if len(sys.argv)>1:
    s=sys.argv[1]

image_filter = PREVIEW
Alive =True
win_name = "Camera Filters"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result= None
source = cv2.VideoCapture(s)

while Alive:
    ret, frame = source.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    if image_filter==PREVIEW:
        result = frame
    elif image_filter==BLUR:
        result = cv2.blur(frame, (13,13))
    elif image_filter==CANNY:
        result=cv2.Canny(frame,30,150)
    elif image_filter==FEATURES:
        result = frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_frame, **feature_params)
        if corners is not None:
            for x,y in np.float64(corners).reshape(-1, 2):
                cv2.circle(result, (x,y), 10, (0,255,0),1)
    
    cv2.imshow(win_name,result)
    key = cv2.waitKey(1)
    if key==ord('q') or key==ord('Q') or key==27:
        Alive=False
    elif key==ord('C') or key==ord('c'):
        image_filter=CANNY
    elif key==ord('B') or key==ord('b'):
        image_filter=BLUR
    elif key==ord('P') or key==ord('p'):
        image_filter=PREVIEW
    elif key==ord('f') or key==ord('F'):
        image_filter=FEATURES

source.release()
cv2.destroyWindow(win_name)