from configparser import Interpolation
from cv2 import IMREAD_COLOR
import matplotlib.pyplot as plt
import cv2
cb_img = cv2.imread("Checkerboard_pattern.svg.png",0)
print(cb_img[0,512])
img_line = cv2.line(cb_img, (200, 400), (1300, 400), color=(0,255,255))
# plt.imshow(img_line)
window1 = cv2.namedWindow("w1")
cv2.imshow(window1, img_line)
cv2.waitKey(5000)
