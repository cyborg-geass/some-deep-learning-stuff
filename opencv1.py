from multiprocessing.resource_sharer import stop
import matplotlib
import numpy as np
import cv2
from IPython.display import Image
import matplotlib.pyplot as plt
# %matplotlib inline
print(cv2.__version__)
cb_pattern = cv2.imread("Checkerboard_pattern.svg.png")
# print(cb_pattern)
coke_img = cv2.imread("unnamed.jpg")
# print("The size of the checkerboard is: ", cb_pattern.shape)
# print(cb_pattern.dtype)
# print("The size of the coke image is: ", coke_img.shape)
# print(coke_img.dtype)
# plt.imshow(coke_img)
# Image(filename="Checkerboard_pattern.svg.png")
# plt.show
# cv2.imshow("Image", cb_pattern)
# cv2.waitKey(8000)
# cv2.imshow("CocaCola", coke_img)
# cv2.waitKey(5000)
# plt.imshow(cb_pattern)
# plt.title("Checkerboard")
# plt.show()
# window1 = cv2.namedWindow("w1")
# cv2.imshow(window1, cb_pattern)
# cv2.waitKey(6000)
# cv2.destroyWindow(window1)
# window2 = cv2.namedWindow("w2")
# cv2.imshow(window2,coke_img)
# cv2.waitKey(6000)
# cv2.destroyWindow(window2)
# window3 = cv2.namedWindow("w3")
# cv2.imshow(window3, cb_pattern)
# cv2.waitKey(0)
# cv2.destroyWindow(window3)

# window4 = cv2.namedWindow("w4")
# Alive = True
# while Alive:
#     cv2.imshow(window4, coke_img)
#     keypress = cv2.waitKey(1)
#     if keypress == ord('q'):
#         Alive = False
# cv2.destroyWindow(window4)

# cv2.destroyAllWindows()
# stop=1
plt.imshow(coke_img)
plt.title("CocaCola")
plt.show()
coke_img_reversed = coke_img[:,:,::-1]
plt.imshow(coke_img_reversed)
plt.title("Coke_Org")
plt.show()