import numpy as np
import matplotlib.pyplot as plt
import cv2
nature_img = cv2.imread("nature.jpg", cv2.IMREAD_COLOR)
# window1 = cv2.namedWindow("w1")
# cv2.imshow(window1,nature_img)
# cv2.waitKey(5000)
# b,g,r = cv2.split(nature_img)

# plt.figure(figsize=[20,5])
# plt.subplot(141);plt.imshow(r, cmap="grey");plt.title("Red Channel");
# plt.subplot(142);plt.imshow(g, cmap="grey");plt.title("Grey Channel");
# plt.subplot(143);plt.imshow(b, cmap="grey");plt.title("Blue Channel");

# imMerge = cv2.merge((b,g,r))

# plt.subplot(144);plt.imshow(imMerge[:,:,::-1]);plt.title("Original Image");
# img1= cv2.cvtColor(nature_img, cv2.COLOR_BGR2HSV)
# h,s,v = cv2.split(img1)
# plt.figure(figsize=[20,5])
# plt.subplot(141);plt.imshow(h, cmap="grey");plt.title("H Channel");
# plt.subplot(142);plt.imshow(s, cmap="grey");plt.title("S Channel");
# plt.subplot(143);plt.imshow(v, cmap="grey");plt.title("V Channel");
# plt.subplot(144);plt.imshow(nature_img);plt.title("Original Image");
