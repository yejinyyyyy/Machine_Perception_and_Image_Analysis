# MPIA_task4

import cv2
import numpy as np
from matplotlib import pyplot as plt

# import image
img = cv2.imread("C:/Users/bism0/OneDrive/MPIA/Task5.jpg", cv2.IMREAD_GRAYSCALE)

fig1 = plt.figure(1, figsize=(3,6))
plt.title("Canny edge detection",pad=30)
plt.axis("off")
for i in range (0,4):
    canny = cv2.Canny(img, threshold1=10+40*i, threshold2=110+40*i)
    plt.subplot(2,2,i+1), plt.imshow(canny, cmap="gray")
    plt.axis("off")
    plt.title("threshold={}-{}".format(10+40*i,110+40*i))
plt.show()

# Sobel gradients in x and y
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_5x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobel_5y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

edge_x = cv2.convertScaleAbs(sobel_x)
edge_y = cv2.convertScaleAbs(sobel_y)
edge_5x = cv2.convertScaleAbs(sobel_5x)
edge_5y = cv2.convertScaleAbs(sobel_5y)

# Magnitude of gradients
magnitude = cv2.magnitude(sobel_x, sobel_y)
magnitude = cv2.convertScaleAbs(magnitude)
magnitude5 = cv2.magnitude(sobel_5x, sobel_5y)
magnitude5 = cv2.convertScaleAbs(magnitude5)

fig2 = plt.figure(2, constrained_layout=True, figsize=(8,4))
plt.subplot(231), plt.imshow(edge_x, cmap="gray")
plt.axis("off")
plt.title("Sobel-x filter, k=3")

plt.subplot(232), plt.imshow(edge_y, cmap="gray")
plt.axis("off")
plt.title("Sobel-y filter, k=3")

plt.subplot(233), plt.imshow(magnitude, cmap="gray")
plt.axis("off")
plt.title("Magnitude")

plt.subplot(234), plt.imshow(edge_5x, cmap="gray")
plt.axis("off")
plt.title("Sobel-x filter, k=5")

plt.subplot(235), plt.imshow(edge_5y, cmap="gray")
plt.axis("off")
plt.title("Sobel-y filter, k=5")

plt.subplot(236), plt.imshow(magnitude5, cmap="gray")
plt.axis("off")
plt.title("Magnitude")

plt.show()


# Apply erosion and dilation
fig3 = plt.figure(3,figsize=(6,6))
plt.title("Erosion",pad=30)
plt.axis("off")
for i in range(0,2):
    kernel = np.ones((3+2*i, 3+2*i), np.uint8)  # Define a kernel
    for j in range(1,3):
        erosion = cv2.erode(magnitude5, kernel, iterations=j)
        plt.subplot(2,2,j+i*2), plt.imshow(erosion, cmap="gray")
        plt.title("kernel={}, iter={}".format(3+2*i,j))
        plt.axis("off")
plt.show()

fig4 = plt.figure(4, figsize=(8,8))
plt.title("Dilation",pad=30)
plt.axis("off")
for i in range(0,3):  # 2 times
    kernel = np.ones((3+2*i, 3+2*i), np.uint8)  # Define a kernel
    for j in range(1,4): # 3times
        dilation = cv2.dilate(magnitude, kernel, iterations=j)
        plt.subplot(3,3,j+i*3), plt.imshow(dilation, cmap="gray")
        plt.title("kernel={}, iter={}".format(3+2*i,j))
        plt.axis("off")
plt.show()

# Final comparison (Magnitude vs. Erosion vs. Dilation)
kernel = np.ones((3,3), np.uint8)
erosion_ = cv2.erode(magnitude, kernel, iterations=1)
dilation_ = cv2.dilate(magnitude, kernel, iterations=1)

fig5 = plt.figure(5, figsize=(8,4))
plt.subplot(131), plt.imshow(magnitude, cmap="gray")
plt.axis("off")
plt.title("Original (Magnitude)")

plt.subplot(132), plt.imshow(erosion_, cmap="gray")
plt.axis("off")
plt.title("Erosion")

plt.subplot(133), plt.imshow(dilation_, cmap="gray")
plt.axis("off")
plt.title("Dilation")

plt.show()

# save figures
fig1.savefig("cannyedge.png")
fig2.savefig("sobel.png")
fig3.savefig("Erosion.jpg")
fig4.savefig("Dilasion.png")
fig5.savefig("comparison.png")