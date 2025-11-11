# MPIA_task3

import cv2
import numpy as np
from matplotlib import pyplot as plt

# import image
img_og = cv2.imread("C:/Users/bism0/OneDrive/MPIA/Task3_1_point.png", cv2.IMREAD_COLOR_RGB)

# Reference(input) points
src_point = np.array([(280,1328), (2105,1886), (2549,2524), (467,1880)], dtype=np.float32)

# Affine transformation
dst_affine = np.array([(0,0), (400, 0), (400, 300)], dtype=np.float32)
affine_mat = cv2.getAffineTransform(src_point[:3], dst_affine)
affine_img = cv2.warpAffine(img_og, affine_mat, (400, 300))

# Perspective transformation
dst_psp = np.array([(0,0), (400, 0), (400, 300), (0, 300)], dtype=np.float32)
psp_mat = cv2.getPerspectiveTransform(src_point, dst_psp)
psp_img = cv2.warpPerspective(img_og, psp_mat, (400, 300))

# Display & save results
fig1 = plt.figure(1)
plt.imshow(affine_img)
plt.axis("off")
plt.title("Affine Transformation")
plt.show()

fig2 = plt.figure(2)
plt.imshow(psp_img)
plt.axis("off")
plt.title("Perspective Transformation")
plt.show()

print("affine transform matrix\n", affine_mat)
print("perspective transform matrix\n",psp_mat)

fig1.savefig("affine.png")
fig2.savefig("perspective.png")
