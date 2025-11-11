# MPIA_task5

import cv2
import numpy as np
from matplotlib import pyplot as plt

# import image
img1 = cv2.imread("C:/Users/bism0/OneDrive/MPIA/task5_1.jpg", cv2.IMREAD_COLOR_RGB)
img2 = cv2.imread("C:/Users/bism0/OneDrive/MPIA/task5_2.jpg", cv2.IMREAD_COLOR_RGB)
img3 = cv2.imread("C:/Users/bism0/OneDrive/MPIA/task5_3.jpg", cv2.IMREAD_COLOR_RGB)
img4 = cv2.imread("C:/Users/bism0/OneDrive/MPIA/task5_4.jpg", cv2.IMREAD_COLOR_RGB)
img5 = cv2.imread("C:/Users/bism0/OneDrive/MPIA/task5_5.jpg", cv2.IMREAD_COLOR_RGB)
temp = cv2.imread("C:/Users/bism0/OneDrive/MPIA/task5_temp.jpg", cv2.IMREAD_COLOR_RGB)

# Convert to grayscale
img1_gr = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img2_gr = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
img3_gr = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)
img4_gr = cv2.cvtColor(img4, cv2.COLOR_RGB2GRAY)
img5_gr = cv2.cvtColor(img5, cv2.COLOR_RGB2GRAY)
temp_gr = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

# Show original image
fig1 = plt.figure(1, constrained_layout=True, figsize=(6,3))
plt.title("Original Images", pad=10)
plt.axis("off")
plt.subplot(1,5,1), plt.imshow(img1)
plt.axis("off")
plt.title("1")
plt.subplot(1,5,2), plt.imshow(img2)
plt.axis("off")
plt.title("2")
plt.subplot(1,5,3), plt.imshow(img3)
plt.axis("off")
plt.title("3")
plt.subplot(1,5,4), plt.imshow(img4)
plt.axis("off")
plt.title("4")
plt.subplot(1,5,5), plt.imshow(img5)
plt.axis("off")
plt.title("5")
plt.show()

w, h = temp_gr.shape[::-1]

images = [
    img1,img2,img3,img4,img5,
    img1_gr,img2_gr,img3_gr,img4_gr,img5_gr
]

methods = [
    'cv2.TM_CCOEFF',
    'cv2.TM_CCOEFF_NORMED',
    'cv2.TM_CCORR_NORMED',
    'cv2.TM_SQDIFF_NORMED'
]

for i in range(1,6):
    image = images[i-1]
    gray = images[i+4]
    # Create a figure with 2 rows: heatmap | detection
    fig2 = plt.figure(2, figsize=(10, 3 * len(methods)))
    plt.title(f'Heatmap - Detection, image {i}', pad=30)
    plt.axis('off')

    for j, meth in enumerate(methods):
        method = eval(meth)

        # Apply template matching
        res = cv2.matchTemplate(gray, temp_gr, method)

        # Determine best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc if method in [cv2.TM_SQDIFF_NORMED] else max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Draw rectangle on a copy of the original image
        img_display = image.copy()
        cv2.rectangle(img_display, top_left, bottom_right, (0, 255, 0), 8)

        # Heatmap subplot
        plt.subplot(2, len(methods), j+1)
        plt.imshow(res, cmap='hot')
        plt.title(f'{meth}')
        plt.axis('off')

        # Detection subplot
        plt.subplot(2, len(methods), 5+j)
        plt.imshow(img_display)
        plt.axis('off')

    plt.show()
    fig2.savefig(f"result_{i}.png")

# save figures
fig1.savefig("original.png")
