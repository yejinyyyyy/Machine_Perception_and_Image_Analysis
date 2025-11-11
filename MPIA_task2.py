import cv2
import numpy as np
from matplotlib import pyplot as plt

#===============================================
# Import image
#===============================================
img_og = cv2.imread("C:/Users/bism0/OneDrive/MPIA/Task2_4.jpg", cv2.IMREAD_COLOR_RGB)
img_gr = cv2.imread("C:/Users/bism0/OneDrive/MPIA/Task2_4.jpg", cv2.IMREAD_GRAYSCALE)

#===============================================
# Show histogram by different bins
#===============================================
hist_gr = cv2.calcHist([img_gr],[0],None,[256],[0,256])
hist_gr1 = cv2.calcHist([img_gr],[0],None,[128],[0,256])
hist_gr2 = cv2.calcHist([img_gr],[0],None,[64],[0,256])

fig0 = plt.figure(0, constrained_layout=True, figsize=(4,8))
plt.subplot(311), plt.plot(hist_gr)   
plt.title("Histogram, BIN = 256")
plt.xlabel("Intensity")
plt.ylabel("Pixels")  

plt.subplot(312), plt.plot(hist_gr1)   
plt.title("Histogram, BIN = 128")
plt.xlabel("Intensity")
plt.ylabel("Pixels")  

plt.subplot(313), plt.plot(hist_gr2)   
plt.title("Histogram, BIN = 64")
plt.xlabel("Intensity")
plt.ylabel("Pixels")  

plt.show()

#===================================================
# Histogram equalization in Grayscale
#===================================================
bins = 256

img_eq = cv2.equalizeHist(img_gr)
hist_eq = cv2.calcHist([img_eq],[0],None,[bins],[0,256])

fig1 = plt.figure(1, constrained_layout=True, figsize=(12,6))
plt.subplot(221), plt.imshow(img_gr,'gray')   
plt.axis('off')
plt.title("Original Image in Grayscale")

plt.subplot(223), plt.plot(hist_gr)
plt.xlim([0,bins])
plt.title("Histogram - Original")
plt.xlabel("Intensity")
plt.ylabel("Pixels")    

plt.subplot(222), plt.imshow(img_eq,'gray')
plt.axis('off')
plt.title("Equalized Image in Grayscale")

plt.subplot(224), plt.plot(hist_eq)
plt.xlim([0,bins])
plt.title("Histogram - Equalized(BIN = 256)")
plt.xlabel("Intensity")
plt.ylabel("Pixels")

plt.show()  

#======================================================
# Histogram Equalization in YCrCb (Y-Channel)
#======================================================
img_ycrcb = cv2.cvtColor(img_og, cv2.COLOR_RGB2YCrCb)
img_y = img_ycrcb[:,:,0]

hist_y = cv2.calcHist([img_ycrcb],[0],None,[bins],[0,256]) 

img_y_eq = img_ycrcb.copy()
img_y_eq[:,:,0] = cv2.equalizeHist(img_y)
hist_y_eq = cv2.calcHist([img_y_eq],[0],None,[bins],[0,256])

img_eq_rgb = cv2.cvtColor(img_y_eq, cv2.COLOR_YCrCb2RGB)

fig2 = plt.figure(2, constrained_layout=True, figsize=(12,6))
plt.subplot(221), plt.imshow(img_og)
plt.axis('off')
plt.title("Original Image in RGB")

plt.subplot(223), plt.plot(hist_y)
plt.xlim([0,bins])
plt.title("Histogram - Original")
plt.xlabel("Intensity")
plt.ylabel("Pixels")  

plt.subplot(222), plt.imshow(img_eq_rgb)
plt.axis('off')
plt.title("Equalized(Y-channel) Image in RGB")

plt.subplot(224), plt.plot(hist_y_eq)
plt.xlim([0,bins])
plt.title("Histogram - Equalized(Y-channel, BIN = 256)")
plt.xlabel("Intensity")
plt.ylabel("Pixels")  

plt.show()  

#====================================================================
# Compare Histogram Equalization results (Grayscale vs. Y-channel) 
#====================================================================
fig3 = plt.figure(3, constrained_layout=True, figsize=(6,3))
plt.plot(hist_eq, color = 'k', label = 'grayscale'), plt.plot(hist_y_eq, color = 'r', label = 'Y-channel')
plt.title("Histogram - Equalized(Grayscale & Y-channel), BIN = 256")
plt.xlabel("Intensity")
plt.ylabel("Pixels") 
plt.legend() 

plt.show()


#===================================
# Save figures as png
#===================================
fig0.savefig("fig0_bin.png")
fig1.savefig("fig1_gray.png")
fig2.savefig("fig2_ycrcb.png")
fig3.savefig("fig3_eq.png")