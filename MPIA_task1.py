import cv2
# create resizable window
cv2.namedWindow("image", cv2.WINDOW_NORMAL)  

# read and display image
image = cv2.imread("C:/Users/bism0/OneDrive/MPIA/Task1.jpg", 1)  # 0 for grayscale image
cv2.imshow("image",image)

# wait until user input for closure
cv2.waitKey(0)
cv2.destroyAllWindows()
