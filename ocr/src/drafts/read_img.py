import cv2
import matplotlib.pyplot as plt
# img =cv2.imread('output.jpg')
img =cv2.imread('/Users/qnkhuat/Desktop/output.jpg')
img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.imshow(img,cmap='hsv')
plt.show()
