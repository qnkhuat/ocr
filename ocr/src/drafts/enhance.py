import cv2
import numpy as np
from PIL import ImageEnhance as pi
import PIL

def nothing(x):
    pass
# Creating a window for later use
cv2.namedWindow('result')

# Starting with 100's to prevent error while masking

# Creating track bar
cv2.createTrackbar('h_l', 'result',0,255,nothing)

while(1):

    img = cv2.imread('../resources/test_images/doc/output.jpg')
    #converting to HSV

    # get info from track bar and appy to result
    h_l = cv2.getTrackbarPos('h_l','result')
    img_pil = PIL.Image.fromarray(img)
    contrast = pi.Contrast(img_pil)
    contrast.enhance(5).show()

cap.release()

cv2.destroyAllWindows()
