import cv2
import matplotlib.pyplot as plt
import numpy as np



def main():
    img = cv2.imread('output.jpg',0)

    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,1)
    cv2.imwrite('test.jpg',img)



if __name__ == '__main__':
    main()
