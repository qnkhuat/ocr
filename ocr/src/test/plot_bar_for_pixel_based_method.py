import sys
sys.path.append('../../..')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


from ocr.src.doc.image_process import remove_horizontal_vertical_lines as remove
from ocr.src.cut import cut_image as ci
from ocr.src import helpers


TEST_FOLDER = '../../../resources/images/test/digits'

def remove_noise(img):

    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

    mask = mask/255
    out = img * mask 

    kernel = np.ones([2,2])
    out = cv2.erode(out,kernel)
    
    return out

def plot_sum(img,title=''):

    plt.figure(figsize=(10,6))

    # plot the image
    plt.subplot(211)
    plt.title(title)
    plt.imshow(img)

    # plot the density map
    plt.subplot(212)
    max_value  = np.max(img)

    if max_value > 2:
        img = img /255
    
    sum_x = np.sum(1 - img,axis=0)

    mean_value = np.sum(sum_x) / len(sum_x)

    plt.plot(np.arange(len(sum_x)),np.full(len(sum_x),mean_value),label='mean line')
    plt.plot(np.arange(len(sum_x)),np.full(len(sum_x),mean_value*4/3),label='1/4')
    plt.plot(np.arange(len(sum_x)),np.full(len(sum_x),mean_value*2/3),label='3/4')
    plt.plot(np.arange(len(sum_x)),sum_x,label='sum map')

    plt.xlim(0,len(sum_x))

    plt.legend()
    plt.show()


for filename in os.listdir(TEST_FOLDER):
    file_path = os.path.join(TEST_FOLDER,filename)
    if not helpers.verify_filename(file_path) :
        continue
    if filename!= 'img1.jpg':
        #continue
        pass

    img = cv2.imread(file_path,0)

    point1,point2 = ci._get_rectangle(img,6)

    img = ci.crop_image(img,point1,point2)

    #img = remove(img)

    _,img= cv2.threshold(img,240,1,cv2.THRESH_BINARY)
    #img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    #img = cv2.bitwise_not(img)


    img = remove_noise(img)

    plot_sum(img,filename)


