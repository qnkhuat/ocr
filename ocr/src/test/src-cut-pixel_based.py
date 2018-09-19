import sys
sys.path.append('../../..')

from ocr.src.cut import pixel_based as pb
from ocr.src.cut import cut_image as ci
from ocr.src.doc import image_process as ip
from ocr.src import helpers 

import matplotlib.pyplot as plt 
import numpy as np
import os
import cv2



TEST_FOLDER = '../../../resources/images/test/digits'

def compute_sum(img,axis=0):
    max_value  = np.max(img)

    if max_value > 2:
        img = img /255
    img_inv = 1 - img
    
    sum_map= np.sum(img_inv,axis=axis)

    mean = np.mean(sum_map)

    return sum_map,mean

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
    
    img_inv = 1 - img
    sum_x = np.sum(img_inv,axis=0)
    max_x = np.max(sum_x)
    min_x = np.min(sum_x)
    q3 = (max_x - min_x)*3/4
    q1 = (max_x - min_x)*1/4
    std = np.std(sum_x)
    var = np.var(sum_x)

    mean_value = np.sum(sum_x) / len(sum_x)

    plt.plot(np.arange(len(sum_x)),sum_x)
    plt.plot(np.arange(len(sum_x)),np.full(len(sum_x),mean_value),label='mean line')
    #plt.plot(np.arange(len(sum_x)),np.full(len(sum_x),q1),label='q1')
    #plt.plot(np.arange(len(sum_x)),np.full(len(sum_x),q3),label='q3')
    #plt.plot(np.arange(len(sum_x)),np.full(len(sum_x),mean_value + std),label='mean + 1 std')
    #plt.plot(np.arange(len(sum_x)),np.full(len(sum_x),mean_value - std),label='mean - 1 std')
    
    plt.plot(np.arange(len(sum_x)),np.full(len(sum_x),mean_value + var),label='mean + 1 var')
    plt.plot(np.arange(len(sum_x)),np.full(len(sum_x),mean_value - var),label='mean - 1 var')

    plt.xlim(0,len(sum_x))

    plt.legend()
    plt.show()


def remove_noise(img):

    _,img = cv2.threshold(img,240,255,cv2.THRESH_BINARY)
    
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

    mask = mask/255
    out = img * mask 

    kernel = np.ones([2,2])
    out = cv2.erode(out,kernel)
    
    return out


def main():
    for filename in os.listdir(TEST_FOLDER):
        file_path = os.path.join(TEST_FOLDER,filename)
        if not helpers.verify_filename(file_path) :
            continue
        if filename!= 'img1.jpg':
            #continue
            pass

        img = cv2.imread(file_path,0)


        img = remove_noise(img)
        img = img.astype(np.uint8)

        #img = ip.remove_horizontal_vertical_lines(img)
        #plot_sum(img)
        img = img.astype(np.uint8)
        point1,point2 = ci._get_rectangle(img,6)

        img = ci.crop_image(img,point1,point2)
        
        sum_map,mean = compute_sum(img)

        searcher = pb.PixelBased(img,sum_map,6)
        #searcher.view()
        searcher.search_peak(1)



if __name__ == '__main__':
    main()
