from google.cloud import vision
import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time

from utils import draw
from utils import helpers
from utils import image_process as ip
from utils import filters
from utils import cut_image as ci
from utils import enhance_image as enhance


def find_doc(image,show=False):
    '''
    Find and draw the document given an image
    image : a gray scale image
    '''

    original = image.copy()

    image , h_resized, w_resized = ip.resize_image(image,1000)
    bbox = ip.find_document(image)
    if bbox is not None:

        #draw.draw_intersect_list(image,intersect_points)

        #draw.draw_page(image,intersect_points,color=(0,0,255))

        cropped_img = enhance.crop_and_enhance(original,bbox)

        #partial_img,cut_images = ci.cut_image_by_config(cropped_img,'../resources/configs/config.conf')
        partial_img = None
        cut_images = []

    else:
        cropped_img = image
        partial_img = image
        cut_images = []


    return image,cropped_img,partial_img,cut_images

def main():
    args = helpers.setup_arg()

    start = time.time()
    image = cv2.imread('../resources/images/test/real/doc8.jpg')
    image,cropped_img ,partial_im, _ = find_doc(image,False)
    print('done in ',time.time() - start)
    cv2.imwrite('result.jpg',cropped_img)


if __name__ == '__main__':
    main()
