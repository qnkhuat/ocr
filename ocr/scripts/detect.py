import sys
sys.path.append('../..')

import cv2
import matplotlib.pyplot as plt
from time import time
import os
import argparse

from ocr.src import helpers
from ocr.src import draw
from ocr.src.doc import enhance_image as enhance
from ocr.src.doc import image_process as ip


DETECT_SAVE_DIR= '../../resources/images/result/detect'
CROP_SAVE_DIR= '../../resources/images/result/crop'

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',action='store_true',
            help='enbale if draw the document')
    parser.add_argument('-i',default=None,help='Image to detect')
    parser.add_argument('--show',action='store_true',
            help='Enable when show the image after detect')

    parser.add_argument('-p',default=None,
            help='Detect and save all images in this folder')
    args = parser.parse_args()

    return args


def detect(image,is_draw=False):
    """ Detect the document inside image 
    Args :
        image (np.array/str) : the image capture a document
    Returns:
        image : original image with draw if is_draw=True
        cropped_image : document image
    """

    # if input path ,load it
    if isinstance(image,str):
        image = cv2.imread(image)

    image_resized , h_resized, w_resized = ip.resize_image(image,1000)
    bbox = ip.find_document(image_resized)

    if bbox is not None:
        # have to draw before crop, if not the bbox will be deviated
        cropped_img = enhance.crop_and_enhance(image,bbox)
        if is_draw:
            image = draw.draw_page(image_resized,bbox)
    else:
        print('Document not found')
        cropped_img = image

    return image,cropped_img

def post_process_result(returns,is_save,show=False):

    for fname,r in returns.items():
        if show:
            plt.figure(figsize=(10,20))
            plt.subplot(121)
            plt.imshow(r[0])
            plt.title('Original')
            plt.subplot(122)
            plt.imshow(r[1])
            plt.title('Cropped')
            plt.show()

        if is_save:
            detect_save_path = os.path.join(DETECT_SAVE_DIR,fname)
            cv2.imwrite(detect_save_path,r[0])

            crop_save_path = os.path.join(CROP_SAVE_DIR,fname)
            cv2.imwrite(crop_save_path,r[1])

    if is_save:
        print('detect images save at ',os.path.abspath(DETECT_SAVE_DIR))
        print('crop images save at ',os.path.abspath(CROP_SAVE_DIR))

def main():
    args = setup_args()
    if args.p is not None:
        helpers.test_folder(args.p,
                detect,process_result_func = post_process_result,
                process_params={'is_draw':args.d},
                process_result_params={'is_save':True,'show':args.show} )
    elif args.i is not None:
        filename = args.i.split('/')[-1]

        start = time()
        img, doc = detect(args.i,args.d)
        print('Processed time',time()-start)

        cv2.imwrite(os.path.join(DETECT_SAVE_DIR,filename),img)
        cv2.imwrite(os.path.join(CROP_SAVE_DIR,filename),doc)

        if args.show:
            plt.subplot(121)
            plt.imshow(img)
            plt.title('Original image')
            plt.subplot(122)
            plt.imshow(doc)
            plt.title('Cropped document')
            plt.show()

        print('detect images saved at ',os.path.abspath(DETECT_SAVE_DIR))
        print('document images saved at ',os.path.abspath(CROP_SAVE_DIR))

    # default , not input anything
    else:
        helpers.test_folder('../../resources/images/test/full',
                detect,process_result_func = post_process_result,
                process_params= {'is_draw':args.d},
                process_result_params={'is_save':True,'show':args.show} )


if __name__ == '__main__':
    main()
