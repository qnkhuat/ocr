import sys
sys.path.append('../')

import os
import cv2
import math
import logging
import argparse
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from ocr.src.digits import find_numbers as digits
from ocr.src.doc import enhance_image as enhance
from ocr.src.doc import image_process as ip
from ocr.src.cut import cut_image as ci
from ocr.src.checkboxs import predict as checkboxs
from ocr.src import draw
from ocr.src import helpers

class_names = ['No','Yes']
CONFIG_FILE = '../resources/configs/config.conf'
DIGIT_MODEL = '../resources/models/digit/model01_99.61/model.ckpt'
CHECKBOX_MODEL= '../resources/models/checkbox/lastest/model.ckpt'

def setup_args():
    parser = argparse.ArgumentParser(description="python endtoend.py -i ../resources/images/test/full/img1.jpg \
            --checkbox ../resources/models/checkbox/lastest/model.ckpt \
            --digit ../resources/models/digit/model01_99.61/model.ckpt \
            -c ../resources/configs/config.conf \
            -s ../resrouces/images/result/crop \
            ")

    parser.add_argument('-i','--image',dest='image',
            help='Path to image')

    parser.add_argument('-p','--path',dest='images_path',
            help='Path to images folder',default='../resources/images/test/full')

    parser.add_argument('-s','--save_path',dest='save_path',
            help='Path to save folder',default='../resources/images/result/crop')

    parser.add_argument('-c','--config',dest='config',
            help='Path to config file',default=CONFIG_FILE)

    parser.add_argument('--digit',dest='digit',
            help='Path to digit model.ckpt',default=DIGIT_MODEL)

    parser.add_argument('--checkbox',dest='checkbox',
            help='Path to checkbox model.ckpt',default=CHECKBOX_MODEL)

    parser.add_argument('--draw',dest='is_draw',
            help='Draw the cut part on image',action='store_true')

    parser.add_argument('-d','--dynamic',dest='dynamic',
            help='Use dynamic method',action='store_false')

    args = parser.parse_args()
    return args

def annotate(img,infos,digit_model,checkbox_model):
    """ Find , predict and annotate checkboxs and numbers
    Args:
        img (np.array) : Original image
        infos (list) : [[target_name,cut_image,target_bboxes,is_checkboxes],...]
            return from cut_image function
    Returns :
        results (dict) : dict map the target_name with its result after inference
    """
    checkbox_images = []
    number_images= []
    checkbox_idxes = []
    number_idxes = []
    results = {}

    for idx,info in enumerate(infos):
        if info[3]:
            checkbox_images.append(info[1])
            checkbox_idxes.append(idx)
        else:
            number_images.append(info[1])
            number_idxes.append(idx)

    # Detect with mnist model
    if len(number_images) > 0:
        # get list of number of boxes
        numbers_of_box = []
        for info in infos:
            # retrieve number of box from config
            if info[4] is not None:
                numbers_of_box.append(info[4])
        number_labels = digits.predict_row_box_numbers(number_images,
                numbers_of_box, model_path = digit_model)
    else:
        print("Can't find number image")

    # close mnist graph
    tf.reset_default_graph()

    # Detect with checkbox model
    if len(checkbox_images) > 0:
        checkbox_labels = checkboxs.predict_batch(checkbox_images,
                model_path=checkbox_model)
    else:
        print("Can't find checkbox")

    # draw on images
    for i,idx in enumerate(number_idxes):
        pos = infos[idx][2][0]
        label = number_labels[i]
        results[infos[idx][0]] = label
        draw.put_text(img,label,pos,font_scale=2,thickness=3)

    for i,idx in enumerate(checkbox_idxes):
        pos = infos[idx][2][0]
        label = checkbox_labels[i]
        results[infos[idx][0]] = label
        draw.put_text(img,label,pos,font_scale=2,thickness=3)

    return results


def detect(image,config_file,digit_model,checkbox_model,is_draw):
    ''' Detect the doc the classify checkbox + recognitino number in image
    Return:
        image (np.array) : original image
        cropped_img (np.array) : the document inside image if it is
        data (dict) : The dict map the target_name with its result after inference
            {target_name:predicted_value}
    '''

    if isinstance(image,str):
        image = cv2.imread(image)

    image_resized , h_resized, w_resized = ip.resize_image(image,1000)

    # detect the document
    bbox = ip.find_document(image_resized)

    if bbox is not None:
        cropped_img = enhance.crop_and_enhance(image,bbox)
        results = ci.cut_image(
                cropped_img,config_file,
                is_draw=is_draw,is_gray = True)


        data = annotate(cropped_img,results,digit_model,checkbox_model)
    else:
        print('Failed to find documenet')
        cropped_img = image
        cut_images = []
        data = {}

    return image,cropped_img,data

def demo(img_dir,save_dir,args):
    print('Input folder : ',img_dir)
    for img in os.listdir(img_dir):
        if not helpers.verify_filename(img):
            continue
        start = time.time()
        image = os.path.join(img_dir,img)
        print('start',image)
        image,cropped_img,results = detect(image,args.config,args.digit,
                args.checkbox,args.is_draw)
        print('process time: ',time.time() -start)
        cv2.imwrite(os.path.join(save_dir,img),cropped_img)
        print('====================================')
    print('Finished')
    print('Results save at ',save_dir)

def main():
    args = setup_args()
    if args.image is not None:
        filename = args.image.split('/')[-1]

        print('Start ',args.image)
        start = time.time()
        image,cropped_image,results = detect(args.image,args.config,args.digit,
                args.checkbox,args.is_draw)
        cv2.imwrite(os.path.join(args.save_path,filename),cropped_image)
        print('Processed time: ',time.time()-start)
        print('Finished')
        print('Results save at ',args.save_path)
    elif args.images_path is not None:
        image,cropped_img , data = demo(args.images_path,args.save_path,args)
    else:
        print('Invalid arguments')


if __name__ == '__main__':
    main()
