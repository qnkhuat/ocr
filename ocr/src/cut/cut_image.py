import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import re
from unidecode import unidecode
from ocr.src import helpers
from ocr.src.doc import image_process as ip
from ocr.src import draw
"""
NOTATIONS
-   Target box: the box we need to draw
-   Anchor word : the word we select to be the anchor in order to find the target box
"""


def crop_image(image,corner1,corner2):
    '''
    corner1: upper left
    corner2: bottom right
        corner[x,y]
    '''
    image = image[corner1[1]:corner2[1],corner1[0]:corner2[0]]
    return image

def crop_images(image,bboxes):
    """ Extened of crop_image
    Args:
        image (np.array) : 2-D image
        bboxes (list): list of bboxes
            ((x1,y1),(x2,y2)) : upper-left ,botton right
    Returns:
        cropped_images (list) : a list of cropped images
    """
    cropped_images = []

    for bbox in bboxes:
        number_image = crop_image(image,bbox[0],bbox[1])
        cropped_images.append(number_image)

    return cropped_images


def convert_target_box(anchor_config,image_shape):
    """ Convert from ratio in config to bounding box correspond to image shape
    Argss:
        anchor_config (dict)
        image_shape (np.array/list)
    Returns:
        target_bbox (list) : a list of upperleft, bottom right point of bbox
    """
    box_ratio = anchor_config[1]
    h,w = image_shape[:2]
    target_bbox =  [(int(box_ratio[0][0]*w),int(box_ratio[0][1]*h)),
                    (int(box_ratio[1][0]*w),int(box_ratio[1][1]*h))]

    return target_bbox


def cut_image(image,config_file,is_static=True,is_draw=False,is_gray=True):
    '''
    CONFIG_FORMAT
        anchor_config[target_name,target_bbox,is_checkbox]
    Args:
        image : Full image to cut
        config_file: path to config file
        is_static : Whether use STATIC method or DYNAMIC
        is_draw : True to draw on return image
        is_gray : True when return gray image

    Return:
        [[target_name,cut_image,target_bboxes,is_checkboxes,number_of_box],...]
        In which:
            target_name: the name of target
            cut_images: list of cut images
            target_bboxes: list of bbox of those images
            is_checkboxes: list of bool indicate which box is checbox
            number_of_box : if not checkbox so how many numbers in it

    '''

    config = json.load(open(config_file))
    image_shape = image.shape[:2]

    # store vars
    results = []

    # Loop thourgh each config
    for idx,anchor_config in enumerate(config):
        result = [None,None,None,None,None]


        target_bbox = convert_target_box(anchor_config,image_shape)

        # add to results
        # target name
        result[0] = anchor_config[0]
        # is_checkbox
        result[3] = anchor_config[2]
        # number of box
        result[4] = anchor_config[3]

        if target_bbox == None:
            print("Can't find target bbox")
            continue

        cropped_image = crop_image(image,target_bbox[0],target_bbox[1])

        if is_gray :
            cropped_image = cv2.cvtColor(cropped_image,cv2.COLOR_RGB2GRAY)

        if is_draw:
            draw.draw_rectangle(image,target_bbox[0],target_bbox[1])

        # Add to results
        result[1] = cropped_image
        result[2] = target_bbox

        results.append(result)
    return results
