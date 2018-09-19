import cv2
import json
import numpy as np

from ocr.src.cut import cut_image as ci
from ocr.src import draw
from ocr.src import helpers

def _convert_input_bbox(target_bbox):
    ''' Convert from input to usable bbox'''
    target_bbox = target_bbox.split(',')
    target_bbox = list(map(lambda x:int(x),target_bbox))
    target_bbox = [[target_bbox[0],target_bbox[1]],[target_bbox[2],target_bbox[3]]]
    return target_bbox


def process_config_data(target_name,target_bbox,image_shape,number_of_box=None,checkbox=False):
    '''
    target_bbox: has to be the string that input(ex : '1,2,3,4')
    checkbox : True when this is a box contains checkbox else number/word
    Return :
        [target_name,cut_image,target_bboxes,is_checkboxes,number_of_box]
        In which:
            target_name: the name of target
            cut_images: list of cut images
            target_bboxes: list of bbox of those images
            is_checkboxes: list of bool indicate which box is checbox
            number_of_box : if not checkbox so how many numbers in it
    '''
    h,w = image_shape[:2]
    target_bbox = _convert_input_bbox(target_bbox)
    target_bbox = np.asarray(target_bbox)

    # target box has to be inside image
    if 0 < target_bbox[0][0]/w < w and 0< target_bbox[1][0] < w \
        and 0< target_bbox[0][1]< h and 0< target_bbox[1][1] <h:
        pass
    else:
        print('Invalid target_bbox for ',target_name)
        return None

    # upper left
    point1 = (target_bbox[0][0]/w,target_bbox[0][1]/h)
    # bottom right
    point2 = (target_bbox[1][0]/w,target_bbox[1][1]/h)

    target_bbox = [point1,point2]


    return [target_name , target_bbox ,checkbox,number_of_box]
