import argparse
import time
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def test_folder(source_path,process_func,process_params=None,
        process_result_func=None,process_result_params=None):
    '''
    one_image : if set is true it will show result image one at a time.
    '''
    fnames = os.listdir(source_path)
    returns = {}
    start_time = time.time()  
    count = 0
    for fname in fnames :
        start = time.time()
        if 'jpg' in fname.lower() or 'png' in fname.lower():
            print('Processing ',fname)
            count+=1

            image_url = os.path.join(source_path,fname)
            img = cv2.imread(image_url)
            if process_params is None:
                returns[fname] = process_func(img)
            else:
                returns[fname] = process_func(img,**process_params)
            print('Process time:',time.time() -start )

    end_time = time.time() - start_time

    print('================')
    print('Processed {} files'.format(count))
    print('Total time:',end_time)
    print('Average time:',end_time/count)


    if process_result_func is not None:
        if process_result_params is None:
            process_result_func(returns)
        else:
            process_result_func(returns,**process_result_params)

    return returns

def verify_filename(filename):
    if 'jpg' in filename.lower() or 'png' in filename.lower():
        return True
    return False

def get_shape(img):
    try :
        [h,w,c] = img.shape
    except:
        [h,w] = img.shape

    return [h,w]

def setup_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i','--images',dest='images',help='Input folder image',default='../resources/test_images/real')
    args = parser.parse_args()
    return args

def imshow(image,title=''):
    plt.imshow(image)
    plt.title(title)
    plt.show()

def get_size(box):
    w = abs( box[1][0] - box[0][0] )
    h = abs( box[1][1] - box[0][1] )
    return h,w
    
def get_center(box):
    x = int( (box[0][0] + box[1][0])/2 )
    y = int( (box[0][1] + box[1][1])/2 )
    return (x,y)
