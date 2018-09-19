import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from ocr.src.digits import cnn_model
from ocr.src.digits import predict as p
from ocr.src.cut import cut_image as ci
from ocr.src.digits import find_numbers as fn
from ocr.src.cut import number_detection as nd
from ocr.src import helpers

def get_img_contour_thresh(img):
    if len(img.shape) >2 :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 135, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))

    # MORPH_OPEN DO ERODE THEN DILATE
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    img2 , contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def crop_image(image,corner1,corner2):
    '''
    corner1: upper-left
    corner2: lopwer right
    corner[x,y]
    '''
    image = image[corner1[1]:corner2[1],corner1[0]:corner2[0]]
    return image

def put_text(img,text,position,font = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,color=(255,0,0),thickness=1,lineType = cv2.LINE_AA ):
    '''
    position : bottom left
    '''
    cv2.putText(img,text,position,font,fontScale,color,thickness,lineType)


def expand_image(image,rate = 1.5):
    img_shape = image.shape
    image = cv2.resize(image ,(int(img_shape[1]*1.5),img_shape[0]))
    return image

def _get_img_area(image):
    img_shape = img.shape
    img_area = img_shape[0]*img_shape[1]
    return img_area

def _offset_bbox(bbox,ratio=0.1):
    ''' Convert and expand boundingbox return from boundingRect
    Args:
        bbox[x,y,w,h] : return from boundingRect
        ratio: the ratio will expand with its size
    Return:
        bbox[(x,y),(x,y)]
    '''
    x,y,w,h = bbox
    offset_x,offset_y = int(w*ratio), int(h*ratio)
    pos1 = (x-offset_x,y-offset_y)
    pos2 = (x + w + offset_x , y + h + offset_y)

    # sanity check
    pos1 = tuple([pos if pos >=0 else 0 for pos in pos1])
    pos2 = tuple([pos if pos >=0 else 0 for pos in pos2])

    bbox = [pos1,pos2]
    return bbox

def _filter_contours_and_convert_to_bbox(contours,img_shape):
    """
    return
        filtered_bbox: bboxes that are likely to contains numbers
        full_bboxes: all bbox found by contours
    """
    filtered_bbox = []
    full_bboxes = []
    for idx,cnt in enumerate(contours):
        bbox  = cv2.boundingRect(cnt)
        x,y,w,h = bbox
        full_bboxes.append(bbox)
        # if h>w*1.00 and h>img_shape[0]*0.3:
        if h>w*1.03 and h>img_shape[0]*0.2:
            # expand and also convert format
            bbox = _offset_bbox(bbox,ratio=0.1)
            filtered_bbox.append(bbox)
    return filtered_bbox ,full_bboxes


# NOTE: need to order the contours first
def recognize(img,model_path='../recognition/model/model01_99.61/model.ckpt'):
    img = expand_image(img,rate=1.5)
    plt.imshow(img)
    plt.show()

    cnts  = get_img_contour_thresh(img)
    bboxes,full_bboxes = _filter_contours_and_convert_to_bbox(cnts,img.shape)

    cropped_images = []
    labels = []
    # need to get all first because if draw first the cropped will contains the drawed rectangle
    if len(bboxes)==1:
        print('Not found any number contour')
        return img,[]
    img_draw = img.copy()
    for bbox in full_bboxes:
        x,y,w,h= bbox
        pos1 = (x,y)
        pos2 = (x+w,y+h)
        cv2.rectangle(img_draw,pos1,pos2,(0,255,0),3)

    plt.imshow(img_draw)
    plt.title('Full bbox')
    plt.show()

    for bbox in bboxes:
        cv2.rectangle(img,bbox[0],bbox[1],(0,255,0),3)
        cropped_image = crop_image(img,bbox[0],bbox[1])
        cropped_images.append(cropped_image)

    # plt.imshow(img)
    # plt.title('After filter')
    # plt.show()

    for idx,cropped_image in enumerate(cropped_images):
        bbox = bboxes[idx]
        label ,score = p.predict_image(cropped_image,model_path)
        put_text(img,str(label),bbox[0])

        labels.append(str(label))
        cv2.rectangle(img,bbox[0],bbox[1],(255,0,0),1)

    plt.imshow(img)
    plt.title('Finished')
    plt.show()

    return img,labels

def _crop_bboxes(image,bboxes):
    """
    Args:
        Image: Original image
        bboxes : bboxes of numbers from find contours
    Return:
        cut_images : list of images cut by bbox
    """
    cut_images = []
    for bbox in bboxes:
        cut_image = crop_image(image,bbox[0],bbox[1])
        cut_images.append(cut_image)
    return cut_images

def get_number_images(image):
    """ End to end cut image contains number """
    cnts  = get_img_contour_thresh(image)
    bboxes,_= _filter_contours_and_convert_to_bbox(cnts,image.shape)
    number_images = _crop_bboxes(image,bboxes)
    return number_images,bboxes

def convert_img(image):
    # change background to be black
    if len(image.shape) > 2:
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    _,image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #_,image = cv2.threshold(image,200,255,cv2.THRESH_BINARY )
    image = cv2.resize(255 - image, (28, 28))
    image = image.flatten()

    # has to be in range [-0.5,0.5]
    image = (image /255.0) -0.5
    return image


def predict_batch(images,model_path):
    """ Predict number from a batch of number images cut from find_contours
    Args:
        images (list) : 2-D images that are feed into mnist model
        model_path (str) : path to mnist model
    Return:
        label (str) : label for input image
    """

    # convert to the format of network input
    x_input = [convert_img(image) for image in images]

    tf.reset_default_graph()

    # Import data
    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 784])
    y = cnn_model.CNN(x, is_training=is_training)

    # Add ops to save and restore all the variables
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

        # Restore variables from disk
        saver = tf.train.Saver()

        saver.restore(sess, model_path)

        y_final = sess.run(y, feed_dict={x: x_input, is_training: False})

        labels = np.argmax(y_final,axis=1)

    label = ''.join(list(labels.astype(str)))
    return label


def predict_number(image,model_path='../recognition/model/model01_99.61/model.ckpt'):
    """ Predict multi-digits image
    Args:
        image: Image cut from pre-defined config
        model_path: path to model to restore
    Return:
        labels : a string of numbers
    """
    img = expand_image(image,rate=1.5)

    number_images,number_bboxes = get_number_images(image)
    if len(number_images) == 0:
        return ''

    label = predict_batch(number_images,model_path)
    return label


def predict_numbers(images,model_path='../recognition/model/model01_99.61/model.ckpt'):
    """ Predict multi-digits from a list of images  """
    labels = []
    for image in images:
        label = predict_number(image,model_path)
        labels.append(label)

    return labels

def predict_row_box_number(image,number_of_box,model_path):
    """ Predict digits inside image with a row of box
    Args:
        image (np.array) : image which inside is a row of linked box with number inside it
        number_of_box (int) : box count in image
        model_path (str) : path to model checkpoint
    Returns:
        label (str) : The return label
    """

    number_images= nd.detect_and_cut_number(image,number_of_box)
    if len(number_images) == 0:
        return 'no'

    label = fn.predict_batch(number_images,model_path)

    return label

def predict_row_box_numbers(images,numbers_of_box,model_path):
    """ API for detect a list of image by predict_row_box_number
    Args :
        images (list) : A list of images
        numbers_of_box (list) : A list of int
        model_path (str) : path to model checkpoint
    Returns:
        labels (list) : A list of labels
    """


    assert len(images) == len(numbers_of_box), 'length of images have be equal to number of numbers_of_box'

    labels = []
    for image,number_of_box in zip(images,numbers_of_box):
        label = predict_row_box_number(image,number_of_box,model_path)
        labels.append(label)
    return labels

def process_result(img,labels):
    plt.imshow(img)
    plt.show()

def main():
    # helpers.test_folder('../../resources/images/result/cut',recognize,
    #         process_result_func=process_result)
    img = cv2.imread('../../resources/images/result/cut/scan3_adjusted3.jpg')
    predict_number(img)



if __name__ == '__main__':
    main()
