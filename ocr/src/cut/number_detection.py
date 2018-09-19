import cv2
import numpy as np

from ocr.src.cut import cut_image as ci

def detect_rectangle(image,number_of_box):
    """ Cut the rectangle inside image using findcontours
    Criterions to filter contours: largest contour, the width/height ratio has to be
        corresponse with the number_of_box
    Args:
        image (np.array) : image to cut
        number_of_box (int) : how many box number inside this image
    Returns:
        point,point (tuple) : 2 point of rectangle box
    """
    # Avoid side effect on original image
    img = image.copy()

    if len(img.shape) != 2 :
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    # do theshold
    # better when use cv2.adaptiveThreshold than cv2.threshold
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    thresh = cv2.bitwise_not(thresh)

    _, cnts , _ = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter to get the best contour
    # Temp var for optimal values
    max_area,x_op ,y_op,w_op,h_op = 0,0,0,0,0
    image_area = img.shape[0]*img.shape[1]
    for cnt in cnts:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        approx_area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)

        if (approx_area > max_area and
            w > h*( number_of_box - 3 ) ):# The width of row should be at least n-3 time width
            # update optimal values
            max_area = approx_area
            x_op,y_op,w_op,h_op = x,y,w,h

    # if these variable are not changed means didn't find any box
    if w_op == 0 and h_op == 0 :
        return (None,None),(None,None)
    else :
        return (x_op,y_op),(x_op+w_op,y_op+h_op)

def remove_border(img,width=2,color=255):
    """ Replace some pixel at the border of image by white color
    Args:
        image (np.array) : an 2-D image
        width (int) : the width of the border need to remove
        color (int) : the color of the border
    Return:
        image (np.array) : an 2-D image
    """
    mask = np.ones_like(img)*color
    mask[width:-width,width:-width] = img[width:-width,width:-width]
    return mask

def remove_borders(images,width=2,color=255):
    """ The extent version of remove_border for a list of image
    Args:
        images (list) : a list of 2-D images
        width (int) : the width of the border will remove
        color (int) : the color of the border

    Return:
        results (list) : a list of 2-D images
    """
    results = []
    for image in images:
        results.append(remove_border(image,width))
    return results


def _divide_box(image,number_of_box,bounding_width=3):
    """ Divide box acording to the number of box inside image
    Args:
        image (np.array) : an 2-D image contains boxes
        number_of_box (int) : how many box in side this image
        bounding_width (int) : the width of bounding line
    Returns:
        cut_images (list) : a list of cut image box
    """
    h,w = image.shape[:2]

    # compute the boxsize to cut
    box_size = int((w - bounding_width*(number_of_box + 1 )) / number_of_box)

    cut_images = []

    # cut with the full size of image
    for idx,i in enumerate(range(number_of_box)):
        x1 = (i+1)*bounding_width + i*box_size
        y1 = 0
        x2 = x1 + box_size
        y2 = h

        cut_image = ci.crop_image(image,(x1,y1),(x2,y2))
        cut_images.append(cut_image)


    return cut_images



def _spot_number(image,number_of_box):
    """ Use findcontours to spot the number inside image
    Approach : because the findcontours return many contour, we create
        some creterions for each contour to gain score.
        Then use this score to filter the contours we need
    Args:
        image (np.array) : a gray image (should be the row of numbers box)
    Returns:
        bboxes (list) : a list of number bbox
            ((x1,y1),(x2,y2))
    """

    img = image.copy()

    img_h,img_w = img.shape[:2]
    img_area = img_h*img_w

    # approximate the width of box number
    approx_box_width = img_w/(number_of_box*2)

    _,thresh = cv2.threshold(img,200,255,cv2.THRESH_BINARY)

    _,cnts,_ = cv2.findContours(thresh,cv2.RETR_TREE , cv2.CHAIN_APPROX_NONE)


    # to store the score that later we compare to find the best contours
    storer = []

    for idx,cnt in enumerate(cnts) :
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        approx_area = cv2.contourArea(approx)
        x,y,w,h = cv2.boundingRect(cnt)

        # skip if the box to big or too small( dont use w because the number 1 has very small width)
        if w > approx_box_width*3 or h<10 or h > (img_h-4):
            continue
        # the approx area shouldn't be too small
        elif approx_area < approx_area/1000*5:
            continue
        # the area of box contour should be small
        elif approx_area > img_area/(number_of_box*1.5):
            continue
        #if True:
        else :
            bbox_spec = dict(score=0,x=0,# x for sort the order of output
                    bbox=[])

            # more approx means more score
            if len(approx) > 5 :
                bbox_spec['score']+=1

            if len(approx)>8:
                bbox_spec['score']+=2

            if len(approx)>15:
                bbox_spec['score']+=3

            # scrore by width and height
            if approx_area > 10:
                bbox_spec['score']+=1

            if h > img_h/3 :
                bbox_spec['score']+=2

            if h > img_h/4 :
                bbox_spec['score']+=1

            if w > approx_box_width :
                bbox_spec['score']+=2

            # the center height of contour should be near the center height of image
            if 2*img_h/3 > (y+h)/2 > img_h/3:
                bbox_spec['score'] +=2

            if 3*img_h/4 > (y+h)/2 > img_h/4:
                bbox_spec['score'] +=1

            # store the bounding box
            bbox_spec['bbox'] = [(x,y) , (x+w,y+h)]
            # to sort the order of output
            bbox_spec['x'] = x

            storer.append(bbox_spec)


    score_sorted = sorted(storer ,
            key = lambda d: d['score'],reverse=True)

    # just get enough
    results = score_sorted[:number_of_box]
    #results = score_sorted

    # sort by x in order to get the right or der
    results_sorted = sorted(results,
            key = lambda d: d['x'])

    bboxes = [result['bbox'] for result in results_sorted]

    return bboxes

def _filter_number_from_contours(cnts,image):
    """ A set of rules to filter the appropriate number given contours """
    img_h,img_w = image.shape
    img_area = img_h*img_w
    max_area,x_op ,y_op,w_op,h_op = 0,0,0,0,0

    for idx,cnt in enumerate(cnts) :
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        approx_area = cv2.contourArea(approx)
        x,y,w,h = cv2.boundingRect(cnt)

        x_center = x+w/2
        y_center = y+h/2


        if img_w*.2 < x_center <img_w*.8 and img_h*.2 < y_center < img_h*.8: # the center of box should be at the center of image
            if approx_area > 5 and h > 10 and approx_area > max_area and approx_area < img_area*.7:
                if w < img_w*.8 :
                    max_area = approx_area
                    x_op,y_op,w_op,h_op = x,y,w,h

    point1,point2 = (x_op,y_op), (x_op+w_op , y_op + h_op)

    return point1,point2

def _spot_number_batch(images):
    """ Spot the number in a list of images(each image is a image with single number inside)
    NOTE : disvantage for this approach compare to _spot_number is that this method depend on
        the preivous step(cut a row of number image into number_of_box image. If that step cut the image
        wrong(the number isn't at the center of image) then this method will fail)
    Args:
        images (list) : a list of images
    Returns :
        number_images (list) : a list of (28x28) image that are ready to fit into mnist model
    """
    number_images = []

    for image in images:

        _,thresh = cv2.threshold(image,200,255,cv2.THRESH_BINARY)

        _,cnts,_ = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        point1,point2 = _filter_number_from_contours(cnts,image)

        # if can't find that image, just include it into the result
        if point1 == (0,0) and point2 == (0,0) :
            # this mean that filter can't find the appropirate box
            mnist_image = image
            #helpers.imshow(image,'shit')
        else :
            number_image = ci.crop_image(image,point1,point2)
            mnist_image = convert_to_image_mnist(number_image)


        # make the take bolders
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # mnist_image = cv2.erode(mnist_image,kernel)

        number_images.append(mnist_image)
    return number_images

def _resize_image_mnist(image):
    """ Resize the input image to be either 20 width or 20 height
    to fit the regular size of mnist data
    Args:
        image (np.array): an image

    Returns:
        resized_image (np.array)
    """
    h,w = image.shape[:2]
    # if h > w expand the height to 20
    if h > w :
        ratio = h/w
        new_w = int(w*20/h)
        resized_image = cv2.resize(image,(new_w,20))

    # if h < w expand the width to 20
    else :
        ratio = h/w
        new_h = int(h*20/w)
        resized_image = cv2.resize(image,(20,new_h))

    return resized_image


def _center_image_mnist(image):
    """ Put the 20pixel image at the center of a 28x28 box
    Args:
        image (np.array): an image that has at least 1-D is 20px
    Returns:
        mnist_image (np.array) : 28x28 image that contain the input image
    """
    # a white image
    mnist_image = np.ones([28,28],dtype=np.uint8)*255
    h , w = image.shape[:2]

    start_w = int((28-w)/2)
    end_w = start_w + w
    start_h = int((28-h)/2)
    end_h = start_h + h
    mnist_image[start_h:end_h,start_w:end_w] = image

    return mnist_image

def convert_to_image_mnist(image):
    """ The shortcut for _resize_image_mnist + _center_image_mnist
    Args:
        image : the image that are very closely to the number inside it
    Retunrs:
        mnist_image : 28x28 image with the number inside has been resized to fit 20x20 window
    """
    resized_image = _resize_image_mnist(image)

    mnist_image = _center_image_mnist(resized_image)
    return mnist_image

def _expand_image_mnist_size(images):
    """ Given an image contain number, expand its to fit the format of mnist input
        Mnist format: image size (28x28) in which each number fit in the 20x20 box
    Args:
        images (list) : a list of image with fitted number inside
                [(np.array)...]
    Return
        expanded_images (list) : a list of image has been expanded with size 28x28
                [(np.array)...]
    """
    expanded_images = []
    for image in images:
        # resize to image to has height or width = 20
        resized_image = _resize_image_mnist(image)

        expanded_image = _center_image_mnist(resized_image)

        expanded_images.append(expanded_image)

    return expanded_images

def detect_and_cut_number_look_once(image,number_of_box):
    """ Seperate the boxes given a row of linked box
    This function has the same action as the dect_and_cut but this
        function find the number by use findcontours once
    Args :
        image (np.array) : A 2-D image contains a row of boxes
        number_of_box (int) : The number of box inside image
    Returns:
        cut_images (list) : A list of 2-D box image
    """

    img = image.copy()

    if len(img.shape) != 2 :
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # get the position of rectangle inside image
    (x1,y1),(x2,y2) = detect_rectangle(img,number_of_box)
    if x1 is None:
        return []

    # crop to the rectangle row of that image
    cropped_img = ci.crop_image(img,(x1,y1),(x2,y2))
    draw.draw_rectangle(image,(x1,y2),(x2,y2))

    # remove horizontal
    # to get the better result while predict # took ~ 0.0005 s
    cropped_img = ip.remove_horizontal_lines(cropped_img)

    bboxes = _spot_number(cropped_img,number_of_box)

    number_images = ci.crop_images(cropped_img,bboxes)

    number_images_expanded = _expand_image_mnist_size(number_images)

    return number_images_expanded


def detect_and_cut_number(image,number_of_box):
    """ Given an image with a row of number inside, detect the number and cut 
        it to seperate image
    This function has the same action as the dect_and_cut_look_once but this
        function find the number by use findcontours multiple times
    Args :
        image (np.array) : A 2-D image contains a row of boxes
        number_of_box (int) : The number of box inside image
    Returns:
        cut_images (list) : A list of 2-D box image
    """

    img = image.copy()

    if len(img.shape) != 2 :
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # get the position of rectangle inside image
    (x1,y1),(x2,y2) = detect_rectangle(img,number_of_box)
    if x1 is None:
        return []

    # crop to the rectangle row of that image
    cropped_img = ci.crop_image(img,(x1,y1),(x2,y2))

    # divive the row number images to a list of single number images
    number_box_images = _divide_box(cropped_img,number_of_box)

    # remove the border of each image
    number_box_images_no_border = remove_borders(number_box_images)

    # find the number inside the single image
    number_images = _spot_number_batch(number_box_images_no_border)

    return number_images
