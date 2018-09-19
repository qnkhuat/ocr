import cv2
import numpy as np
from ocr.src.doc import filters

def resize_image(image,h_resized=1000):
    h,w = image.shape[:2]
    # resize to fixed a height but still preserve the scale
    w_resized = int(w*h_resized/h)
    image = cv2.resize(image, (w_resized, h_resized))
    return image , h_resized , w_resized

def adjust_ratio(image,ratio=70/99):
    '''
    adjust ratio while preserver original height
    '''
    h,w = image.shape[:2]
    adjusted_w = int(h*ratio)
    image = cv2.resize(image,(adjusted_w,h))

    return image

def do_hough_lines(img,rho = 1 , theta = np.pi/180, threshold = 135):
    lines = cv2.HoughLines(img,rho,theta,threshold)
    return lines


def convert_hough_lines(houghLines):
    '''
    Given the result of houghlines, find a straight line cut cross the image
    Args:
        houghLines: return of cv2.houghLines
    Returns:
        all_lines (list) : [(x1,y1),(x2,y2),...]
    '''
    all_lines  = []

    for line in houghLines:
        for rho,theta in line:
            # theta : angle of the line (radian)
            # y = -cos(theta)/sin(theta)*x + rho/sin(theta)
            a = np.cos(theta) #
            b = np.sin(theta)

            x0 = a*rho
            y0 = b*rho

            # ax0 + by0 = 0
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))

            # append to compute line_intersection
            all_lines.append([(x1,y1),(x2,y2)])
    return all_lines

def find_lines(image):
    '''
    Find lines in image
    '''

    edges_image = preprocess_image(image)
    
    hough_lines = do_hough_lines(edges_image)

    all_lines = convert_hough_lines(hough_lines)

    return all_lines


def find_intersect(all_lines,h,w):
    ''' Find interect points given lines '''
    all_intersect = []
    for idx,coords1 in enumerate(all_lines):
        for coords2 in all_lines[idx+1:]:
            if coords1 != coords2:
                x,y = line_intersection(coords1,coords2)
                # the point have to be insinde the image
                if  0 < x < w and 0 < y < h:
                    all_intersect.append([x,y])

    return all_intersect

def line_intersection(line1, line2):
    """ Give start and end points of 2 lines , find its intersecet
    Args:
        line1 (list) : [[x,y],[x,y]]
        line2 (list) : [[x,y],[x,y]]
    Returns:
        x,y (int) : the coordinate of intersect point
    """

    try :
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        div = det(xdiff, ydiff)

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
    except:
        return 0,0

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def remove_horizontal_lines(image,magnitude=.1):
    """ Remove any horizontal 
    Args:
        image (np.array) : image to remove
        magnitude (int) : the magnitute for determine line to remove, 
            the higher this numberis the is bigger the lines will be remove

    Returns:
        image (np.array) : horizontal,vertical lines removed image
    """

    img = image.copy()


    # Transform source image to gray if it is not already
    if len(img.shape) != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img 

    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv2.bitwise_not(gray)
    # adaptiveThreshold give a better result than threshold
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, -2)

    # [init]
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    # [init]

    # Specify size on verticales,horizontals line
    # NOTE : MODIFY the int under to get the different result
    cols = horizontal.shape[1]
    horizontal_size = int(cols / (1/magnitude))


    # START VERTICAL REMOVAL
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # Remove the horizontal line the original image
    horizontal = 1 - horizontal /255
    result = np.multiply(horizontal,gray)
    result = 255 - result

    
    # convert to the original dtype
    result = np.uint8(result)

    return result

def remove_vertical_lines(image,magnitude=.5):
    """ Remove any horizontal and vertical lines
    Args:
        image (np.array) : image to remove

    Returns:
        image (np.array) : horizontal,vertical lines removed image
        magnitude (int) : the magnitute for determine line to remove, 
            the higher this numberis the is bigger the lines will be remove

    """

    img = image.copy()


    # Transform source image to gray if it is not already
    if len(img.shape) != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img 

    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv2.bitwise_not(gray)
    # adaptiveThreshold give a better result than threshold
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, -2)

    # [init]
    # Create the images that will use to extract the horizontal and vertical lines
    vertical = np.copy(bw)
    # [init]

    # Specify size on verticales,horizontals line
    # NOTE : MODIFY the int under to get the different result
    rows = vertical.shape[0]
    verticalsize = int(rows / (1/magnitude))


    # START VERTICAL REMOVAL
    # Create structure element for extracting horizontal lines through morphology operations
        # result = 255 - result

    # START HORIZONTAL REMOVAL
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)


    # Remove the horizontal line the original image
    vertical = 1 - vertical /255
    result = np.multiply(vertical,gray)
    result = 255 - result

    # convert to the original dtype
    result = np.uint8(result)

    return result

def remove_horizontal_vertical_lines(image,magnitude_v =.5 ,magnitude_h=.1):
    """ Remove any horizontal and vertical lines
    Args:
        image (np.array) : image to remove
        magnitude (int) : the magnitute for determine line to remove, 
            the higher this numberis the is bigger the lines will be remove


    Returns:
        image (np.array) : horizontal,vertical lines removed image
    """

    img = image.copy()


    # Transform source image to gray if it is not already
    if len(img.shape) != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img 

    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv2.bitwise_not(gray)
    # adaptiveThreshold give a better result than threshold
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, -2)

    # [init]
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    # [init]

    # Specify size on verticales,horizontals line
    # NOTE : MODIFY the int under to get the different result
    cols = horizontal.shape[1]
    rows = vertical.shape[0]
    horizontal_size = int(cols / (1/magnitude_h))
    verticalsize = int(rows / (1/magnitude_v))


    # START VERTICAL REMOVAL
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # Remove the horizontal line the original image
    horizontal = 1 - horizontal /255
    result = np.multiply(horizontal,gray)
    # result = 255 - result

    # START HORIZONTAL REMOVAL
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)


    # Remove the horizontal line the original image
    vertical = 1 - vertical /255
    result1 = np.multiply(vertical,result)
    result1 = 255 - result1

    # convert to the original dtype
    result1 = np.uint8(result1)

    return result1

def preprocess_image(image):
    ''' Find edges '''
    # blur small details
    processed_image = cv2.GaussianBlur(image, (5, 5), 0,0)

    # Threshold to get black/white image
    ret , processed_image = cv2.threshold(processed_image,100,255,type=cv2.THRESH_OTSU)

    #filter noise
    processed_image = cv2.bilateralFilter(processed_image,11, 17, 17)

    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, kernel)
    
    # find the edges
    processed_image = cv2.Canny(processed_image, 100, 200)

    return processed_image

def find_document(image):
    ''' Given an image find the bbox of the document
    Args :
        image (np.array) : image to find
    Retunrs :
        bbox [tuple,tuple,tuple,tuple] : positions of 4 corners ul, ur , br , bl
    '''
    h,w = image.shape[:2]

    if len(image.shape) !=2:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # find all the cross lines in that image
    all_lines = find_lines(image)

    # Find all the intersect point
    intersects = find_intersect(all_lines,h,w)

    # Complicated filters to get the final bbox
    bbox = filters.filter_intersect_point(intersects,h,w)
    return bbox


