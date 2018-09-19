import numpy as np
import cv2 
from ocr.src.doc.image_process import adjust_ratio

def _compute_size_persp_transform(sPoints):
    # Euclidean distance - calculate maximum height and width
    width = max(np.linalg.norm(sPoints[0] - sPoints[1]),
                 np.linalg.norm(sPoints[2] - sPoints[3]))
    height = max(np.linalg.norm(sPoints[1] - sPoints[2]),
                 np.linalg.norm(sPoints[3] - sPoints[0]))
    return width,height

def _compute_target_point(width,height):
    tPoints = np.array([[0, 0],
                        [width, 0],
                        [width, height],
                        [0, height],
                        ], np.float32)

    return tPoints

def _do_transform(img,sPoints,tPoints,width,height):
    M = cv2.getPerspectiveTransform(sPoints, tPoints)
    return cv2.warpPerspective(img, M, (int(width), int(height)))

def _convert_and_resize_sPoints(img,sPoints):
    sPoints = np.asarray(sPoints)

    #rescale to the original image ratio
    sPoints = sPoints.dot(img.shape[0]/1000) # ratio h/1000

    # getPerspectiveTransform() needs float32
    if sPoints.dtype != np.float32:
        sPoints = sPoints.astype(np.float32)
    return sPoints

def persp_image_transform(img, sPoints):
    """ Transform perspective from start points to target points """

    # resize to fit original image size
    sPoints = _convert_and_resize_sPoints(img,sPoints) 

    width , height = _compute_size_persp_transform(sPoints)
    # Create target points
    tPoints = _compute_target_point(width,height)
    
    transformed_img = _do_transform(img,sPoints,tPoints,width,height)
    
    return transformed_img

def remove_noise(img,a,g):
    processed_image = cv2.addWeighted(img,a,np.zeros(img.shape,img.dtype),0,g)
    return processed_image 

def whiten_background(img):
    # copy from : https://stackoverflow.com/questions/46522056/how-to-whiten-background-and-blaken-grid-in-a-same-image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilated = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    processed_image = 255 - cv2.subtract(dilated, img)

    # do medianBlur
    median = cv2.medianBlur(dilated, 5)
    processed_image = 255 - cv2.subtract(median, img)

    ## do normalize
    processed_image = cv2.normalize(processed_image,None, 0, 255, cv2.NORM_MINMAX )

    return processed_image
 
def enhance_text(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

def whiten_and_remove_noise(img):
    """
    Args:
        img (np.array) : Image
    Return:
        img (np.array) : Processed image
    """
    processed_image = whiten_background(img)

    processed_image = remove_noise(processed_image,1,13)

    # make the text bolder
    processed_image = enhance_text(processed_image, gamma=0.5)

    return processed_image

def crop_and_enhance(image,bbox,scale=70/99):
    '''
    Args:
        image (np.array) 
        scale: the scale of your output image. A4: 70/99 
    '''

    cropped_img = persp_image_transform(image,bbox) 
    adjusted_img = adjust_ratio(cropped_img,scale)
    enhanced_img = whiten_and_remove_noise(adjusted_img)

    return enhanced_img
