import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from helpers import implt, resize, ratio
import argparse
from PIL import ImageGrab
import time
import os
from utils import *

def edges_det(img, minVal, maxVal):
    """ Preprocessing (gray, thresh, filter, border) + Canny edge detection """
    img = cv2.cvtColor(resize(img), cv2.COLOR_BGR2GRAY)

    # Applying blur and threshold
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)

    # Median blur replace center pixel by median of pixels under kelner
    # => removes thin details
    img = cv2.medianBlur(img, 11)

    # Add black border - detection of border touching pages
    # Contour can't touch side of image
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return cv2.Canny(img, minVal, maxVal)


def four_corners_sort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right"""
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


def contour_offset(cnt, offset):
    """ Offset contour because of 5px border """
    cnt += offset
    cnt[cnt < 0] = 0
    return cnt


def find_page_contours(edges):
    """ Finding corner points of page contour """
    # Getting contours
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Finding biggest rectangle otherwise return original corners
    height = edges.shape[0]
    width = edges.shape[1]
    MIN_COUNTOUR_AREA = height * width * 0.5
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

    maxArea = MIN_COUNTOUR_AREA
    pageContour = np.array([[0, 0],
                            [0, height-5],
                            [width-5, height-5],
                            [width-5, 0]])
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        # Page has 4 corners and it is convex
        if (len(approx) == 4 and
                cv2.isContourConvex(approx) and
                maxArea < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):

            maxArea = cv2.contourArea(approx)
            pageContour = approx[:, 0]

    # Sort corners and offset them
    pageContour = four_corners_sort(pageContour)
    return contour_offset(pageContour, (-5, -5))

def persp_image_transform(img, sPoints):
    """ Transform perspective from start points to target points """
    # Euclidean distance - calculate maximum height and width
    height = max(np.linalg.norm(sPoints[0] - sPoints[1]),
                 np.linalg.norm(sPoints[2] - sPoints[3]))
    width = max(np.linalg.norm(sPoints[1] - sPoints[2]),
                 np.linalg.norm(sPoints[3] - sPoints[0]))

    # Create target points
    tPoints = np.array([[0, 0],
                        [0, height],
                        [width, height],
                        [width, 0]], np.float32)

    # getPerspectiveTransform() needs float32
    if sPoints.dtype != np.float32:
        sPoints = sPoints.astype(np.float32)

    M = cv2.getPerspectiveTransform(sPoints, tPoints)
    return cv2.warpPerspective(img, M, (int(width), int(height)))


def find_doc(img,show=False):
    '''
    function to find the document given an image
    '''
    # output a gray image
    image_edges = edges_det(img, 200, 250)
    closed_edges = cv2.morphologyEx(image_edges, cv2.MORPH_CLOSE, np.ones((5, 11)))

    # find contour
    page_contour = find_page_contours(image_edges)#

    # draw the contours
    image_drawed = cv2.drawContours(resize(img), [page_contour], -1, (0, 255, 0), 3)

    # rescale the contour
    page_contour = page_contour.dot(ratio(img))

    # Transform perspective
    new_image = persp_image_transform(img, page_contour)

    if show:
        implt(image_drawed,'Result')
    return image_drawed



def main():

    args = setup_arg()
    test_folder(args.images,find_doc)





if __name__ == '__main__':
    main()
