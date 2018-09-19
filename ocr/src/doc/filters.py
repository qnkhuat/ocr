import numpy as np
import itertools
import cv2

def corners_spot(all_points,h,w):
    '''
    divide image into a 4x4 grid.
    filter the points at the corners only. denoted corners as 1,2,3,4
    '''
    corners = {
        1:[],# upper left
        2:[],# upper right
        3:[],# bottom right
        4:[],# bottom left
    }

    # rate for corners [left,right]
    rate = [1/3,2/3]

    for point in all_points:
        # left
        if point[0] < w*rate[0] :

            # upper left
            if point[1] < h*rate[0]:
                if point not in corners[1]:
                    corners[1].append(point)

            # bottom left
            elif point[1] > h*rate[1]:

                if point not in corners[4]:
                    corners[4].append(point)

        #right
        elif point[0] > w*rate[1]:

            # upper right
            if point[1] < h*rate[0]:
                if point not in corners[2]:
                    corners[2].append(point)

            # bottom right
            elif point[1] > h*rate[1]:
                if point not in corners[3]:
                    corners[3].append(point)

    return corners

def compute_area(corners):
    '''
    corners ((x,y) , (x,y))
    '''
    corners = np.asarray(corners)
    corners = corners.astype(int)
    return cv2.contourArea(corners)

def average_points(all_points):
    '''
    Compute an anverage point

    all_points [(x,y),(x,y)...]
    '''

    # convert to array to be able to slice matrix
    all_points = np.asarray(all_points)
    return (np.mean(all_points[:,0]),np.mean(all_points[:,1])) # x,y

def filter_near_points(corners,distance_threshold=20):
    ''' Spot then averange any near points to 1 point'''
    ''' For those who's trying to read this, wish you luck :) '''

    corners_filtered = {
        1:[],# upper left
        2:[],# upper right
        3:[],# bottom right
        4:[],# bottom left
    }

    for key,corner in corners.items():
        # store image has been take to compute average
        processed_idx = []
        for idx1,point1 in enumerate(corner):
            near_corners =[]

            if idx1 not in processed_idx :
                near_corners.append(point1)
                processed_idx.append(idx1)

            for idx2,point2 in enumerate(corner):
                # avoid duplicate
                if idx2 <= idx1:
                    continue

                dist = np.linalg.norm(np.asarray(point1)-np.asarray(point2))

                if dist < distance_threshold and idx2 not in processed_idx:
                    near_corners.append(point2)
                    processed_idx.append(idx2)

            if len(near_corners) >0 :
                corners_filtered[key].append(average_points(near_corners))

    # Congrats, but if you think you understood, You don't :)
    return corners_filtered


def find_highest_area(corners):
    # TODO OPTIMIZE: This is too naive, Sometimes it will be a counter-attack
    ''' Given a set of corners, find the box has largest area '''

    try:
        # shuffle points of all the corners
        shuffled_corners = list(itertools.product(corners[1],corners[2],corners[3],corners[4]))
        max_area = 0
        for box in shuffled_corners:
            area = compute_area(box)
            if area > max_area:
                max_area = area
                chose_box = box
            
        # convert all float point to int
        chose_box = [tuple(map(int,corner)) for corner in chose_box]
        return chose_box
    
    except :
        # if not enough corners
        return None

def filter_intersect_point(all_points,h,w):
    '''
    function to reduce the number of intersect point
    Args:
        all_points: all the intersect poitns [[x,y]...]
        h : height of the original image
        w : width of the original image
    Returns :
        cornres[tuple,tuple,tuple,tuple] : positions of 4 corners ul, ur , br , bl
    '''

    corners = corners_spot(all_points,h,w) # input a list,output a dict
    corners = filter_near_points(corners) # output a dict
    corners = find_highest_area(corners) # output a list

    return corners

