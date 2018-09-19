import cv2
import numpy as np


def draw_point(img,x,y,radius=5,color=(255,0,0)):
    cv2.circle(img,(int(x),int(y)),radius,color,5)

def put_text(img,text,position,font = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale= 1,color=(255,0,0),thickness=1,line_type= cv2.LINE_AA ):
    '''
    position : bottom left 
    '''
    text = str(text)
    cv2.putText(img,text,position,font,font_scale,color,thickness,line_type)

def draw_rectangle(img,pos1,pos2,color=(0,255,0),thickness=5):
    cv2.rectangle(img,pos1,pos2,color,thickness)


def draw_intersect_dict(img,all_intersect):
    ''' Draw intersect given a dict of poitns '''
    for key,value in all_intersect.items():
        for intersect in value:
            x,y=intersect
            draw_point(img,x,y)

def draw_intersect_list(img,all_intersect):
    for intersect in all_intersect:
        x,y=intersect
        draw_point(img,x,y)

def draw_lines(img,all_lines,color=[0, 255, 0], thickness=2):
    ''' Quick function to draw many lines '''
    for line in all_lines:
        cv2.line(img,line[0],line[1],color,thickness)

def draw_page(img,corners,color=(0,255,0),thickness=5):
    ''' Draw lines surround the page '''

    # transform corners to the expected form of cv2.drawContours
    # [[(x,y)],[(x,y)]...]
    corners = np.expand_dims(corners,axis=1)
    corners = np.asarray(corners)
    corners = corners.astype(int)

    img = cv2.drawContours(img,[corners],-1,color,thickness)
    return img


def draw_page_dict(img,corners,color=(0,0,255),thickness=5):
    '''
    Function to draw the rectangle of document given a dict of corners
    corners = {
        1:[],# upper left
        2:[],# upper right
        3:[],# bottom right
        4:[],# bottom left
    }
    '''

    for points in list(itertools.product(corners[1],corners[2],corners[3],corners[4])) :

        points = np.expand_dims(points,axis=1)
        points = np.asarray(points)
        points = points.astype(int)

        img = cv2.drawContours(img,[points],-1,color,thickness)


def draw_contours(image,cnts,thickness=1,color=(0,255,0),show=False):
    cv2.drawContours(image,cnts,-1,color,thickness)
    if show:
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.title('draw_contours')
        plt.show()
