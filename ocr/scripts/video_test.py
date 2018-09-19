import numpy as np
import argparse
import time
import cv2
import time
import imutils
from detect import detect

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--video',dest='video',help='Function to use: h/c',default='../../resources/test.m4v')
    args = parser.parse_args()
    return args


def main():
    '''
    macbook pro 13" 2016: 16.4 fps
    '''
    args = get_args()
    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    fps_holder = []
    # compute fps
    num_frames = 0
    start_process = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret ==True:
            num_frames+=1
            start_time = time.time()
            frame = imutils.resize(frame, width=1000)

            # magic happens here
            result,cropped_img = detect(frame)
            cv2.imshow('Result',result)
            print('Loop in :',time.time() - start_time)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            break


    fps = num_frames/(time.time() - start_process)
    print('Average fps:',fps)
    cap.release()

if __name__ == '__main__':
	main()
