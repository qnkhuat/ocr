import sys
sys.path.append('../../..')

from ocr.src.cut import cut_image as ci 
from ocr.src.doc.image_process import remove_horizontal_vertical_lines as remove
from ocr.src import draw
from ocr.src import helpers
import os
import cv2
import matplotlib.pyplot as plt

TEST_FOLDER = '../../../resources/images/test/digits'
NUMBER_OF_BOX = 6

for filename in os.listdir(TEST_FOLDER):
    file_path = os.path.join(TEST_FOLDER,filename)
    if not helpers.verify_filename(file_path) :
        continue
    if filename!= 'img1.jpg':
        #continue 
        pass

    img = cv2.imread(file_path,0)
    # img = remove(img)

    # cut the rectangle
    (x1,y1),(x2,y2) = ci._get_rectangle(img,NUMBER_OF_BOX)

    cropped_img = ci.crop_image(img,(x1,y1),(x2,y2))
    
    bboxes= ci._spot_number(cropped_img,NUMBER_OF_BOX)
    for bbox in bboxes:
        draw.draw_rectangle(cropped_img,bbox[0],bbox[1],thickness=1)
    plt.title(filename)
    plt.imshow(cropped_img)
    plt.show()
