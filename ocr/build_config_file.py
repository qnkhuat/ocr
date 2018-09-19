import sys
sys.path.append('..')
import os
import json
import cv2
import argparse

from ocr.src.cut import build_config as bc

"""
    There are 2 methods of finding anchor words:
        -   static : using fixed ratio to locate
        -   Dynamic : send images to google vision to locate the words | This methods is take more times
"""


def build_config(output_path=None):
    img = cv2.imread('../resources/images/result/crop/demo.jpg')
    image_shape = img.shape

    data = [
        bc.process_config_data(
            target_name = 'ILLITERATE',
            target_bbox = '172,55,256,135',
            image_shape = image_shape,
            checkbox=True,
        )
        ,
        bc.process_config_data(
            target_name = 'VISUALITY',
            target_bbox = '158,154,248,238',
            image_shape = image_shape,
            checkbox=True,
        )
        ,
        bc.process_config_data(
            target_name = 'STAFF',
            target_bbox = '160,254,259,344',
            image_shape = image_shape,
            checkbox=True,
        )
        ,
        bc.process_config_data(
            target_name = 'PARDANSANIN',
            target_bbox = '173,373,262,457',
            image_shape = image_shape,
            checkbox=True,
        )
        ,
        bc.process_config_data(
            target_name = 'INTERNET',
            target_bbox = '164,468,260,553',
            image_shape = image_shape,
            checkbox=True,
        )
        ,
        bc.process_config_data(
            target_name = 'UNDERTAKINGS',
            target_bbox = '158,581,258,669',
            checkbox=True,
            image_shape = image_shape
        )
        ,
        bc.process_config_data(
            target_name = 'primary applicant',
            target_bbox = '364,1581,447,1649',
            image_shape = image_shape,
            checkbox=True
        )
        ,
        bc.process_config_data(
            target_name = 'PF number',
            target_bbox = '1143,266,1409,342',
            image_shape = image_shape,
            number_of_box=6,
        )
        ,
        bc.process_config_data(
            target_name = 'PF No',
            target_bbox = '988,1569,1271,1657',
            image_shape = image_shape,
            number_of_box=6,
        )
        ,
        bc.process_config_data(
            target_name = 'PA/PF NO',
            target_bbox = '1263,1736,1515,1797',
            image_shape = image_shape,
            number_of_box=6,
        ),
        bc.process_config_data(
            target_name = 'PA/PF NO',
            target_bbox = '1265,1892,1521,1958',
            image_shape = image_shape,
            number_of_box=6,
        )
        ]
    
    if None in data:
        return

    # save
    if output_path is None:
        output_path = os.path.join(HOME,'resources/configs/config.conf')

    with open(output_path,'w') as f:
        json.dump(data,f,indent=4)

    print('Done')
    print('Config saved at',output_path)

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',help='path to where to store config file')
    args = parser.parse_args()
    return args


def main():
    args = setup_args()

    build_config(args.o)


if __name__ == '__main__':
    main()
