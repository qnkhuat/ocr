# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time

from argparse import ArgumentParser

from ocr.src.digits import cnn_model


# refernce argument values
MODEL_DIRECTORY = "model"
TEST_BATCH_SIZE = 5000
ENSEMBLE = True

# build parser
def build_parser():
    parser = ArgumentParser()

    parser.add_argument('-m','--model-dir',
                        dest='model_directory', help='directory where model to be tested is stored',
                        metavar='MODEL_DIRECTORY', required=False,default='model/model01_99.61')
    parser.add_argument('-i','--image',dest='image',default='test_image/1.png',help='Path to test image')
    return parser
# test with test data given by mnist_data.py


def convert_img(image):
    # change background to be black
    if len(image.shape) >2:
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = cv2.resize(255 - image, (28, 28))
    image = image.flatten()
    
    # has to be in range [-0.5,0.5]
    image = (image /255.0) -0.5
    image = np.expand_dims(image,axis=0)
    return image

def softmax(scores):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(scores - np.max(scores))
    return e_x / e_x.sum(axis=0)


def predict(model_directory,image_dir):
    tf.reset_default_graph()
    # prepare data
    img = cv2.imread(image_dir,0)
    x_input = convert_img(img)

    # Import data
    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])  # answer
    y = cnn_model.CNN(x, is_training=is_training)

    # Add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # Restore variables from disk
    saver = tf.train.Saver()

    saver.restore(sess, model_directory)

    y_final = sess.run(y, feed_dict={x: x_input, is_training: False})
    y_idx = np.argmax(y_final[0])
    y_softmax = softmax(y_final[0])
    
    return y_idx,y_softmax[y_idx]

def predict_image(img,model_path='model/model01_99.61/model.ckpt'):
    x_input = convert_img(img)
    tf.reset_default_graph()

    # Import data
    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])  # answer
    y = cnn_model.CNN(x, is_training=is_training)

    # Add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # Restore variables from disk
    saver = tf.train.Saver()

    saver.restore(sess, model_path)

    y_final = sess.run(y, feed_dict={x: x_input, is_training: False})
    y_idx = np.argmax(y_final[0])
    y_softmax = softmax(y_final[0])
    
    return y_idx,y_softmax[y_idx]


if __name__ == '__main__':
    # Parse argument
    parser = build_parser()
    args = parser.parse_args()
    model_directory = args.model_directory
    input_image_path = args.image
    for input_image in os.listdir(input_image_path):
        if '.jpg' in input_image or '.png' in input_image:
            input_image = os.path.join(input_image_path,input_image)
            start = time.time()
            y,y_score = predict(model_directory+'/model.ckpt',input_image)
            print('Inference time',time.time()-start)
            img = cv2.imread(input_image)
            plt.imshow(img)
            title = str(y) + ' - ' + str(y_score)
            plt.title(title)
            plt.show()
