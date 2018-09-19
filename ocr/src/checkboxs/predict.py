import numpy as np
import tensorflow as tf 

from ocr.src.checkboxs import utils
from ocr.src.checkboxs import model

# Do not predict one image at a time. This will take a really long time
def predict_batch(images_list,model_path):
    """ Predict checkbox from a list of images"""

    class_name =['No','Yes']
    images = [utils.normalize(image) for image in images_list ]

    assert isinstance(images,list) , "Input has to be a list of images"
    images = np.asarray(images)

    x = tf.placeholder(shape=[None,28,28],dtype = tf.float32)
    is_training = tf.placeholder(tf.bool,name='MODE')

    y = model.CNN(x,is_training=False)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init,feed_dict = {is_training: False})

        try:
            saver.restore(sess,model_path)
            print('Loaded checkpoint')
        except:
            raise Exception('No checkpoint found')

        responses = sess.run(y,feed_dict = {x:images})
        responses = np.argmax(responses,axis=1)

    labels = [class_name[response] for response in responses ]
    return labels 


