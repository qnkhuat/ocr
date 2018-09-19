import tensorflow as tf
import tensorflow.contrib.slim as slim

def CNN(inputs,is_training=True):
    '''
    CONV2D > POOL > CONV2D > POOL > FC > DROP >FC
    '''

    batch_norm_params = {
            'is_training':is_training,
            'decay':0.9,
            'updates_collections':None
            }

    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        normalizer_fn = slim.batch_norm,
                        normalizer_params = batch_norm_params):

        # inputs is in flat shape
        # slim 2d has default : padding:'SAME', activation : relu
        x = tf.reshape(inputs,[-1,28,28,1])

        net = slim.conv2d(x,32,[5,5],scope='conv1')

        net = slim.max_pool2d(net,[2,2],scope='pool1')

        net = slim.conv2d(net,64,[5,5],scope='conv2d')

        net = slim.max_pool2d(net, [2,2], scope='pool2')

        net = slim.conv2d(net,128,[5,5],scope='conv3d')

        net = slim.max_pool2d(net, [2,2], scope='pool3')

        # fixed bug the rank-4 output of logits
        net = tf.contrib.layers.flatten(net)

        net = slim.fully_connected(net,1024,scope='fc4')

        net = slim.dropout(net,is_training = is_training, scope='dropout4') # 0.5 by default

        outputs = tf.contrib.layers.fully_connected(net, 2, activation_fn=None,normalizer_fn = None, scope='fco')

    return outputs

