import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class Resnet(object):
    """
    Residual Network - 34

    """
    def __init__(
      self, input_layer,label_layer):

        self.input_layer = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32,name='input')
        self.label_layer = tf.placeholder(shape=[None],dtype=tf.int32)
        label_oh = slim.layers.one_hot_encoding(label_layer,10)
        #3×3, 15 conv
        layer = slim.conv2d(input_layer,16,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))

        #  conv1_x
        for j in range(5):
                layer = self.resUnit(layer,16,1,j)
            
        #  conv2_x
        for j in range(5):
            if j==0:
                layer = self.resUnit_pj(layer,32,2,j)
            else : 
                layer = self.resUnit2(layer,32,2,j)
            
        #  conv3_x
        for j in range(5):
            if j==0:
                layer = self.resUnit_pj(layer,64,3,j)
            else : 
                layer = self.resUnit(layer,64,3,j)
         

   
       #Global avg pool
       with tf.variable_scope('global_average_pooling'):
            layer= tf.reduce_mean(layer, axis=[1, 2])
        
        #fully connected layer 
        layer = slim.fully_connected(layer, 10, scope='fc')
        #softmax
        output = slim.layers.softmax(layer)

        loss = tf.reduce_mean(-tf.reduce_sum(label_oh * tf.log(output) + 1e-10, axis=[1]))
        trainer = tf.train.AdamOptimizer(learning_rate=0.001)
        update = trainer.minimize(loss)



            for j in range(units_between_stride):
                layer1 = resUnit(layer1,j + (i*units_between_stride))
            layer1 = slim.conv2d(layer1,64,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_s_'+str(i))
            
        top = slim.conv2d(layer1,10,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')

        output = slim.layers.softmax(slim.layers.flatten(top))

        loss = tf.reduce_mean(-tf.reduce_sum(label_oh * tf.log(output) + 1e-10, axis=[1]))
        trainer = tf.train.AdamOptimizer(learning_rate=0.001)
        update = trainer.minimize(loss)


    def resUnit(input_layer,filter_size,i,j):
        with tf.variable_scope("res_unit"+str(i)+"_"+str(j)):
            layer = slim.batch_norm(input_layer,activation_fn=None)
            layer = tf.nn.relu(layer)
            layer = slim.conv2d(layer,filter_size,[3,3],activation_fn=None)
            layer = slim.batch_norm(layer,activation_fn=None)
            layer = tf.nn.relu(layer)
            layer = slim.conv2d(layer,filter_size,[3,3],activation_fn=None)
            output = input_layer + layer
            return output

    def resUnit_pj(input_layer,filter_size,i):
        """shortcut projection with 1x1 convlution"""
        with tf.variable_scope("res_unit_pj"+str(i)):
            layer = slim.batch_norm(input_layer,activation_fn=None)
            layer = tf.nn.relu(layer)
            layer = slim.conv2d(layer,2*filter_size,[3,3],stride=[2,2],activation_fn=None)
            layer = slim.batch_norm(layer,activation_fn=None)
            layer = tf.nn.relu(layer)
            layer = slim.conv2d(layer,filter_size,[3,3],activation_fn=None)
            projected_input_layer=  slim.conv2d(layer, 2*filter_size,[1,1], stride=[2,2], activation_fn=None)
            output = projected_input_layer + layer
            return output



