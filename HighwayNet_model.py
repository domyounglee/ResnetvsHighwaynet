import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim



class Resnet(object):
    """
    Residual Network - 34
    trained on CIFAR-10
    """
    def __init__(self):

        self.input_layer = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32,name='input')
        self.label_layer = tf.placeholder(shape=[None],dtype=tf.int64)
        label_oh = slim.layers.one_hot_encoding(self.label_layer,10)
        layer = slim.conv2d(self.input_layer,16,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))
        

        
        for i in range(5):
            for j in range(5):
                layer = self.HighwayUnit(layer,16,i,j)
            layer = slim.conv2d(layer,16,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_s_'+str(i))
        """

        #  conv1_x
        for j in range(5):
                layer = self.HighwayUnit(layer,16,1,j)
        
        print("conv1_x:",layer)
        #  conv2_x
        for j in range(5):
            if j==0:
                layer = self.HighwayUnit_pj(layer,32,2)
            else : 
                layer = self.HighwayUnit(layer,32,2,j)
        
        print("conv2_x:",layer)
        #  conv3_x
        for j in range(5):
            if j==0:
                layer = self.HighwayUnit_pj(layer,64,3)
            else : 
                layer = self.HighwayUnit(layer,64,3,j)
         
        print("conv3_x:",layer)
        """
       #Global avg pool
        with tf.name_scope('global_average_pooling'):
            layer= tf.reduce_mean(layer, axis=[1, 2])
        
        #fully connected layer 
        layer = slim.fully_connected(layer, 10, scope='fc')

        #softmax
        output = slim.layers.softmax(layer, scope='softmax')
        

        #loss
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(-tf.reduce_sum(label_oh * tf.log(output) + 1e-10, axis=[1]))
        


      
        # Accuracy
        with tf.name_scope("accuracy"):
            predictions = tf.argmax(output, 1, name="predictions")
            correct_predictions = tf.equal(predictions,self.label_layer)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")





    def HighwayUnit(self,input_layer,filter_size,i,j):
        with tf.variable_scope("highway_unit"+str(i)+"_"+str(j)):
            layer = slim.batch_norm(input_layer,activation_fn=None)
            H = slim.conv2d(layer,filter_size,[3,3])

            T = slim.conv2d(layer,filter_size,[3,3], #We initialize with a negative bias to push the network to use the skip connection
            biases_initializer=tf.constant_initializer(-1.0),activation_fn=tf.nn.sigmoid)
            output = H*T + layer*(1.0-T)
            return output



    def HighwayUnit_pj(self,input_layer,filter_size,i):
        """shortcut projection with 1x1 convlution"""
        with tf.variable_scope("highway_unit_pj"+str(i)):
            layer = slim.batch_norm(input_layer,activation_fn=None)
            H = slim.conv2d(layer,filter_size,[3,3],stride=[2,2])
            T = slim.conv2d(layer,filter_size,[3,3], #We initialize with a negative bias to push the network to use the skip connection
            stride=[2,2],biases_initializer=tf.constant_initializer(-1.0),activation_fn=tf.nn.sigmoid)
            projected_input_layer=  slim.conv2d(layer, filter_size,[1,1], stride=[2,2], activation_fn=None)
            output = H*T + projected_input_layer*(1.0-T)
            return output



