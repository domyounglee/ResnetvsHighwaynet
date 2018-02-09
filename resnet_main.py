#Load necessary libraries
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from plain_model import Resnet
import time
import os

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


#load data 
cifar = unpickle('./cifar10/data_batch_1')
cifarT = unpickle('./cifar10/test_batch')



 #set layers of your network 
total_layers = 25
units_between_stride = total_layers / 5




init = tf.global_variables_initializer()

#hyperparameters
batch_size = 128
currentCifar = 1
total_steps = 20000
num_checkpoints=1
    
aT = []


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)


    with sess.as_default():
        model=Resnet()
        

         # Define Training procedure
        timestamp = str(int(time.time()))
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(0.0001)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        # make directory
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))


        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("acc", model.accuracy)
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        #iterate mini-batch
        i = 0
        draw = range(10000)
        while i < total_steps:
            if i % (10000/batch_size) != 0:
                batch_index = np.random.choice(draw,size=batch_size,replace=False)
            else:
                draw = range(10000)
                if currentCifar == 5:
                    currentCifar = 1
                    print "Switched CIFAR set to " + str(currentCifar)
                else:
                    currentCifar = currentCifar + 1
                    print "Switched CIFAR set to " + str(currentCifar)
                cifar = unpickle('./cifar10/data_batch_'+str(currentCifar))
                batch_index = np.random.choice(draw,size=batch_size,replace=False)
           

            x = cifar['data'][batch_index]
            x = np.reshape(x,[batch_size,32,32,3],order='F')
            x = (x/256.0)
            x = (x - np.mean(x,axis=0)) / np.std(x,axis=0)
            y = np.reshape(  np.array(cifar['labels'])[batch_index],[batch_size,1]  ) 
            
            feed_dict = {
              model.input_layer: x,
              model.label_layer: np.hstack(y)
            }
            #train
            _,step,summaries,lossA,accuracy = sess.run([train_op, global_step, train_summary_op,model.loss,model.accuracy],feed_dict)

            #write summary
            train_summary_writer.add_summary(summaries, step)
            current_step = tf.train.global_step(sess, global_step)


            if i % 10 == 0: print "Step: " + str(i) + " Loss: " + str(lossA) + " Accuracy: " + str(accuracy)
            
            if i % 100 == 0: 
                point = np.random.randint(0,10000-500)
                xT = cifarT['data'][point:point+500]
                xT = np.reshape(xT,[500,32,32,3],order='F')
                xT = (xT/256.0)
                xT = (xT - np.mean(xT,axis=0)) / np.std(xT,axis=0)
                yT = np.reshape(np.array(cifarT['labels'])[point:point+500],[500])

                feed_dict = {
                  model.input_layer: xT,
                  model.label_layer: yT
                }
                aT.append(accuracy)
                accuracy = sess.run([model.accuracy],feed_dict)
                print "Test set accuracy: " + str(accuracy)
            




            i+= 1


        print(np.max(aT))