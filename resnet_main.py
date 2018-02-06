#Load necessary libraries
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import input_data




#load data 
cifar = unpickle('./cifar10/data_batch_1')
cifarT = unpickle('./cifar10/test_batch')



 #set layers of your network 
total_layers = 25
units_between_stride = total_layers / 5




init = tf.global_variables_initializer()

#hyperparameters
batch_size = 64
currentCifar = 1
total_steps = 20000
l = []
a = []
aT = []


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)


    with sess.as_default():

        resnet=Resnet()
        sess.run(init)

         # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(0.025)
        grads_and_vars = optimizer.compute_gradients(rnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))


        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", rnn.loss)
        acc_summary = tf.summary.scalar("acc", rnn.accuracy)
        train_summary_op = tf.summary.merge([loss_summary, acc_summary,grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)


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
            y = np.reshape(np.array(cifar['labels'])[batch_index],[batch_size,1])
            #train
            _,lossA,yP,LO = sess.run([update,loss,output,label_oh],feed_dict={input_layer:x,label_layer:np.hstack(y)})
            #accuracy 
            accuracy = np.sum(np.equal(np.hstack(y),np.argmax(yP,1)))/float(len(y))
            l.append(lossA)
            a.append(accuracy)
            if i % 10 == 0: print "Step: " + str(i) + " Loss: " + str(lossA) + " Accuracy: " + str(accuracy)
            if i % 100 == 0: 
                point = np.random.randint(0,10000-500)
                xT = cifarT['data'][point:point+500]
                xT = np.reshape(xT,[500,32,32,3],order='F')
                xT = (xT/256.0)
                xT = (xT - np.mean(xT,axis=0)) / np.std(xT,axis=0)
                yT = np.reshape(np.array(cifarT['labels'])[point:point+500],[500])
                lossT,yP = sess.run([loss,output],feed_dict={input_layer:xT,label_layer:yT})
                accuracy = np.sum(np.equal(yT,np.argmax(yP,1)))/float(len(yT))
                aT.append(accuracy)
                print "Test set accuracy: " + str(accuracy)
            i+= 1

