# ---------------------------------------------------------------------------------------------- 
# the purpose of this code is to classify mnist data with linear regression.
# when i write this code, i refered to https://github.com/Hvass-Labs/TensorFlow-Tutorials
# I am not so good at English. So please be good to understand my awkward English. 
# ---------------------------------------------------------------------------------------------- 
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

# import mnist data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("MNIST_data/", one_hot=True)

# translate input data from onehot form to class number.
data_cls = np.array([label.argmax() for label in data.test.labels])

# parameta
img_size = 28
img_shape = (28,28,1)
img_shape_plot =(28,28)
img_flat = 28*28
num_classes = 10

# helper function to plot mnist images and indicate its true labels and predicted labels
def plot_img(images,data_true,data_pred = None):
    # make debug code 
    assert len(images)== len(data_true) == 9
    
    # make 3*3 number of plots
    fig,axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace =0.3, wspace =0.3)
    
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape_plot),cmap ='binary')
        
        if data_pred is None:
            xlabel = "True: {0}".format(data_true[i])
        else:
            xlabel = "True:{0},  ped:{1}".format(data_true[i], data_pred[i])
        
        ax.set_xlabel(xlabel)
        
        ax.set_xticks([])
        ax.set_yticks([])

X = tf.placeholder(tf.float32, shape = [None,img_flat], name = "INPUT")

# we should input one_hot shape data into Y
Y = tf.placeholder(tf.float32, shape = [None,num_classes], name ="OUTPUT")

# make weight and bias for linear regression
# we will use xavier initializer.
W = tf.get_variable("W", shape =[img_flat,num_classes], initializer = 
                   tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros(shape =[num_classes]), name ="Bias")

# hypothesis
hypothesis = tf.matmul(X,W)+b

# softmax 
softmax = tf.nn.softmax(hypothesis)

prediction = tf.argmax(softmax,1)
pred_one_hot = tf.one_hot(prediction,num_classes)

# cross entropy 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y))

# optimize cost function with learning rate 0.01
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)

# calculate Accuracy
is_correct = tf.equal(prediction,tf.argmax(Y,1))
Accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# make Session and initialize all variables
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# for calculating more comfortable, we use batches and epoches
batch_size = 100
epoch = 15

# training
print("------------------------------------------------------------------")
print("learning_start")
for epoch in range(epoch):
    cost_val = 0
    total_batch = int(data.train.num_examples/ batch_size)
    
    for a in range(total_batch):
        batch_xs,batch_ys = data.train.next_batch(batch_size)
        c,_ = sess.run([cost,optimizer], feed_dict = {X:batch_xs, Y:batch_ys})
        cost_val += c/total_batch
    print("epoch:", epoch,"cost_val:",cost_val)
print("learning_end")
print("------------------------------------------------------------------")
Accuracy_val = sess.run([Accuracy], feed_dict ={X:data.test.images, Y:data.test.labels})
print("Accuracy :",Accuracy_val)

# create function which searches errors and plots it
def plot_errors():

    correct, cls_pred = sess.run([is_correct, prediction],
                                    feed_dict ={X:data.test.images, Y:data.test.labels})

    incorrect = (correct == False)
    
    q = data.test.images[incorrect]
    w = data_cls[incorrect]
    e = cls_pred[incorrect]
    q = q[0:9]
    w = w[0:9]
    e = e[0:9]
    
    plot_img(images = q, data_true =w, data_pred = e)
    plt.show()

plot_errors()