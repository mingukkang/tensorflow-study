import tensorflow as tf

import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, shape = [None,784], name = 'input_data')

Y = tf.placeholder(tf.float32, shape = [None,10], name = 'Label_data')

# for CNN, we should input 4rank data
X_img = tf.reshape(X,[-1,28,28,1])

# parameta
batch_size = 100
training_epoch = 15
keep_prob = tf.placeholder(tf.float32)
display_step = 1
logs_path = './logs1'

# first convolutional layers with 32 filters whose shape are [2,2,1]
# i will use Adam Optimizer and xavier initializer
with tf.name_scope('conv1'):
    conv_w1 = tf.get_variable("conv_w1", shape = [2,2,1,32], initializer = tf.contrib.layers.xavier_initializer())
    layer1 = tf.nn.conv2d(X_img, conv_w1, strides = [1,1,1,1], padding = "SAME")

# first relu layer
with tf.name_scope("relu1"):
    layer1 = tf.nn.relu(layer1)
# tensor layer1's shape = [-1,28,28,32]

# first max_pooling layer
# i will make out tensor layer1 dimidiate
with tf.name_scope("max_pool1"):
    layer1 = tf.nn.max_pool(layer1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

# Second convolutional layers with 64 filters whose shape are [2,2,1]
# i also use Adam Optimizer and xavier initializer

with tf.name_scope("conv2"):
    conv_w2 = tf.get_variable("conv_w2", shape = [2,2,32,64], initializer = tf.contrib.layers.xavier_initializer())
    layer2 = tf.nn.conv2d(layer1,conv_w2,strides = [1,2,2,1], padding ="SAME")
    # the layer2's shape is [-1,7,7,64]. you might see this, if you use print(layer2)

# second relu layer
with tf.name_scope("relu2"):
    tf.nn.relu(layer2)

# second max_pooling layer
# i will make layer2's shape same
with tf.name_scope("max_pool2"):
    layer2 = tf.nn.max_pool(layer2, ksize = [1,1,1,1], strides = [1,1,1,1], padding = "SAME")

print(layer2)

# layer2's shape is [-1,7,7,64] and for fully connected layer it is necessary to shape layer2 [?,7*7*64]
#7*7*64 = 3136
with tf.name_scope("Reshape"):
    layer2 = tf.reshape(layer2, [-1,3136])

# in fully connected layer, i will use AdamOptimizer, xavier initializer and dropout 
# first fully connected layer, 3136 number of input data, 1024 number of output data
with tf.name_scope("first_fully"):
    W1 = tf.get_variable("W1", shape = [3136,1024], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([1024]), name = 'Bias_1')
    L1 = tf.nn.relu(tf.matmul(layer2,W1)+b1)
    L1 = tf.nn.dropout(L1, keep_prob = keep_prob)

# second fully connected layer, 1024 number of input data, 1024 number of output data
with tf.name_scope("second_fully"):
    W2 = tf.get_variable("W2", shape = [1024,1024], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([1024]), name = 'Bias_2')
    L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
    L2 = tf.nn.dropout(L2, keep_prob = keep_prob)

# third fully connected layer, 1024 number of input data, 1024 number of output data
with tf.name_scope("third_fully"):
    W3 = tf.get_variable("W3", shape = [1024,1024], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([1024]), name = 'Bias_3')
    L3 = tf.nn.relu(tf.matmul(L2,W3)+b3)
    L3 = tf.nn.dropout(L3, keep_prob = keep_prob)

# forth fully connected layer, 1024 number of input data, 512 number of output data
with tf.name_scope("forth_fully"):
    W4 = tf.get_variable("W4", shape = [1024,512], initializer = tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([512]), name = 'Bias_4')
    L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)
    L4 = tf.nn.dropout(L4,keep_prob = keep_prob)
    
# final fully connected layer, 512 number of input data, 10 number of output data
with tf.name_scope("final_fully"):
    W5 = tf.get_variable("W5", shape = [512,10], initializer = tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([10]), name = 'Bias_5')
    hypothesis = tf.matmul(L4,W5)+b5
    
# cost function with cross_entropy tensorflow function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y))

# AdamOptimizer code
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

with tf.name_scope("Accuracy_part"):
    is_correct = tf.equal(tf.arg_max(hypothesis,1), tf.arg_max(Y,1))
    Accuracy = tf.reduce_mean(tf.cast(is_correct, dtype = tf.float32))


#create a summary to monitor cost value
tf.summary.scalar("cost",cost)
#create a summary to monitor Accuracy value
tf.summary.scalar("Accuracy", Accuracy)

#mege all summaries
summary = tf.summary.merge_all()

sess = tf.Session()
#initialize all variables
sess.run(tf.initialize_all_variables())

writer = tf.summary.FileWriter(logs_path)
writer.add_graph(sess.graph)

# training cycle
print("Learniong_start")
for epoch in range(training_epoch):
    cost_val = 0
    total_batch = int(mnist.train.num_examples/ batch_size) # define number of total batch
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # define training data set 
        feed_dict = {X:batch_xs, Y:batch_ys, keep_prob: 0.7}  # define feed_dict and i will use drop out rate 0.7
        c,s,_ = sess.run([cost,summary,optimizer], feed_dict = feed_dict)
        writer.add_summary(s, epoch*total_batch+i) # epoch*total_batch + i is global step
        cost_val += c/ total_batch 
    if (epoch +1) % display_step ==0:
        print("Epoch", '%04d' % (epoch+1), "cost = ", "{:.9f}".format(cost_val))
print("Learning_finish")

Accuracy_val = sess.run([Accuracy], feed_dict = {X:mnist.test.images, Y:mnist.test.labels, keep_prob:1})
print("Accuracy:", Accuracy_val)
