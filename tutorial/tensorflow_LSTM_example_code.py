# ----------------------------------------------------------------------------------------------
# the purpose of this code is to make algorism where if you input "hihell" to your computer, 
# the computer will give you "ihello" output
# when i write this code, i refered to the youtube lecture which link
# "https://www.youtube.com/watch?v=vwjt1ZE5-K4&index=45&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm"
# I am not so good at English. So please be good to understand my awkward English.
# instead of using "hihello", i use lyrics(see you again).
# ----------------------------------------------------------------------------------------------

import tensorflow as tf

import numpy as np

# Music see you again intro
see_you_again = "It.s been a long day without you,my friend and I.ll tell you all about it  when i see you again We.ve come a long way from where we began Oh, I.ll tell you all about it when I see you again When I see you again"

# extract characters from see_you_again. you may identify this by using print(idx2char)
idx2char = list(set(see_you_again))

# make dictionary
char2idx ={c: i for i,c in enumerate(idx2char)}

# identify how many alphabats exist. you can see there are 210 numbers of alphabats.
A = len(see_you_again)

# parameta and logs_path for tensorboard
batch_size = 10 # i just select one number which perfectly devides 210.

# the number of batches with strides 1 is (210-10 +1)/1. but i don't use the last one. thus the number of batches is
# (210-10) /1  = 200 
# you may be confused by this. if you do, you can understand by studying below youtube video which prof sungkim from Hkust makes 
# https://www.youtube.com/watch?v=39_P23TqUnw&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=43
total_batch = len(see_you_again) - batch_size 

sequence_length = batch_size # it is same with batch_size

input_size = len(idx2char) # input_size =26, and it is same with output_size

# it will make automatically folders named "graph" and "RNN_exam"
logs_path = "./graph/RNN_exam"

# before make X_data, Y_data, we should make see_you_again alphabats to index
data_matrix = [char2idx[c] for c in see_you_again]

# make batch data saver
X_data = []
Y_data = []

# make X_data and Y_data
with tf.name_scope("make_batch_saver") as scope:
    for i in range(total_batch):
        onebatch_X = data_matrix[i:i+ batch_size]
        onebatch_Y = data_matrix[i+1:i+batch_size+1]
        X_data.append(onebatch_X)
        Y_data.append(onebatch_Y)

X = tf.placeholder(tf.int32, shape = [None,sequence_length], name ="X")

Y = tf.placeholder(tf.int64, shape = [None,sequence_length], name ="Y")

x_one_hot = tf.one_hot(X,input_size)

# make cell and calculate outputs 
with tf.name_scope("RNN_cell") as scope:
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = input_size ,state_is_tuple = "True")
    cell = tf.nn.rnn_cell.MultiRNNCell([cell]*2, state_is_tuple ="True")
    initial_state = cell.zero_state(total_batch, tf.float32)
    outputs,_states = tf.nn.dynamic_rnn(cell,x_one_hot,initial_state = initial_state, dtype = tf.float32)

# softmax_classifier
with tf.name_scope("softmax_classifier") as scope:
    X_for_softmax = tf.reshape(outputs,[-1,input_size]) # for softmax we should make outputs's shape [None, input_size]
    w_for_softmax = tf.get_variable("w_for_softmax", shape = [input_size,input_size], initializer = tf.contrib.layers.
                                   xavier_initializer())
    b_for_softmax = tf.Variable(tf.random_normal([input_size], name = "bias1"))
    outputs = tf.matmul(X_for_softmax,w_for_softmax) + b_for_softmax
    outputs = tf.reshape(outputs,[total_batch,sequence_length,input_size])

# define loss function and optimizer
with tf.name_scope("loss_and_optimizer") as scope:
    weights = tf.ones([total_batch,sequence_length])
    
    # i use tensorflow version 1.3.0 and this version provides contrib.seq2seq.sequence_loss lib.
    loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits = outputs, targets = Y, weights = weights))
    
    # i will use AdamOptimizer and learning_rate 0.005
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.005).minimize(loss)

with tf.name_scope("Accuracy_calculator") as scope:
    prediction = tf.argmax(outputs,2)
    is_correct = tf.equal(prediction,Y)
    Accuracy = tf.reduce_mean(tf.cast(is_correct, dtype = tf.float32))

# I define loss_graph for watching loss_val in tensorboard 
loss_graph = tf.summary.scalar("loss", loss)

# merge summary
summary = tf.summary.merge_all()

# initialize all variables
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# define writer
writer = tf.summary.FileWriter(logs_path)
writer.add_graph(sess.graph)

print("------------------------------------------------------------------")
# training
print("learning_start")
for k in range(10001):
    loss_val,result,s,_ = sess.run([loss,outputs,summary,optimizer], feed_dict = {X:X_data,Y:Y_data})
    writer.add_summary(s,global_step = k)
    if k %1000==0:
        print("step:",k,"-----","loss_val:",loss_val)
print("learning_end")
print("------------------------------------------------------------------")

# expresses sentence of batches after Machine Learning
print("translating start") 
alpha_holder = []

for j,results in enumerate(result):
    index = np.argmax(results,1)
    if j is 0:
        first_sentence = [idx2char[t] for t in index]
    else:
        alpha_holder.append(idx2char[index[-1]])       
print(''.join(first_sentence[:]),''.join(alpha_holder[:]))
print("translating end")
print("------------------------------------------------------------------")

# calculating Accuracy
Accuracy_val = sess.run([Accuracy], feed_dict = {X:X_data,Y:Y_data})
print("Accuracy:", Accuracy_val)
