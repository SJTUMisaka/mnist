import numpy as np
import tensorflow as tf 

x = tf.placeholder(tf.float32, [None, 2025]) # placeholder for figure data
y_actual = tf.placeholder(tf.float32, shape=[None, 10]) # placeholder for labels

data_num = 60000  # number of figures
fig_w = 45       # width of each figure
k_size = 5  # kernel size of convoltion layer
p_size = 3   # pool size of max-pooling layer

trainx = np.fromfile("processed_train_data",dtype=np.float64)
trainy = np.fromfile("processed_train_label",dtype=np.float64)

testx = np.fromfile("processed_test_data",dtype=np.float64)
testy = np.fromfile("processed_test_label",dtype=np.float64)

# Reshape the matrix
trainx = trainx.reshape(data_num,2025)
testx = testx.reshape(10000,2025)
trainy = trainy.reshape(data_num,10)
testy = testy.reshape(10000,10)    



# Functions for constructing network
# Initialize weight W
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# Initialize bias b
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
# Construct a convolution layer
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Construct a max-pooling layer
def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, p_size, p_size, 1],strides=[1, 3, 3, 1], padding='SAME')




# Network
x_image = tf.reshape(x, [-1,fig_w,fig_w,1])         # reshape data to fit the network
W_conv1 = weight_variable([k_size, k_size, 1, 32])      
b_conv1 = bias_variable([32])       
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)     # first convolution layer
h_pool1 = max_pool(h_conv1)                                  # first max-pooling layer

W_conv2 = weight_variable([k_size, k_size, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      # second convolution layer
h_pool2 = max_pool(h_conv2)                                   # second max-pooling layer

W_fc1 = weight_variable([5 * 5 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*64])              
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)    # fully connected layer

keep_prob = tf.placeholder("float") 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  # dropout layer

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)   # softmax layer

cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))     # cross entropy
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess=tf.InteractiveSession()                          
sess.run(tf.global_variables_initializer())


for i in range(1000):
  batch0 = trainx[(i*100)%data_num:((i+1)*100)%data_num]
  batch1 = trainy[(i*100)%data_num:((i+1)*100)%data_num]
  if i%100 == 0:                # test every 100 times
    train_acc = accuracy.eval(feed_dict={x:batch0, y_actual: batch1, keep_prob: 1.0})
    print('step',i,'training accuracy',train_acc)
  train_step.run(feed_dict={x:batch0, y_actual: batch1, keep_prob: 0.5})

test_acc=accuracy.eval(feed_dict={x: testx, y_actual: testy, keep_prob: 1.0})
print("test accuracy",test_acc)

