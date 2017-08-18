import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from internal import util, math, plot

plt.style.use('ggplot')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x,W, [1, 1, 1, 1], padding='VALID')

def apply_depthwise_conv(x,kernel_size,num_channels,depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights),biases))

def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')


dataset = util.read_data('WISDM_at_v2.0_raw.txt', 1)
dataset['x-axis'] = math.normalize(dataset['x-axis'])
dataset['y-axis'] = math.normalize(dataset['y-axis'])
dataset['z-axis'] = math.normalize(dataset['z-axis'])


unique_activities = pd.unique(dataset["activity"])

for activity in unique_activities:
    subset = dataset[dataset["activity"] == activity][:180]
    plot.activity(activity,subset)

segments, labels = util.create_segments(dataset)
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
reshaped_segments = segments.reshape(len(segments), 1,90, 3)


train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]


input_height = 1
input_width = 90
num_labels = unique_activities.shape[0]
num_channels = 3

batch_size = 10
kernel_size = 60
depth = 60
num_hidden = 1000

learning_rate = 0.0001
training_epochs = 8

total_batches = train_x.shape[0] // batch_size


X = tf.placeholder(tf.float32, shape=[None,input_height,input_width,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])

c = apply_depthwise_conv(X,kernel_size,num_channels,depth)
p = apply_max_pool(c,20,2)
depth_2 = depth//10
filter_width_2 = 6 #Should be less than equal to 6
c = apply_depthwise_conv(p,filter_width_2,depth*num_channels,depth_2)

shape = c.get_shape().as_list()
c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])

f_weights_l1 = weight_variable([shape[1] * shape[2] * depth * num_channels * depth_2, num_hidden])
f_biases_l1 = bias_variable([num_hidden])
f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1),f_biases_l1))

out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)


loss = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


cost_history = np.empty(shape=[1],dtype=float)

with tf.Session() as session:
    tf.initialize_all_variables().run()
    for epoch in range(training_epochs):
        for b in range(total_batches):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            _, loss_value = session.run([optimizer, loss],feed_dict={X: batch_x, Y : batch_y})
            cost_history = np.append(cost_history,loss_value)
        print("Epoch: ",epoch," Training Loss: ",loss_value," Training Accuracy: ",
              session.run(accuracy, feed_dict={X: train_x, Y: train_y}))

    print("Testing Accuracy:", session.run(accuracy, feed_dict={X: test_x, Y: test_y}))