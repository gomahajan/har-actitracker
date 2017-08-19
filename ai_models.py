import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def apply_depthwise_conv(x, kernel_size, num_channels, depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    depthwise_conv2d = tf.nn.depthwise_conv2d(x, weights, [1, 1, 1, 1], padding='VALID')
    return tf.nn.relu(tf.add(depthwise_conv2d, biases))


def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')


class har_model:
    """

    """
    def __init__(self, num_labels, input_height, input_width, num_channels, filter_width, channels_multiplier,
                 num_hidden, filter_width_2):
        assert filter_width <= input_width, "Filter width should be less than input width"

        self.X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, num_channels])
        self.Y = tf.placeholder(tf.float32, shape=[None, num_labels])

        c = apply_depthwise_conv(self.X, filter_width, num_channels, channels_multiplier)
        p = apply_max_pool(c, 20, 2)
        # Use any channels_multiplier_2
        channels_multiplier_2 = channels_multiplier // 10

        assert filter_width_2 <= p.shape[2], "Filter width 2 should be less than input width 2"
        c = apply_depthwise_conv(p, filter_width_2, channels_multiplier * num_channels, channels_multiplier_2)

        shape = c.get_shape().as_list()
        c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])

        assert shape[3] == channels_multiplier * num_channels * channels_multiplier_2

        f_weights_l1 = weight_variable([shape[1] * shape[2] * shape[3], num_hidden])
        f_biases_l1 = bias_variable([num_hidden])
        f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1), f_biases_l1))

        out_weights = weight_variable([num_hidden, num_labels])
        out_biases = bias_variable([num_labels])
        self.y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)
        self.loss = -tf.reduce_sum(self.Y * tf.log(self.y_))

    def train_and_evaluate(self, train_x, train_y, test_x, test_y):
        learning_rate = 0.0001
        training_epochs = 8
        batch_size = 10
        total_batches = train_x.shape[0] // batch_size
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        cost_history = np.empty(shape=[1], dtype=float)

        with tf.Session() as session:
            tf.initialize_all_variables().run()
            for epoch in range(training_epochs):
                for b in range(total_batches):
                    offset = (b * batch_size) % (train_y.shape[0] - batch_size)
                    batch_x = train_x[offset:(offset + batch_size), :, :, :]
                    batch_y = train_y[offset:(offset + batch_size), :]
                    _, loss_value = session.run([optimizer, self.loss], feed_dict={self.X: batch_x, self.Y: batch_y})
                    cost_history = np.append(cost_history, loss_value)
                print("Epoch: ", epoch, " Training Loss: ", loss_value, " Training Accuracy: ",
                      session.run(accuracy, feed_dict={self.X: train_x, self.Y: train_y}))

            print("Testing Accuracy:", session.run(accuracy, feed_dict={self.X: test_x, self.Y: test_y}))
