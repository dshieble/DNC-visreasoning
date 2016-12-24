import numpy as np
import tensorflow as tf
from dnc.controller import BaseController


"""
A 2-Layers feedforward neural network with 128, 256 nodes respectively
"""

class FeedforwardController(BaseController):

    def network_vars(self):
        initial_std = lambda in_nodes: np.min(1e-2, np.sqrt(2.0 / in_nodes))
        input_ = self.nn_input_size

        
        
        self.W1 = tf.Variable(tf.truncated_normal([input_, 256], stddev=initial_std(input_)), name='layer1_W')
        self.W2 = tf.Variable(tf.truncated_normal([256, 256], stddev=initial_std(128)), name='layer2_W')
        self.W3 = tf.Variable(tf.truncated_normal([256, 256], stddev=initial_std(128)), name='layer3_W')
        self.W4 = tf.Variable(tf.truncated_normal([256, 256], stddev=initial_std(128)), name='layer4_W')

        self.b1 = tf.Variable(tf.zeros([256]), name='layer1_b')
        self.b2 = tf.Variable(tf.zeros([256]), name='layer2_b')
        self.b3 = tf.Variable(tf.zeros([256]), name='layer3_b')
        self.b4 = tf.Variable(tf.zeros([256]), name='layer4_b')

    def network_op(self, X):
        l1_output = tf.matmul(X, self.W1) + self.b1
        l1_activation = tf.nn.relu(l1_output)

        l2_output = tf.matmul(l1_activation, self.W2) + self.b2
        l2_activation = tf.nn.relu(l2_output)

        l3_output = tf.matmul(l2_activation, self.W3) + self.b3
        l3_activation = tf.nn.relu(l3_output)
        
        l4_output = tf.matmul(l3_activation, self.W4) + self.b4
        l4_activation = tf.nn.relu(l4_output)
        
        return l4_activation
