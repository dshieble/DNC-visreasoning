import numpy as np
import tensorflow as tf
from dnc.controller import BaseController
from tensorflow.python.ops import rnn_cell
"""
A recurrent controller

NOTE - LARGER BATCH SIZE (at least 10) IMPROVES convergence for this controller

"""

class BasicRecurrentController(BaseController):

    def __init__(self, input_size, output_size, memory_read_heads,
                     memory_word_size, sequence_length, batch_size=1, focus_type="mask"):
        self.focus_type = focus_type
        super(BasicRecurrentController, self).__init__(input_size, output_size, memory_read_heads,  
                                                       memory_word_size, sequence_length, batch_size=batch_size)
 
    def network_vars(self, **kwargs):
        initial_std = lambda in_nodes: np.min(1e-2, np.sqrt(2.0 / in_nodes))
        input_ = self.nn_input_size

        
        self.W1 = tf.Variable(tf.truncated_normal([input_, 256], stddev=initial_std(input_)), name='layer1_W')
        self.W2 = tf.Variable(tf.truncated_normal([256, 256], stddev=initial_std(128)), name='layer2_W')
        self.W3 = tf.Variable(tf.truncated_normal([256, 256], stddev=initial_std(128)), name='layer3_W')

        self.b1 = tf.Variable(tf.zeros([256]), name='layer1_b')
        self.b2 = tf.Variable(tf.zeros([256]), name='layer2_b')
        self.b3 = tf.Variable(tf.zeros([256]), name='layer3_b')

        self.C1 = rnn_cell.GRUCell(256)
        self.state = state = self.C1.zero_state(self.batch_size, tf.float32)

    def network_op(self, X, state, t):
        l1_output = tf.matmul(X, self.W1) + self.b1
        l1_activation = tf.nn.relu(l1_output)

        l2_output = tf.matmul(l1_activation, self.W2) + self.b2
        l2_activation = tf.nn.relu(l2_output)

        l3_output = tf.matmul(l2_activation, self.W3) + self.b3
        l3_activation = tf.nn.relu(l3_output)
        
        activation, self.state = self.C1(l3_activation, state)

        return activation, self.state

    
    def get_state(self):            
        return self.state
    
    def update_state(self, state):
        self.state = state
        return self.state
        
