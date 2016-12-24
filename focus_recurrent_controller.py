import numpy as np
import tensorflow as tf
from dnc.controller import BaseController
from tensorflow.python.ops import rnn_cell
import dnc.utility as uf
"""
A recurrent controller that uses a focus window on the input at each time step

"""

class FocusRecurrentController(BaseController):
    window_size = 4
    
    def network_vars(self, batch_size):
        initial_std = lambda in_nodes: np.min(1e-2, np.sqrt(2.0 / in_nodes))
        input_ = self.nn_input_size

        
        
        self.W1 = tf.Variable(tf.truncated_normal([input_, 256], stddev=initial_std(input_)), name='layer1_W')
        self.W2 = tf.Variable(tf.truncated_normal([256, 256], stddev=initial_std(128)), name='layer2_W')
        self.W3 = tf.Variable(tf.truncated_normal([256, 256], stddev=initial_std(128)), name='layer3_W')

        self.b1 = tf.Variable(tf.zeros([256]), name='layer1_b')
        self.b2 = tf.Variable(tf.zeros([256]), name='layer2_b')
        self.b3 = tf.Variable(tf.zeros([256]), name='layer3_b')

        self.C1 = rnn_cell.GRUCell(256)
        self.state = state = self.C1.zero_state(batch_size, tf.float32)

        self.focus = tf.Variable([0,0], name="focus", dtype=int64)
        
        
    def network_op(self, X, state):
        Xw = uf.window_vector(X, self.focus[0], self.focus[1], window_size=self.window_size)
        
        l1_output = tf.matmul(Xw, self.W1) + self.b1
        l1_activation = tf.nn.relu(l1_output)

        l2_output = tf.matmul(l1_activation, self.W2) + self.b2
        l2_activation = tf.nn.relu(l2_output)

        l3_output = tf.matmul(l2_activation, self.W3) + self.b3
        l3_activation = tf.nn.relu(l3_output)
        
        output, self.state = self.C1(l3_activation, state)

        #Use softmax style instead ->  long vectors with argmaxes
        self.focus = tf
        
        return output, self.state

    
    def get_state(self):            
        return self.state
    
    def update_state(self, state):
        self.state = state
        return self.state
        