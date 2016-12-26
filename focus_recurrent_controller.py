import numpy as np
import tensorflow as tf
from dnc.controller import BaseController
from tensorflow.python.ops import rnn_cell
import dnc.utility as uf
"""
A recurrent controller that uses a focus window on the input at each time step

To use this controller, pass in a "sequence" of the same image repeated multiple times. At each timestep, the controller will apply attention to the image and pass the attended-image into a standard recurrent controller

"""

class FocusRecurrentController(BaseController):
    window_size = 4
    nn_output_size = 256
    
    def network_vars(self):
        initial_std = lambda in_nodes: np.min(1e-2, np.sqrt(2.0 / in_nodes))

        windowed_input_size = self.window_size**2 + self.word_size * self.read_heads
        
        self.W1 = tf.Variable(tf.truncated_normal([windowed_input_size, 256], 
                                                  stddev=initial_std(windowed_input_size)), name='layer1_W')
        self.W2 = tf.Variable(tf.truncated_normal([256, 256], stddev=initial_std(128)), name='layer2_W')
        self.W3 = tf.Variable(tf.truncated_normal([256, 256], stddev=initial_std(128)), name='layer3_W')

        self.b1 = tf.Variable(tf.zeros([256]), name='layer1_b')
        self.b2 = tf.Variable(tf.zeros([256]), name='layer2_b')
        self.b3 = tf.Variable(tf.zeros([256]), name='layer3_b')

        self.C1 = rnn_cell.GRUCell(self.nn_output_size)
        self.state = self.C1.zero_state(self.batch_size, tf.float32)
        
        #Number of possible rows/columns to place the top left corner of the focus window
        self.focus_range = int(np.sqrt(self.input_size)) - self.window_size
        #Multipliers to update the row focus indices
        self.focus_row_updater = tf.Variable(tf.truncated_normal([self.nn_output_size, self.focus_range]),
                                               name='focus_updater_row') 
        #Multiplier to update the col focus indices
        self.focus_col_updater = tf.Variable(tf.truncated_normal([self.nn_output_size, self.focus_range]),
                                               name='focus_col_updater')
                
        #The focus row and focus column vectors encode the distribution of possible index locations to move the focus window,
        #encoded as a vector of index-weights. Indices are chosen by an argmax. These tensors store a matrix of the focus
        #vectors at each timestep. Note that the focus isn't a variable. It's dynamically updated on each batch.
        self.focus_row = [tf.random_uniform((self.batch_size, self.focus_range)) for s in range(self.max_sequence_length + 1)]
        self.focus_col = [tf.random_uniform((self.batch_size, self.focus_range)) for s in range(self.max_sequence_length + 1)]
        
        
    def network_op(self, X, state, t):
        #Reset focus at the beginning of each iteration
        if t == 0:
            self.focus_row = [tf.random_uniform((self.batch_size, self.focus_range)) 
                              for s in range(self.max_sequence_length + 1)]
            self.focus_col = [tf.random_uniform((self.batch_size, self.focus_range)) 
                              for s in range(self.max_sequence_length + 1)]
    
        #The attended-to image
        Xf = self.apply_attention(X, self.focus_row[t], self.focus_col[t]) # Here we add an attentional filter to the image
        l1_output = tf.matmul(Xf, self.W1) + self.b1
        l1_activation = tf.nn.relu(l1_output)

        l2_output = tf.matmul(l1_activation, self.W2) + self.b2
        l2_activation = tf.nn.relu(l2_output)

        l3_output = tf.matmul(l2_activation, self.W3) + self.b3
        l3_activation = tf.nn.relu(l3_output)
        
        output, self.state = self.C1(l3_activation, state)

        return output, self.state

    def apply_attention(self, X, focus_row, focus_col):
        #Given some input X and some focus vectors, apply attention and get the input to the controller. 
        #Note that this could also include the application of a CNN on the image.
        ind_row = tf.cast(tf.argmax(focus_row, 1), tf.int32)
        ind_col = tf.cast(tf.argmax(focus_col, 1), tf.int32)
        core = X[:, :self.input_size]
        rest = X[:, self.input_size:]
        height, width = np.int32(np.sqrt(self.input_size)), np.int32(np.sqrt(self.input_size))
        assert height * width == self.input_size, (height, width, input_size)
        
        coreSq = tf.reshape(core, (self.batch_size, height, width))
        
        coreSqW = tf.concat(0, [uf.window_vector(coreSq[i:i+1], ind_row[i], ind_col[i], window_size=self.window_size) 
                                for i in range(self.batch_size)])
        
        
        coreFocus =  tf.reshape(coreSqW, (self.batch_size, self.window_size**2))
        return tf.concat(1, (coreFocus, rest))
        
    def final_output(self, pre_output, nn_output, new_read_vectors, t):
        """
            Override the basic final_output method to update focus according to the memory and the recurrent state
        """
        flat_read_vectors = tf.reshape(new_read_vectors, (self.batch_size, self.word_size * self.read_heads))

        final_output = pre_output + tf.matmul(flat_read_vectors, self.mem_output_weights)

        self.focus_row[t+1], self.focus_col[t+1] = self.get_new_focus(self.focus_row[t], self.focus_col[t], nn_output)
        
        return final_output

    def get_new_focus(self, focus_row, focus_col, nn_output):
        focus_row_adder = tf.matmul(nn_output, self.focus_row_updater)
        focus_col_adder = tf.matmul(nn_output, self.focus_col_updater)            

        new_focus_row = focus_row + focus_row_adder
        new_focus_col = focus_col + focus_col_adder
        return new_focus_row, new_focus_col
    
    def get_state(self):            
        return self.state
    
    def update_state(self, state):
        self.state = state
        return self.state
        
    