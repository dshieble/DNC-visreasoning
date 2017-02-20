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
     #Add focus type as a parameter
    def __init__(self, input_size, output_size, memory_read_heads, 
                 memory_word_size, sequence_length, batch_size=1, focus_type="mask"):
        self.focus_type = focus_type
        super(FocusRecurrentController, self).__init__(input_size, output_size, memory_read_heads,  
                                                       memory_word_size, sequence_length, batch_size=batch_size)

    
    window_size = 3
    nn_output_size = 256
    max_scope = 8
    def network_vars(self):
        initial_std = lambda in_nodes: np.min(1e-2, np.sqrt(2.0 / in_nodes))
        self.W1 = tf.Variable(tf.truncated_normal([self.nn_input_size, 256],
                               stddev=initial_std(self.nn_input_size)), name='layer1_W')


        self.W2 = tf.Variable(tf.truncated_normal([256, 256], stddev=initial_std(256)), name='layer2_W')
        self.W3 = tf.Variable(tf.truncated_normal([256, 256], stddev=initial_std(256)), name='layer3_W')

        self.b1 = tf.Variable(tf.zeros([256]), name='layer1_b')
        self.b2 = tf.Variable(tf.zeros([256]), name='layer2_b')
        self.b3 = tf.Variable(tf.zeros([256]), name='layer3_b')

        self.C1 = rnn_cell.GRUCell(self.nn_output_size)
        
        
        #Number of possible rows/columns to place the top left corner of the focus window
        self.focus_range = int(np.sqrt(self.input_size))

        #Multipliers to update the row focus indices. I.e. weights
        self.focus_row_updater = tf.Variable(tf.truncated_normal([self.nn_output_size, self.focus_range],
                                                                  stddev=initial_std(self.nn_output_size)),
                                                                  name='focus_updater_row') 
        #Multiplier to update the col focus indices
        self.focus_col_updater = tf.Variable(tf.truncated_normal([self.nn_output_size, self.focus_range],
                                                                  stddev=initial_std(self.nn_output_size)),
                                                                  name='focus_col_updater')
        #Multiplier to update the mask
        self.focus_mask_updater = tf.Variable(tf.truncated_normal([self.nn_output_size, self.input_size],
                                                                  stddev=initial_std(self.nn_output_size)),
                                                                  name='focus_mask_updater')
        
        #The focus row and focus column vectors encode the distribution of possible index locations to move the focus window,
        #encoded as a vector of index-weights. These tensors store a matrix of the focus
        #vectors at each timestep. Note that the focus isn't a variable. It's dynamically updated on each batch.
        self.state = tf.truncated_normal([self.batch_size, self.nn_output_size])
        self.focus_row = [tf.random_uniform((self.batch_size, self.focus_range)) for s in range(self.sequence_length + 1)]
        self.focus_col = [tf.random_uniform((self.batch_size, self.focus_range)) for s in range(self.sequence_length + 1)]
        self.focus_mask = [tf.random_uniform((self.batch_size, self.input_size)) for s in range(self.sequence_length + 1)]
        
    def network_op(self, X, state, t):
        #Reset focus at the beginning of each iteration
        if t == 0:
            self.state = tf.truncated_normal([self.batch_size, self.nn_output_size])
            self.focus_row = [tf.random_uniform((self.batch_size, self.focus_range)) 
                              for s in range(self.sequence_length + 1)]
            self.focus_col = [tf.random_uniform((self.batch_size, self.focus_range)) 
                              for s in range(self.sequence_length + 1)]
            self.focus_mask = [tf.random_uniform((self.batch_size, self.input_size)) 
                              for s in range(self.sequence_length + 1)]
        #The attended image
        Xf = self.apply_attention(X, self.focus_row[t], self.focus_col[t], self.focus_mask[t]) # Here we add an attentional filter to the image
        nn_output = self.run_controller_network(Xf, state)

        return nn_output, self.state

    def run_controller_network(self, X, state):
        #Runs the controller portion of the network on the attended-to image
        self.l1_output = tf.matmul(X, self.W1) + self.b1
        self.l1_activation = tf.nn.relu(self.l1_output)

        self.l2_output = tf.matmul(self.l1_activation, self.W2) + self.b2
        self.l2_activation = tf.nn.relu(self.l2_output)

        self.l3_output = tf.matmul(self.l2_activation, self.W3) + self.b3
        self.l3_activation = tf.nn.relu(self.l3_output)

        self.rnn_output, self.state = self.C1(self.l3_activation, state)
        self.nn_output = tf.nn.relu(self.rnn_output)
        return self.nn_output

    def apply_attention(self, X, focus_row, focus_col, focus_mask):    
        #Given some input X and some focus vectors, apply attention and get the input to the controller. 
        core = X[:, :self.input_size]
        rest = X[:, self.input_size:] #the read-memory part of the input
        height, width = np.int32(np.sqrt(self.input_size)), np.int32(np.sqrt(self.input_size))

        coreSq = tf.reshape(core, (self.batch_size, height, width))

        coreSqW = tf.concat(0, [uf.apply_mask(coreSq[i:i+1], focus_row[i], focus_col[i], 
                                              focus_mask[i], self.focus_range, self.focus_type)
                               for i in range(self.batch_size)])

        coreFocus =  tf.reshape(coreSqW, (self.batch_size, height**2))
        return tf.concat(1, (coreFocus, rest))

    def final_output(self, pre_output, nn_output, new_read_vectors, t):
        """
            Override the basic final_output method to update focus according to the memory and the recurrent state
        """
        flat_read_vectors = tf.reshape(new_read_vectors, (self.batch_size, self.word_size * self.read_heads))

        final_output = pre_output + tf.matmul(flat_read_vectors, self.mem_output_weights)

        self.focus_row[t+1], self.focus_col[t+1], self.focus_mask[t+1] = self.get_new_focus(self.focus_row[t], self.focus_col[t], self.focus_mask[t], nn_output)

        return final_output

    def get_new_focus(self, focus_row, focus_col,  focus_mask, nn_output):
        #The focus row and focus col are multiplied together to yield the generated mask
        nn_output = nn_output/(1e-4 + tf.reduce_sum(tf.abs(nn_output)))

        new_focus_row = tf.matmul(nn_output, self.focus_row_updater)
        new_focus_col = tf.matmul(nn_output, self.focus_col_updater)
        new_focus_mask = tf.matmul(nn_output, self.focus_mask_updater)

        new_focus_row = new_focus_row/(1e-4 + tf.reduce_sum(tf.abs(new_focus_row)))
        new_focus_col = new_focus_col/(1e-4 + tf.reduce_sum(tf.abs(new_focus_col)))
        new_focus_mask = new_focus_mask/(1e-4 + tf.reduce_sum(tf.abs(new_focus_mask)))

        return new_focus_row, new_focus_col, new_focus_mask

    def get_state(self):            
        return self.state

    def update_state(self, state):
        self.state = state
        return self.state





