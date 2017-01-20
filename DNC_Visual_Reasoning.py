import os
import sys
import itertools
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from dnc.dnc import DNC
import dnc.utility as uf
from tqdm import tqdm
# from feedforward_controller import FeedforwardController
from basic_recurrent_controller import BasicRecurrentController
from focus_recurrent_controller import FocusRecurrentController

import time
import ipdb

sess = None
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

# TODO: For the DNC-controlled sliding window, just make the "sequence input" to the DNC be a sequence of images. Then
# make the first step of the controller to be to apply the index window on top of the image

def make_ims(kind, size=8, splits=4):
    if kind == "center":
        Input, _, Target_Output = uf.get_center_bar_images(bsize, size=size, splits=splits, stagger=False)
    elif kind == "right":
        Input, _, Target_Output = uf.get_right_bar_images(bsize, size=size, splits=splits, stagger=False)
    elif kind == "sd":
        Input,_, Target_Output = uf.get_sd_images(bsize, size=size, splits=splits,
        stagger=False,half_max_item=half_max_item)
    return Input, Target_Output

if not sess is None:
    sess.close()

#Remove logging  from previous training runs
os.system("rm /media/data_cifs/DNC_Visual_Reasoning_Results_Logs/*.npy")
    
task = "sd" 
num_iter = 15000
bsize = 10
input_side = 24
input_size = input_side**2
splits = 4
num_labels = 2
sequence_length = 16
device = "/cpu:0"
# SD parameters
half_max_item = 3
tf.reset_default_graph()
with tf.device(device):
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    ncomputer = DNC(
        FocusRecurrentController,
        input_size=input_size,
        output_size=num_labels,
        max_sequence_length=sequence_length,
        memory_words_num=10,
        memory_word_size=10,
        memory_read_heads=1,
        batch_size=bsize
    )
    assert ncomputer.controller.has_recurrent_nn

    raw_outputs, memory_views = ncomputer.get_outputs()
    output = tf.argmax(raw_outputs[:, sequence_length - 1, :], 1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(raw_outputs[:, sequence_length - 1, :], 
                                                                  ncomputer.target_output_final))

    start = time.time()
    updt = uf.get_updt(loss)
    print time.time() - start

    init = tf.global_variables_initializer()
    print "initializing..."
    sess.run(init)
    print "initialized!"

    print_step = 4000
    losses = []
    inputs = []
    outputs = []
    targets = []
    views = []
    raw_focuses_row = []
    raw_focuses_col = []
    focuses = []

    for i in tqdm(range(num_iter)):

        Input, Target_Output = make_ims(task,input_side,splits)

        OUT = sess.run([
            loss,
            output,
            memory_views,
            updt] + 
            ncomputer.controller.focus_row +
            ncomputer.controller.focus_col
            , feed_dict={
            ncomputer.input_data: Input,
            ncomputer.target_output_final: Target_Output,
            ncomputer.sequence_length: sequence_length
        })
        l, o, v = OUT[:3]
        fr = OUT[4:4+len(ncomputer.controller.focus_row)]
        fc = OUT[4+len(ncomputer.controller.focus_row):]
        pairs = zip(np.argmax(np.array(fr)[:,0,:], -1), np.argmax(np.array(fr)[:,0,:], -1))

        #TODO: retrieve scope for visualization
        losses.append(l)
        inputs.append(Input)
        outputs += list(o)
        views.append(v)
        targets += list(np.argmax(Target_Output, axis=-1))
        raw_focuses_row.append(np.array(fc)[:,0,:])
        raw_focuses_col.append(np.array(fc)[:,0,:])
        focuses.append(pairs)
        if len(targets) % print_step == 0 and len(targets) > 0:
            print "loss", np.mean(losses[-print_step:])
            print "matches", np.mean(np.array(targets[-print_step:]) == np.array(outputs[-print_step:]))
            print "saving logging for {}".format(i)
            np.save("/media/data_cifs/DNC_Visual_Reasoning_Results_Logs/targets_{}.npy".format(i), targets[-100:])
            np.save("/media/data_cifs/DNC_Visual_Reasoning_Results_Logs/inputs_{}.npy".format(i), inputs[-100:])
            np.save("/media/data_cifs/DNC_Visual_Reasoning_Results_Logs/raw_focuses_row_{}.npy".format(i), raw_focuses_row[-100:])
            np.save("/media/data_cifs/DNC_Visual_Reasoning_Results_Logs/raw_focuses_col_{}.npy".format(i), raw_focuses_col[-100:])
    ncomputer.save(sess, "ckpts", "recurrent_controller_get_sd_img_task.ckpt")
