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

cifs_path = "/media/data_cifs/DNC_Visual_Reasoning_Results_Logs"
#Remove logging  from previous training runs
os.system("rm {}/*.npy".format(cifs_path))

#Parameters of the task and the training
params = {}
params["timestamp"] = str(int(time.time()))
params["task"] = "lrb" 
params["num_iter"] = 15000
params["bsize"] = 10
params["input_side"] = 24
params["input_size"] = params["input_side"]**2
params["splits"] = 4
params["num_labels"] = 2
params["sequence_length"] = 16
params["half_max_item"] = 3
params["memory_words_num"] = 10
params["memory_word_size"] = 10
params["memory_read_heads"] = 1
params["print_step"] = 500
params["save_step"] = 4000

params["device"] = "/cpu:0" #Set this to /gpu:0 or /gpu:1 etc if you want to use the gpu instead
params["focus_type"] = "mask"

## Loss at every timestep
# params["loss_weightings"] = np.ones(params["sequence_length"]) 

## Loss at last timestep
params["loss_weightings"] = np.array([i == (params["sequence_length"] - 1) for i in range(params["sequence_length"])])  

## Loss increasing by timestep
# params["loss_weightings"] = np.arange(params["sequence_length"]) 

assert len(params["loss_weightings"]) == params["sequence_length"], ("Length of loss weights must be equal to sequence length")

#Test
_, _ = uf.make_ims(params)


#Make the directory for this run of the algorithm and save the params to it
os.system("mkdir {}/{}".format(cifs_path, params["timestamp"]))
np.save("{}/{}/params.npy".format(cifs_path, params["timestamp"]), params)


#Reset the graph and run the algorithm
tf.reset_default_graph()
with tf.device(params["device"]):
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    #build the neural computer
    ncomputer = DNC(
        FocusRecurrentController,
        input_size=params["input_size"],
        output_size=params["num_labels"],
        sequence_length=params["sequence_length"],
        controller_params={"focus_type":params["focus_type"]},
        memory_words_num=params["memory_words_num"],
        memory_word_size=params["memory_word_size"],
        memory_read_heads=params["memory_read_heads"],
        batch_size=params["bsize"]
    )
    assert ncomputer.controller.has_recurrent_nn

    output, loss = ncomputer.get_elementwise_loss(params["loss_weightings"]) 
    
    print "initializing..."
    updt = uf.get_updt(loss)
    init = tf.global_variables_initializer()
    sess.run(init)
    print "initialized!"

    loss_vals = []
    input_vals = []
    output_vals = []
    target_vals = []
    view_vals = []
    focuses = []
    focuses = []
    
    #Run the training
    for i in tqdm(range(params["num_iter"])):

        #Get the data and expected output for this batch
        Input, Target_Output = uf.make_ims(params)

        #Run the  update step
        OUT = sess.run([
            loss,
            output,
            ncomputer.packed_memory_view,
            updt] + 
            ncomputer.controller.focus_row +
            ncomputer.controller.focus_col +
            ncomputer.controller.focus_mask
            , feed_dict={
            ncomputer.input_data: Input,
            ncomputer.target_output: Target_Output
        })
        l, o, v = OUT[:3]
        #For simplicity and easy of switching, we build all of these now. Delete this when we decide on mask versus row_col
        fr = OUT[4:4 + len(ncomputer.controller.focus_row)]
        fc = OUT[4 + len(ncomputer.controller.focus_row):4 + len(ncomputer.controller.focus_row) +
                 len(ncomputer.controller.focus_col)]
        fm = OUT[4 + len(ncomputer.controller.focus_row) + len(ncomputer.controller.focus_col):] 

        #Keep track of the values at this timestep
        loss_vals.append(l)
        input_vals.append(Input)
        output_vals += list(o)
        view_vals.append(v)
        target_vals += list(Target_Output)
        if params["focus_type"] == "rowcol":
            focuses.append((np.array(fr)[:,0,:], np.array(fc)[:,0,:]))
        elif params["focus_type"] == "mask":
            focuses.append(np.array(fm)[:,0,:])

        #Print the loss and accuracy thus far
        if len(target_vals) % params["print_step"] == 0 and len(target_vals) > 0:
            print "np.array(target_vals).shape", np.array(target_vals).shape
            print "np.array(output_vals).shape", np.array(output_vals).shape

            losses = {}
            losses["loss"] = np.mean(loss_vals[-params["print_step"]:])
            losses["matches"] = np.mean(np.argmax(np.array(output_vals)[-params["print_step"]:, -1], -1) == 
                                     np.argmax(np.array(target_vals)[-params["print_step"]:, -1], -1))

            print "loss", losses["loss"]
            print "matches", losses["matches"]

            np.save("{}/{}/losses_{}.npy".format(cifs_path, params["timestamp"], i), losses)

        #Save the model and the masks generated
        if len(target_vals) % params["save_step"] == 0 and len(target_vals) > 0:
            print "saving for {}".format(i)
            np.save("{}/{}/outputs_{}.npy".format(cifs_path, params["timestamp"], i), output_vals[-50:])
            np.save("{}/{}/targets_{}.npy".format(cifs_path, params["timestamp"], i), target_vals[-50:])
            np.save("{}/{}/inputs_{}.npy".format(cifs_path, params["timestamp"], i), input_vals[-50:])
            np.save("{}/{}/focuses_{}.npy".format(cifs_path, params["timestamp"], i), focuses[-50:])
            #Save the weights of the model - disabled because the model checkpoints are big and bulky 
            # ncomputer.save(sess, 
            #                "{}/{}".format(params["timestamp"]), cifs_path, 
            #                "saved_weights_{}.npy".format(i))



