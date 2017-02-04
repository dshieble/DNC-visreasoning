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
from feedforward_controller import FeedforwardController
from basic_recurrent_controller import BasicRecurrentController
from focus_recurrent_controller import FocusRecurrentController
from spotlight_recurrent_controller import SpotlightRecurrentController
import time

cifs_path = "./media/data_cifs/DNC_Visual_Reasoning_Results_Logs"
#Remove logging  from previous training runs
os.system("rm {}/*.npy".format(cifs_path))

#Parameters of the task and the training
params = {}
params["timestamp"] = str(int(time.time())) #the  identifier for this test run
params["task"] = "center" #specify the task
params["num_iter"] = 15000 #the number of batches to run
params["bsize"] = 10 #the batch size
params["input_side"] = 24 #the length of each side of each image
params["input_size"] = params["input_side"]**2 #the number of pixels
params["num_labels"] = 2 #the number of labels
params["sequence_length"] = 16 #the number of images in the sequence
params["half_max_item"] = 3 #parameter for sd task; Note: if this changes, then so should *sigma_max* in spotlight_recurrent_controller.py
params["memory_words_num"] = 10 #the number of memory words
params["memory_word_size"] = 10#the size of memory words
params["memory_read_heads"] = 1 #the number of read heads
params["print_step"] = 500 #the number of steps between each loss printintg
params["save_step"] = 4000 # the number of steps between each save

params["device"] = "/gpu:0" #Set this to /gpu:0 or /gpu:1 etc if you want to use the gpu instead
params["focus_type"] = "spotlight"
params["loss_type"] = "all_steps"


# Import correct controller and define attention attributes

if params["focus_type"] == "none_feedforward":
    from feedforward_controller import FeedforwardController as ctrlr
    attr1 = "focus_row"
    attr2 = "focus_col"
    attr3 = "focus_mask"
elif params["focus_type"] == "none_recurrent":
    from basic_recurrent_controller import BasicRecurrentController as ctrlr
    attr1 = "focus_row"
    attr2 = "focus_col"
    attr3 = "focus_mask"
elif params["focus_type"] == "mask" or params["focus_type"] == "rowcol":
    from focus_recurrent_controller import FocusRecurrentController as ctrlr
    attr1 = "focus_row"
    attr2 = "focus_col"
    attr3 = "focus_mask"
elif params["focus_type"] == "spotlight":
    from spotlight_recurrent_controller import SpotlightRecurrentController as ctrlr
    attr1 = "spotlight_row"
    attr2 = "spotlight_col"
    attr3 = "spotlight_sigma"

# Set loss function

# Loss at all time steps
if params["loss_type"] == "all_steps": 
    params["loss_weightings"] = np.ones(params["sequence_length"])

# Loss at last timestep     
elif params["loss_type"] == "last_step":
    params["loss_weightings"] = np.array([i == (params["sequence_length"] - 1) for i in range(params["sequence_length"])])  

# Loss increasing by timestep
elif params["loss_type"] == "increasing": 
    params["loss_weightings"] = np.arange(params["sequence_length"]) 

assert len(params["loss_weightings"]) == params["sequence_length"], ("Length of loss weights must be equal to sequence length")

#Test
_, _ = uf.make_ims(params)


#Make the directory for this run of the algorithm and save the params to it

os.system("mkdir -p {}/{}".format(cifs_path, params["timestamp"]))
np.save("{}/{}/params.npy".format(cifs_path, params["timestamp"]), params)


#Reset the graph and run the algorithm
tf.reset_default_graph()
with tf.device(params["device"]):
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
  
    #build the neural computer
    ncomputer = DNC(
        ctrlr,
        input_size=params["input_size"],
        output_size=params["num_labels"],
        sequence_length=params["sequence_length"],
        controller_params={"focus_type":params["focus_type"]},
        memory_words_num=params["memory_words_num"],
        memory_word_size=params["memory_word_size"],
        memory_read_heads=params["memory_read_heads"],
        batch_size=params["bsize"]
    )

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
    mem = []

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
	    getattr(ncomputer.controller,attr1) + 
	    getattr(ncomputer.controller,attr2) +
	    getattr(ncomputer.controller,attr3)
	    , feed_dict={
	    ncomputer.input_data: Input,
	    ncomputer.target_output: Target_Output
        })

        l, o, v = OUT[:3]

      	out_attr1 = OUT[4:4 + len(getattr(ncomputer.controller,attr1))]
        out_attr2 = OUT[4 + len(getattr(ncomputer.controller,attr1)):4 + len(getattr(ncomputer.controller,attr1)) +
                 len(getattr(ncomputer.controller,attr2))]
        out_attr3 = OUT[4 + len(getattr(ncomputer.controller,attr1)) +
                 len(getattr(ncomputer.controller,attr2)):4 + len(getattr(ncomputer.controller,attr1)) +
                 len(getattr(ncomputer.controller,attr2)) + len(getattr(ncomputer.controller,attr3))]

        #Keep track of the values at this timestep
        loss_vals.append(l)
        input_vals.append(Input)
        output_vals += list(o)
        view_vals.append(v)
        target_vals += list(Target_Output)
        mem.append(ncomputer.packed_memory_view)
        if params["focus_type"] == "rowcol":
            focuses.append((np.array(out_attr1)[:,0,:], np.array(out_attr2)[:,0,:]))
        elif params["focus_type"] == "mask":
            focuses.append(np.array(out_attr3)[:,0,:])
	elif params["focus_type"] == "spotlight":
	    focuses.append((np.array(out_attr1)[:,0,:], np.array(out_attr2)[:,0,:], np.array(out_attr3)[:,0,:]))

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



