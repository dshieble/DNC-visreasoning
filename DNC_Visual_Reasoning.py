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
from feedforward_controller import FeedforwardController
from basic_recurrent_controller import BasicRecurrentController
from focus_recurrent_controller import FocusRecurrentController
from spotlight_recurrent_controller import SpotlightRecurrentController
from circle_recurrent_controller import CircularSpotlightRecurrentController


cifs_path = "/media/data_cifs/DNC_Visual_Reasoning_Results_Logs"


#Remove logging  from previous training runs
os.system("rm {}/*.npy".format(cifs_path))

#Parameters of the task and the training
params = {}
params["timestamp"] = str(int(time.time())) #the  identifier for this test run
params["task"] = "2_square_detect" #specify the task
params["num_iter"] = 20000 #the number of batches to run
params["bsize"] = 10 #the batch size
params["input_side"] = 24 #the length of each side of each image
params["input_size"] = params["input_side"]**2 #the number of pixels
params["num_labels"] = 2 #the number of labels
params["sequence_length"] = 16 #the number of images in the sequence
params["half_max_item"] = 3 #parameter for sd task; Note: if this changes, then so should *sigma_max* in spotlight_recurrent_controller.py
params["memory_words_num"] = 10 #the number of memory words
params["memory_word_size"] = 10#the size of memory words
params["memory_read_heads"] = 1 #the number of read heads
params["print_step"] = 400 #the number of steps between each loss printintg
params["save_step"] = 4000 # the number of steps between each save
params["device"] = "/gpu:0" #Set this to /gpu:0 or /gpu:1 etc if you want to use the gpu instead
params["focus_type"] = "spotlight "
params["loss_type"] = "all_steps"

params["item_position"] = "random" # fixed or random; controls location of items in square_detect, 2_square_detect and sd tasks
params["item_size"] = "random"     # ""; controls size ""




# Import correct controller and define attention attributes
if params["focus_type"] == "none":
    ctrlr = BasicRecurrentController
    get_attributes = lambda c: ([c.W1], [c.W2], [c.W3])
    attr1 = "W1"
    attr2 = "W2"
    attr3 = "W3"
elif params["focus_type"] == "mask" or params["focus_type"] == "rowcol":
    ctrlr = FocusRecurrentController
    get_attributes = lambda c: (c.focus_row, c.focus_col, c.focus_mask)
elif params["focus_type"] == "spotlight":
    ctrlr =  SpotlightRecurrentController
    get_attributes = lambda c: (c.spotlight_row, c.spotlight_col, c.spotlight_sigma)
elif params["focus_type"] == "circular_spotlight":
    ctrlr = CircularSpotlightRecurrentController
    get_attributes = lambda c: (c.spotlight_row, c.spotlight_col, c.spotlight_radius)

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
os.system("cp DNC_Visual_Reasoning.py {}/{}/DNC_Visual_Reasoning_snapshot.py".format(cifs_path, params["timestamp"]))
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
    attr1, attr2, attr3 = get_attributes(ncomputer.controller)
    output, loss = ncomputer.get_elementwise_loss(params["loss_weightings"]) 
    
    print "initializing..."
    updt, grads = uf.get_updt(loss)
    init = tf.global_variables_initializer()
    sess.run(init)
    print "initialized!"

    loss_vals = []
    input_vals = []
    output_vals = []
    target_vals = []
    view_vals = []
    attributes = []
    mem = []
       
for i in tqdm(range(params["num_iter"])):

    #Get the data and expected output for this batch
    Input, Target_Output = uf.make_ims(params)

    #Run the  update step

    OUT = sess.run([
    loss,
    output,
    ncomputer.packed_memory_view,
    updt] +  attr1 + attr2 + attr3, 
    feed_dict={
        ncomputer.input_data: Input,
        ncomputer.target_output: Target_Output
    })

    l, o, v = OUT[:3]
    out_attr1 = OUT[4:4 + len(attr1)]
    out_attr2 = OUT[4 + len(attr1):4 + len(attr1) + len(attr2)]
    out_attr3 = OUT[4 + len(attr1) +  len(attr2):4 + len(attr1) + len(attr2) + len(attr3)]


    #Keep track of the values at this timestep
    loss_vals.append(l)
    input_vals.append(Input)
    output_vals += list(o)
    view_vals.append(v)
    target_vals += list(Target_Output)
    mem.append(ncomputer.packed_memory_view)
    attributes.append({"attr1":np.array(out_attr1), "attr2":np.array(out_attr2), "attr3":np.array(out_attr3)})

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
        np.save("{}/{}/attributes_{}.npy".format(cifs_path, params["timestamp"], i), attributes[-50:])

        #Save the weights of the model - disabled because the model checkpoints are big and bulky 
        # ncomputer.save(sess, 
        #                "{}/{}".format(params["timestamp"]), cifs_path, 
        #                "saved_weights_{}.npy".format(i))

