import os
import time
from DNC_Visual_Reasoning import dnc_visual_reasoning
params = {}

# EXPERIMENTAL PARAMETERS
params["task"] = "postlocate" #specify the task
params["num_iter"] = 20000 #the number of batches to run
params["bsize"] = 10 #the batch size

# ARCHITECTURE PARAMETERS

params["focus_type"] = "mask"
params["half_max_item"] = 3 #parameter for sd task; Note: if this changes, then so should *sigma_max* in spotlight_recurrent_controller.py
params["memory_words_num"] = 10 #the number of memory words
params["memory_word_size"] = 10 #the size of memory words
params["memory_read_heads"] = 1 #the number of read heads

# STIMULUS PARAMETERS

params["input_side"] = 16 #the length of each side of each image
params["input_size"] = params["input_side"]**2 #the number of pixels
params["num_labels"] = 4 #the number of labels
params["loss_type"] = "cued_loss"
params["item_position"] = "random" # fixed or random; controls location of items in square_detect, 2_square_detect and sd tasks
params["item_size"] = "fixed"     # ""; controls size ""
params["noise"] = False
params["SOA"] = 10
params["ISI"] = 5
if params["task"] == "locate" or params["task"] == "postlocate":
	params["sequence_length"] = 2*params["SOA"] + params["ISI"]
#	assert params["loss_type"] == "cued_loss"
else:
	params["sequence_length"] = 16

# OS PARAMETERS
params["cifs_path"] = "./media/data_cifs/DNC_Visual_Reasoning_Results_Logs"
os.system("rm {}/*.npy".format(params["cifs_path"]))
params["timestamp"] = str(int(time.time())) #the  identifier for this test run
params["print_step"] = 400 #the number of steps between each loss printintg
params["save_step"] = 4000 # the number of steps between each save
params["device"] = "/gpu:0" #Set this to /gpu:0 or /gpu:1 etc if you want to use the gpu instead

# Run model

dnc_visual_reasoning(params)
