from output_visualize import reasoning_visualizer
import numpy as np
import os
import ipdb

task_number = "1487629593"
media_dir = "./media"
output_dir = os.path.join(media_dir, "data_cifs/DNC_Visual_Reasoning_Results_Logs/",task_number)
params = np.load(os.path.join(output_dir,"params.npy")).item()
task_type = params["task"]
focus_type = params["focus_type"]
save_dir = os.path.join(media_dir,task_type + "_" + focus_type + "_" + task_number)

# Run visualizer 

reasoning_visualizer(task_type,focus_type,output_dir,save_dir)


