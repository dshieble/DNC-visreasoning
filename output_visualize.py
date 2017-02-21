import glob
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
import subprocess
import ipdb

plt.ioff()
output_types = ["losses","outputs", "targets", "inputs", "attributes","memory"]


# Internal Parameters
big_act_min = -1
big_act_max = 1
small_act_min = -1e-3
small_act_max = 1e-3
shell_val = True

def reasoning_visualizer(task_type, focus_type,output_dir, save_dir):
		
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	for iter in range(len(output_types)):
	
		# Load and sort output data
		typ = output_types[iter]
		data_list = glob.glob(os.path.join(output_dir,typ + "*.npy"))
		data_list.sort(key = lambda x: int(filter(None, re.split("[._]+", x))[-2]))	
	
		if typ == "losses":
		
			losses = []
			accs = []
			
			for ll in range(len(data_list)):
			
				losses.append(np.load(data_list[ll]).item()["loss"])
				accs.append(np.load(data_list[ll]).item()["matches"])
		
			losses = np.array(losses)
			accs = np.array(accs)
			
			fig = plt.figure()
			plt.plot(losses)
			fig.savefig(os.path.join(save_dir,'losses.png'))
			
			fig = plt.figure()
			plt.plot(accs)
			fig.savefig(os.path.join(save_dir,'accs.png'))
					
		elif typ == "outputs":
			out_start = np.load(data_list[0])[0,:,:]
			out_finish = np.load(data_list[-1])[-1,:,:]
			
			imsave(os.path.join(save_dir,"initial_output.png"),out_start)
			imsave(os.path.join(save_dir,"final_output.png"),out_finish)
			
		elif typ == "targets":
			tg_start = np.load(data_list[0])[0,:,:]
			tg_finish = np.load(data_list[-1])[-1,:,:]
			
			imsave(os.path.join(save_dir,"initial_target.png"),tg_start, vmin = small_act_min, vmax = small_act_max)
			imsave(os.path.join(save_dir,"final_target.png"),tg_finish, vmin = small_act_min, vmax = small_act_max)

		elif typ == "inputs":

			input_dir = os.path.join(save_dir,"Inputs")
			if not os.path.exists(input_dir):
				os.mkdir(input_dir)
			
			sz = int(np.sqrt(np.shape(np.load(data_list[0]))[-1]))
			tm = np.shape(np.load(data_list[0]))[2]
			
			for tt in range(tm):
			
				in_start = np.reshape(np.squeeze(np.load(data_list[0])[0,0,tt,:]),(sz,sz))
				in_finish = np.reshape(np.squeeze(np.load(data_list[-1])[-1,-1,tt,:]),(sz,sz))
				
				imsave(os.path.join(input_dir,"initial_input_" + str('%03d' % (tt + 1)) + ".png"),in_start, vmin = small_act_min*(tt > 0) + big_act_min*(tt==0), 
					vmax = small_act_max*(tt > 0) + big_act_max*(tt == 0))
				
				imsave(os.path.join(input_dir,"final_input_" + str('%03d' % (tt + 1)) + ".png"),in_finish, vmin = small_act_min*(tt > 0) + big_act_min*(tt==0), 
					vmax = small_act_max*(tt > 0) + big_act_max*(tt == 0))
					
				initial_movie_string = "ffmpeg -f image2 -r 2 -i " + \
						os.path.join(input_dir,"initial_input_%03d.png") + " -vf scale=50:50 -vcodec mpeg4 -y "  + \
						os.path.join(input_dir,"initial_input.mp4")
						
				final_movie_string = "ffmpeg -f image2 -r 2 -i " + \
					os.path.join(input_dir,"final_input_%03d.png") +  " -vf scale=50:50 -vcodec mpeg4 -y " + \
					os.path.join(input_dir,"final_input.mp4")

				subprocess.call(initial_movie_string, shell=shell_val)
				subprocess.call(final_movie_string, shell=shell_val)
					
			for fn in glob.glob(os.path.join(input_dir, "*.png")):
				os.remove(fn)
		
		elif typ == "attributes":
		
			focus_dir = os.path.join(save_dir,"Focus")
			if not os.path.exists(focus_dir):
				os.mkdir(focus_dir)
				
			if focus_type == "mask":
				sz = int(np.sqrt(np.shape(np.squeeze(np.load(data_list[0])[0]["attr3"][:,0,:]))[-1]))
				tm = np.shape(np.squeeze(np.load(data_list[0])[0]["attr3"][:,0,:]))[0]

				for tt in range(tm):
						
					focus_start = np.reshape(np.squeeze(np.load(data_list[0])[0]["attr3"][tt,0,:]),(sz,sz))
					focus_finish = np.reshape(np.squeeze(np.load(data_list[-1])[-1]["attr3"][tt,-1,:]),(sz,sz))
					
					imsave(os.path.join(focus_dir,"initial_focus_" + str('%03d' % (tt + 1)) + ".png"),focus_start, small_act_min*(tt > 0) + big_act_min*(tt==0),
						vmax = small_act_max*(tt > 0) + big_act_max*(tt == 0))
					imsave(os.path.join(focus_dir,"final_focus_" + str('%03d' % (tt + 1)) + ".png"),focus_finish, small_act_min*(tt > 0) + big_act_min*(tt==0),
						vmax = small_act_max*(tt > 0) + big_act_max*(tt == 0))
						
					initial_movie_string = "ffmpeg -f image2 -r 2 -i " + \
						os.path.join(focus_dir,"initial_focus_%03d.png") + " -vf scale=50:50 -vcodec mpeg4 -y "  + \
						os.path.join(focus_dir,"initial_focus.mp4")
						
					final_movie_string = "ffmpeg -f image2 -r 2 -i " + \
						os.path.join(focus_dir,"final_focus_%03d.png") +  " -vf scale=50:50 -vcodec mpeg4 -y " + \
						os.path.join(focus_dir,"final_focus.mp4")

					subprocess.call(initial_movie_string, shell=shell_val)
					subprocess.call(final_movie_string, shell=shell_val)
				
				for fn in glob.glob(os.path.join(focus_dir, "*.png")):
					os.remove(fn)
					
					
			elif focus_type == "rowcol":
				sz = int(np.sqrt(np.shape(np.squeeze(np.load(data_list[0])[0]["attr3"][:,0,:]))[-1]))
				tm = np.shape(np.squeeze(np.load(data_list[0])[0]["attr3"][:,0,:]))[0]

				for tt in range(tm):
						
					focusrow_start = np.transpose(np.squeeze(np.load(data_list[0])[0]["attr1"][tt,0,:]))
					focuscol_start = np.squeeze(np.load(data_list[0])[0]["attr2"][tt,0,:])
					
					focusrow_finish = np.transpose(np.squeeze(np.load(data_list[-1])[-1]["attr1"][tt,0,:]))
					focuscol_finish = np.squeeze(np.load(data_list[-1])[-1]["attr2"][tt,0,:])
					
					focus_start = np.kron(focusrow_start,focuscol_start)
					focus_finish = np.kron(focusrow_finish,focuscol_finish)
					
					imsave(os.path.join(focus_dir,"initial_focus_" + str('%03d' % (tt + 1)) + ".png"),focus_start, small_act_min*(tt > 0) + big_act_min*(tt==0),
						vmax = small_act_max*(tt > 0) + big_act_max*(tt == 0))
					imsave(os.path.join(focus_dir,"final_focus_" + str('%03d' % (tt + 1)) + ".png"),focus_finish, small_act_min*(tt > 0) + big_act_min*(tt==0),
						vmax = small_act_max*(tt > 0) + big_act_max*(tt == 0))
						
					initial_movie_string = "ffmpeg -f image2 -r 2 -i " + \
						os.path.join(focus_dir,"initial_focus_%03d.png") + " -vf scale=50:50 -vcodec mpeg4 -y "  + \
						os.path.join(focus_dir,"initial_focus.mp4")
						
					final_movie_string = "ffmpeg -f image2 -r 2 -i " + \
						os.path.join(focus_dir,"final_focus_%03d.png") +  " -vf scale=50:50 -vcodec mpeg4 -y " + \
						os.path.join(focus_dir,"final_focus.mp4")

					subprocess.call(initial_movie_string, shell=shell_val)
					subprocess.call(final_movie_string, shell=shell_val)
					
				for fn in glob.glob(os.path.join(focus_dir, "*.png")):
					os.remove(fn)
					
		elif typ == "memory":
		
			mem_dir = os.path.join(save_dir,'Memory')
			
			if not os.path.exists(os.path.join(mem_dir)):
				os.mkdir(mem_dir)
		
			ag_start = np.squeeze(np.load(data_list[0])[0]["allocation_gates"][0,:,:])
			ww_start = np.squeeze(np.load(data_list[0])[0]["write_weightings"][0,:,:])
			wg_start = np.squeeze(np.load(data_list[0])[0]["write_gates"][0,:,:])
			fg_start = np.squeeze(np.load(data_list[0])[0]["free_gates"][0,:,:])
			rw_start = np.squeeze(np.load(data_list[0])[0]["read_weightings"][0,:,:,:])
			uv_start = np.squeeze(np.load(data_list[0])[0]["usage_vectors"][0,:,:])

			ag_finish = np.squeeze(np.load(data_list[-1])[-1]["allocation_gates"][-1,:,:])
			ww_finish = np.squeeze(np.load(data_list[-1])[-1]["write_weightings"][-1,:,:])
			wg_finish = np.squeeze(np.load(data_list[-1])[-1]["write_gates"][-1,:,:])
			fg_finish = np.squeeze(np.load(data_list[-1])[-1]["free_gates"][-1,:,:])
			rw_finish = np.squeeze(np.load(data_list[-1])[-1]["read_weightings"][-1,:,:,:])
			uv_finish = np.squeeze(np.load(data_list[-1])[-1]["usage_vectors"][-1,:,:])
			
			tm = np.shape(ag_start)[0]
			
			# Plots
			
			fig = plt.figure()
			plt.plot(ag_start)
			fig.savefig(os.path.join(mem_dir, "ag_start.png"))
			
			fig = plt.figure()
			plt.plot(ag_finish)
			fig.savefig(os.path.join(mem_dir, "ag_finish.png"))

			fig = plt.figure()
			plt.plot(wg_start)
			fig.savefig(os.path.join(mem_dir, "wg_start.png"))
			
			fig = plt.figure()
			plt.plot(wg_finish)
			fig.savefig(os.path.join(mem_dir, "wg_finish.png"))

			fig = plt.figure()
			plt.plot(fg_start)
			fig.savefig(os.path.join(mem_dir, "fg_start.png"))
			
			fig = plt.figure()
			plt.plot(fg_finish)
			fig.savefig(os.path.join(mem_dir, "fg_finish.png"))			
			
			# Images
			
			imsave(os.path.join(mem_dir,"ww_start.png"),ww_start,vmin=0,vmax=1e-2)
			
			imsave(os.path.join(mem_dir,"ww_finish.png"),ww_finish,vmin=0,vmax=1e-2)
		
			imsave(os.path.join(mem_dir,"rw_start.png"),rw_start,vmin=0,vmax=1e-2)
			
			imsave(os.path.join(mem_dir,"rw_finish.png"),rw_finish,vmin=0,vmax=1e-2)	

			imsave(os.path.join(mem_dir,"uv_start.png"),uv_start,vmin=0,vmax=1e-2)
			
			imsave(os.path.join(mem_dir,"uv_finish.png"),uv_finish,vmin=0,vmax=1e-2)
			
			# Movies
			
			for tt in range(tm):
			
				mem_matrix_start = np.squeeze(np.load(data_list[0])[0]["memory_matrix"][0,tt,:,:])
				mem_matrix_finish = np.squeeze(np.load(data_list[-1])[-1]["memory_matrix"][-1,tt,:,:])
				
				link_matrix_start = np.squeeze(np.load(data_list[0])[0]["link_matrix"][0,tt,:,:])
				link_matrix_finish = np.squeeze(np.load(data_list[-1])[-1]["link_matrix"][-1,tt,:,:])
				
				imsave(os.path.join(mem_dir,"initial_memory_matrix_" + str('%03d' % (tt + 1)) + ".png"),mem_matrix_start, vmin=-1e-2,vmax=1e-2)
				imsave(os.path.join(mem_dir,"final_memory_matrix_" + str('%03d' % (tt + 1)) + ".png"),mem_matrix_finish, vmin=-1e-2,vmax=1e-2)
				imsave(os.path.join(mem_dir,"initial_link_matrix_" + str('%03d' % (tt + 1)) + ".png"),link_matrix_start, vmin=0,vmax=1e-2)
				imsave(os.path.join(mem_dir,"final_link_matrix_" + str('%03d' % (tt + 1)) + ".png"),link_matrix_start, vmin=0,vmax=1e-2)
				
			initial_movie_string_mem = "ffmpeg -f image2 -r 2 -i " + \
				os.path.join(mem_dir,"initial_memory_matrix_%03d.png") + " -vf scale=50:50 -vcodec mpeg4 -y "  + \
				os.path.join(mem_dir,"initial_memory_matrix.mp4")
					
			final_movie_string_mem = "ffmpeg -f image2 -r 2 -i " + \
				os.path.join(mem_dir,"final_memory_matrix_%03d.png") +  " -vf scale=50:50 -vcodec mpeg4 -y " + \
				os.path.join(mem_dir,"final_memory_matrix.mp4")
				
			initial_movie_string_link = "ffmpeg -f image2 -r 2 -i " + \
				os.path.join(mem_dir,"initial_link_matrix_%03d.png") + " -vf scale=50:50 -vcodec mpeg4 -y "  + \
				os.path.join(mem_dir,"initial_link_matrix.mp4")
					
			final_movie_string_link = "ffmpeg -f image2 -r 2 -i " + \
				os.path.join(mem_dir,"final_link_matrix_%03d.png") +  " -vf scale=50:50 -vcodec mpeg4 -y " + \
				os.path.join(mem_dir,"final_link_matrix.mp4")
				
			subprocess.call(initial_movie_string_mem, shell=shell_val)
			subprocess.call(final_movie_string_mem, shell=shell_val)
			subprocess.call(initial_movie_string_link, shell=shell_val)
			subprocess.call(final_movie_string_link, shell=shell_val)
								
			for fn in glob.glob(os.path.join(mem_dir, "*memory*.png")):
				os.remove(fn)

			for fn in glob.glob(os.path.join(mem_dir, "*link*.png")):
				os.remove(fn)

