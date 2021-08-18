#!/usr/bin/env python3

import argparse
import glob
import os, fnmatch
import numpy as np
import re
import matplotlib.pyplot as plt

# def get_parser():
# 	parser = argparse.ArgumentParser(description="Get Veenstra features")
# 	parser.add_argument("-d","--dataset_dir", help="Path to dataset_dir folder.")

# 	return parser

def read_centers_from_csv(csv_file):

	f=open(csv_file,"r")

	lines=f.readlines()
	result = []

	box = np.array([0,0])
	starting_frame = int(lines[0].split(",")[0])

	for i in lines:
		if np.array(i.split(',')[1].replace("\n","")) != 'interpolate': 
			x = int(i.split(',')[1].replace("\n","").replace("[","").replace("]","").split(" ")[0]) 
			y = int(i.split(',')[1].replace("\n","").replace("[","").replace("]","").split(" ")[-1])
			result.append(np.array([x,y]))

	f.close()

	return np.array(result)

def find_csv_filenames( path_to_dir, suffix=".csv" ):
	filenames = os.listdir(path_to_dir)
	return [ os.path.join(path_to_dir,filename) for filename in filenames if filename.endswith( suffix ) ]

def interpolate(array):

	new_array = np.zeros((4500,2))
	new_array_norm = np.zeros(4500)
	for i in range(0,array.shape[0]-1):

		D = array[i+1] - array[i]
		I = D/14

		for j in np.arange(0,15):
			new_array[15*i+j] = array[i] + I*j
			new_array_norm[15*i+j] = np.linalg.norm(array[i] + I*j)

	new_array_norm[4485:] = np.linalg.norm(array[-1])
	new_array[4485:] = array[-1]
	return new_array,new_array_norm

def get_table_center(file,day_tables_dir,tables):
	regex = re.search('part_._cam..', file)
	part_cam_table = regex.group(0)
	table_index = tables.index(str(day_tables_dir) + "Speed_date_" + str(part_cam_table) + "_table_centers.csv")
	table_file = tables[table_index]				
	table_centers = read_centers_from_csv(table_file)

	regex = re.search('part_._cam.._.', file)
	part_num = int(regex.group(0).split("_")[-1])
	table_center = ""

	if part_num == 0 or part_num == 1:
		table_center = table_centers[0]
	elif part_num == 2 or part_num == 3:
		table_center = table_centers[1]
	elif part_num == 4 or part_num == 5:
		table_center = table_centers[2]
	elif part_num == 6 or part_num == 7:
		table_center = table_centers[3]
	else:
		print("ERROR: Table center not found!")

	return table_center

def getAngle(vector_1,vector_2):

	theta = np.tan(np.abs(vector_2[1]-vector_1[1])/np.abs(vector_2[0]-vector_1[0]))
	angle = 180 - theta*180/np.pi

	return angle

def VAR_AVG_DIFANGLE(table_center,full_array):

	full_angles = []
	for vector in full_array:
		full_angles.append(getAngle(table_center,vector))
	full_angles = np.asarray(full_angles)
	return np.average(full_angles),np.var(full_angles)


def AVG_VARDIS(table_center,full_array):

	full_distance = []
	for vector in full_array:
		full_distance.append(np.linalg.norm(vector-table_center))

	full_distance = np.asarray(full_distance)
	return np.average(full_distance),np.var(full_distance)

def DECRDIS(full_array_a,full_array_b,seconds):

	frames = int(seconds*30)
	person_a_first_frames = full_array_a[:frames]
	person_a_last_frames = full_array_a[-frames:]

	person_b_first_frames = full_array_b[:frames]
	person_b_last_frames = full_array_b[-frames:]

	first_frames_norm = []
	for v_a, v_b in (zip(person_a_first_frames,person_b_first_frames)):
		first_frames_norm.append(np.linalg.norm(v_a-v_b))

	last_frames_norm = []
	for v_a, v_b in (zip(person_a_last_frames,person_b_last_frames)):
		last_frames_norm.append(np.linalg.norm(v_a-v_b))

	return np.average(np.asarray(first_frames_norm)) - np.average(np.asarray(last_frames_norm))


def MOVDISTR(full_array_a,full_array_b):

	direction = []
	for i in range(0,len(full_array_a)-1):
		distance = np.linalg.norm(full_array_a[i]-full_array_a[i+1])
		direction.append(distance*getAngle(full_array_a[i]-full_array_a[i+1],full_array_a[i]-full_array_b[i]))

	freqs,bins = np.histogram(np.asarray(direction), bins=np.arange(0,180))
	mids = 0.5*(bins[1:] + bins[:-1])
	mean = np.average(mids, weights=freqs)
	var = np.average((mids - mean)**2, weights=freqs)
	std = np.sqrt(var)

	# print(mean, np.sqrt(var))
	# plt.hist(freqs,bins)
	# plt.show()

	return mean, std

def MOTIONSYNC_MOTION_REACTION(full_array_a,full_array_b):

	whole_windows_a = []
	whole_windows_b = []
	for i in range(0,len(full_array_a)-30):
		# print(i)
		accumulate_a = 0
		accumulate_b = 0

		for j in range(0,30):

			accumulate_a += np.linalg.norm(full_array_a[i+j+1] - full_array_a[i+j])
			accumulate_b += np.linalg.norm(full_array_b[i+j+1] - full_array_b[i+j])

		whole_windows_a.append(accumulate_a)
		whole_windows_b.append(accumulate_b)

	whole_windows_a_np = np.asarray(whole_windows_a)
	whole_windows_b_np = np.asarray(whole_windows_b)

	H, bins = np.histogram(whole_windows_a_np, bins=4)
	# print(H)
	# plt.plot(H)
	# plt.show()

	return H,np.average(whole_windows_a_np),np.average(whole_windows_b_np)

def MOTION_REACTION(full_array_a,full_array_b):

	windows = []
	for i in range(0,len(full_array_a)-30):

		windows_accum = 0
		for j in range(0,30):
			windows_accum += np.linalg.norm(full_array_a[i+j+1] - full_array_b[i+j])

	return windows/len(np.arange(0,len(full_array_a)-30,30))

def VARPOS(full_array_a,full_array_b):

	varpos_a = []
	varpos_b = []	
	for v_a,v_b in zip(full_array_a,full_array_b):
		varpos_a.append(np.linalg.norm(v_a))
		varpos_b.append(np.linalg.norm(v_b))

	return np.var(varpos_a),np.var(varpos_b)

if __name__ == "__main__":

	## get all dir names for each day
	dataset_dir = "../dataset/"
	day_1_dir = dataset_dir + "train" + "/day_1"
	day_2_dir = dataset_dir + "train" + "/day_2"
	day_3_dir = dataset_dir + "test" + "/day_3"

	day_1_orig_dir = "../spliter/individual_videos/day_1/"
	day_2_orig_dir = "../spliter/individual_videos/day_2/"
	day_3_orig_dir = "../spliter/individual_videos/day_3/"

	## get all dates from each day
	day_1_dates = [os.path.join(day_1_dir, d) for d in os.listdir(day_1_dir)]
	day_2_dates = [os.path.join(day_2_dir, d) for d in os.listdir(day_2_dir)]
	day_3_dates = [os.path.join(day_3_dir, d) for d in os.listdir(day_3_dir)]

	## get tables centers files
	day_1_tables = find_csv_filenames(day_1_orig_dir)
	day_2_tables = find_csv_filenames(day_2_orig_dir)
	day_3_tables = find_csv_filenames(day_3_orig_dir)

	## 
	pairwise_data_reponse_file = "../pairwise_date_response.csv"

	#### For day 1
	# day_1_dates = ["../dataset/train/day_1/11382103"]


	dataset_file = open(dataset_dir+ "dataset.csv", "a")

	dataset_file.write("date,avg_difangle,var_difangle,avgdis,vardis,decardis,movdistr_mean_a,movdistr_var_a,movdistr_mean_b,movdistr_var_b,motionsync_a_1,motionsync_a_2,motionsync_a_3,motionsync_a_4,motion_reaction_a,motion_reaction_b,varpos_a,varpos_b,M_1,M_2,M_3,M_4,M_5,M_6,PP_M_1,PP_M_2,PP_M_3,PP_M_4,PP_M_5,PP_M_6,"+"\n")

	list_of_days = {"day1": day_1_dates, "day2": day_2_dates, "day3": day_3_dates}
	# list_of_days = {"day2": day_2_dates, "day3": day_3_dates}

	for key_day in list_of_days.keys():
		print("INFO: Working on day: ",key_day)
		for date in list_of_days.get(key_day):

			for file in find_csv_filenames(date):
				
				go=False

				if fnmatch.fnmatch(file, '*M_participant*'):

					date_number = str(file.split(sep="/")[4] + "_M")
					person_a = file

					if len(glob.glob(os.path.dirname(file)+"/F_participant*.csv")) > 0:
						person_b = glob.glob(os.path.dirname(file)+"/F_participant*.csv")[0]
						go = True

					ground_thr = os.popen("grep " + str(file.split(sep="/")[4]) + " " + str(pairwise_data_reponse_file) + "| awk -F',' '{if ($13 == 1) print $0}'" ).read()

					ground_thr_a = ground_thr.split(sep=",")[6:12]
					ground_thr_b = ground_thr.split(sep=",")[14:20]

				elif fnmatch.fnmatch(file, '*F_participant*'):

					date_number = str(file.split(sep="/")[4] + "_F")
					person_a = file

					if len(glob.glob(os.path.dirname(file)+"/M_participant*.csv")) > 0:
						person_b = glob.glob(os.path.dirname(file)+"/M_participant*.csv")[0]
						go = True

					ground_thr = os.popen("grep " + str(file.split(sep="/")[4]) + " " + str(pairwise_data_reponse_file) + "| awk -F',' '{if ($13 == 2) print $0}'" ).read()

					ground_thr_a = ground_thr.split(sep=",")[6:12]
					ground_thr_b = ground_thr.split(sep=",")[14:20]

				if go:
					print(file)
					if key_day == "day1":
						table_center_person_a = get_table_center(person_a,day_1_orig_dir,day_1_tables)
						table_center_person_b = get_table_center(person_b,day_1_orig_dir,day_1_tables)
					elif key_day == "day2":
						table_center_person_a = get_table_center(person_a,day_2_orig_dir,day_2_tables)
						table_center_person_b = get_table_center(person_b,day_2_orig_dir,day_2_tables)
					elif key_day == "day3":
						table_center_person_a = get_table_center(person_a,day_3_orig_dir,day_3_tables)
						table_center_person_b = get_table_center(person_b,day_3_orig_dir,day_3_tables)

					centers_array_person_a = read_centers_from_csv(person_a)
					full_array_person_a,norm_person_a = interpolate(centers_array_person_a)

					centers_array_person_b = read_centers_from_csv(person_b)
					full_array_person_b,norm_person_b = interpolate(centers_array_person_b)			

					### Get features
					avg_difangle, var_difangle = VAR_AVG_DIFANGLE(table_center_person_a,centers_array_person_a)
					avgdis, vardis = AVG_VARDIS(table_center_person_a,centers_array_person_a)
					decardis = DECRDIS(full_array_person_a,full_array_person_b,12.5)
					movdistr_mean_a,movdistr_var_a = MOVDISTR(full_array_person_a,full_array_person_b)
					movdistr_mean_b,movdistr_var_b = MOVDISTR(full_array_person_b,full_array_person_a)
					motionsync, motion_reaction_a, motion_reaction_b = MOTIONSYNC_MOTION_REACTION(full_array_person_a,full_array_person_b)
					# motion_reaction = MOTION_REACTION(full_array_person_a,full_array_person_b)
					varpos_a,varpos_b = VARPOS(full_array_person_a,full_array_person_b)

					# print(varpos_a,varpos_b, sep=",")
					motionsync_string = np.array2string(motionsync, precision=2, separator=',', max_line_width=1000)
					# motion_reaction_string = np.array2string(motion_reaction, precision=2, separator=',', max_line_width=1000)
					ground_thr_a_string = np.array2string(np.array(ground_thr_a,dtype=int), precision=2, separator=',', max_line_width=1000)
					ground_thr_b_string = np.array2string(np.array(ground_thr_b,dtype=int), precision=2, separator=',', max_line_width=1000)

					dataset_file.write(str(date_number)+","+str(avg_difangle)+","+str(var_difangle)+","+str(avgdis)+","+str(vardis)+","+str(decardis)+","+str(movdistr_mean_a)+","+str(movdistr_var_a)+","+str(movdistr_mean_b)+","+str(movdistr_var_b)+","+str(motionsync_string[1:-1])+","+str(motion_reaction_a)+","+str(motion_reaction_b)+","+str(varpos_a)+","+str(varpos_b)+","+str(ground_thr_a_string[1:-1])+","+str(ground_thr_b_string[1:-1])+"\n")

	dataset_file.close()

	# plt.plot(full_array[:,0], label='x movement')
	# plt.plot(full_array[:,1], label='y movement')
	# plt.plot(norm, label='norm')
	# plt.legend()
	# plt.show()



