#!/usr/bin/env python3

import argparse
import glob
import os, fnmatch
import numpy as np
import re
import matplotlib.pyplot as plt
import veenstra_features as vf
from scipy import stats, signal
from scipy import signal as sg
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import normalized_mutual_info_score
import inspect
import feature_extraction as fe
from datetime import date, datetime

def replaceZeroes(data):
	if np.nonzero(data)[0].size != 0:
		min_nonzero = np.min(data[np.nonzero(data)])
		data[data == 0] = min_nonzero
	return data

def get_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

def read_accel_from_csv(csv_file):

	f=open(csv_file,"r")

	lines=f.readlines()
	result_triaxial = []
	result_norm = []

	# box = np.array([0,0])
	# starting_frame = int(lines[0].split(",")[0])

	for i in lines:
		x = float(i.split(sep=",")[1]) 
		y = float(i.split(sep=",")[2])
		z = float(i.split(sep=",")[3])
		result_triaxial.append(np.array([x,y,z]))
		result_norm.append(np.linalg.norm(np.array([x,y,z])))
	
	result_triaxial_asarray = np.asarray(result_triaxial)

	f.close()

	if np.linalg.norm(result_triaxial_asarray) > 0 and result_triaxial_asarray.shape[0] == 3000:
		x_zscore = stats.zscore(result_triaxial_asarray[:,0])
		y_zscore = stats.zscore(result_triaxial_asarray[:,1])
		z_zscore = stats.zscore(result_triaxial_asarray[:,2])
		
		magnitud = []
		for x,y,z in zip(x_zscore,y_zscore,z_zscore):
			magnitud.append(np.linalg.norm([x,y,z]))

		# print(np.abs(magnitud))
		return x_zscore,y_zscore,z_zscore,np.abs(x_zscore),np.abs(y_zscore),np.abs(z_zscore),np.asarray(magnitud)

	else:
		return 0,0,0,0,0,0,0
	


def get_pds(signal):
	
	f, Pxx_spec = sg.welch(signal, 20, nperseg=len(signal))
	# print(signal)
	# print(Pxx_spec)
	Pxx_spec_no_zeros = replaceZeroes(Pxx_spec)

	Pxx_spec_log = Pxx_spec_no_zeros
	if Pxx_spec.sum() > 0:
		Pxx_spec_log = np.log(Pxx_spec_no_zeros)

	freqs,bins = np.histogram(Pxx_spec_log, bins=6)
	# print(freqs)

	# plt.figure()
	# plt.plot(freqs)
	# plt.xlabel('frequency [Hz]')
	# plt.ylabel('Linear spectrum [V RMS]')
	# plt.title('Power spectrum (scipy.sg.welch)')
	# plt.show()

	return np.array(freqs)

def get_complex_features(signal_a,signal_b,low_features_a,low_features_b):

	tau = 1

	person_corr = np.zeros(8)
	mutual_info = np.zeros(8)
	lagged_correlation = np.zeros(8)

	for i in np.arange(0,8):
		person_corr[i],p = stats.pearsonr(low_features_a[:,i], low_features_b[:,i])
		#mutual_info[i] = normalized_mutual_info_score(low_features_a[:,i], low_features_b[:,i])
		mutual_info[i] = fe.normalized_mutual_information(low_features_a[:,i], low_features_b[:,i])
		lagged_correlation[i],p = stats.pearsonr(low_features_a[:-tau,i],low_features_b[tau:,i])

	######### mimicry ##################

	mimicry_feat = []
	sym_conv_corr_b = []
	# for lf in np.arange(0,low_features_a.shape[1]):
	D_mimicry = []
	D_sym_conv = []
	for i in np.arange(0,low_features_a.shape[0]-1):
		D_mimicry.append(np.linalg.norm(low_features_a[i,:] - low_features_b[i+1,:]))
		D_sym_conv.append(np.linalg.norm(low_features_a[i,:] - low_features_b[i,:]))

	mimicry = np.asarray(D_mimicry)
	mimicry_feat = [mimicry.min(),mimicry.max(),mimicry.mean(),mimicry.std()]

	######### symetric convergence ##################
	sym_conv = np.asarray(D_sym_conv)
	sym_conv_corr,p = stats.pearsonr(np.arange(len(sym_conv)), sym_conv)

	######### Asymetric convergence ##################

	# Person A
	mean = signal_a[:100*20].mean()
	std = signal_a[:100*20].std()
	pds = get_pds(signal_a[:100*20])
	low_features = np.array([mean, std, pds[0],pds[1],pds[2],pds[3],pds[4],pds[5]])

	D = []
	last_min_windows = 2*(low_features_b.shape[0]//3)

	for i in np.arange(last_min_windows,low_features_b.shape[0]):
		D.append(np.linalg.norm(low_features - low_features_b[i,:]))

	asym_conv_a = np.asarray(D)
	asym_conv_corr_b,p = stats.pearsonr(asym_conv_a, np.arange(len(asym_conv_a)))


	# Person B
	mean = signal_b[:last_min_windows*30].mean()
	std = signal_b[:last_min_windows*30].std()
	pds = get_pds(signal_b[:last_min_windows*30])
	low_features = np.array([mean, std, pds[0],pds[1],pds[2],pds[3],pds[4],pds[5]])

	D = []
	for i in np.arange(last_min_windows,low_features_a.shape[0]):
		D.append(np.linalg.norm(low_features - low_features_a[i,:]))

	asym_conv_b = np.asarray(D)
	asym_conv_corr_a,p = stats.pearsonr( asym_conv_b, np.arange(len(asym_conv_b)))

	######### Global convergence ##################

	mean_a_1 = signal_a[:75*30].mean()
	std_a_1 = signal_a[:75*30].std()
	pds_a_1 = get_pds(signal_a[:75*30])

	mean_a_2 = signal_a[75*30:].mean()
	std_a_2 = signal_a[75*30:].std()
	pds_a_2 = get_pds(signal_a[75*30:])

	low_features_a_1 = np.array([mean_a_1, std_a_1, pds_a_1[0],pds_a_1[1],pds_a_1[2],pds_a_1[3],pds_a_1[4],pds_a_1[5]])
	low_features_a_2 = np.array([mean_a_2, std_a_2, pds_a_2[0],pds_a_2[1],pds_a_2[2],pds_a_2[3],pds_a_2[4],pds_a_2[5]])

	mean_b_1 = signal_b[:75*30].mean()
	std_b_1 = signal_b[:75*30].std()
	pds_b_1 = get_pds(signal_b[:75*30])

	mean_b_2 = signal_b[75*30:].mean()
	std_b_2 = signal_b[75*30:].std()
	pds_b_2 = get_pds(signal_b[75*30:])

	low_features_b_1 = np.array([mean_b_1, std_b_1, pds_b_1[0],pds_b_1[1],pds_b_1[2],pds_b_1[3],pds_b_1[4],pds_b_1[5]])
	low_features_b_2 = np.array([mean_b_2, std_b_2, pds_b_2[0],pds_b_2[1],pds_b_2[2],pds_b_2[3],pds_b_2[4],pds_b_2[5]])


	d0 = np.linalg.norm(low_features_a_1 - low_features_b_1)
	d1 = np.linalg.norm(low_features_a_2 - low_features_b_2)

	global_conv = np.linalg.norm(d1 - d0)

	##################################################

	all_features = np.hstack((person_corr,mutual_info,mimicry_feat,lagged_correlation,sym_conv_corr,asym_conv_corr_a,asym_conv_corr_b,global_conv))

	return all_features

def get_low_level_features(signal,w, Hz=20):

	low_features = []
	for i in np.arange(0,len(signal)-w*Hz,(w*Hz)//2):
		pds = get_pds(signal[i:i+w*Hz])
		low_features.append([signal[i:i+w*Hz].mean(), signal[i:i+w*Hz].std(), pds[0],pds[1],pds[2],pds[3],pds[4],pds[5]])

	return np.asarray(low_features)

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
	day_1_tables = vf.find_csv_filenames(day_1_orig_dir)
	day_2_tables = vf.find_csv_filenames(day_2_orig_dir)
	day_3_tables = vf.find_csv_filenames(day_3_orig_dir)

	## 
	pairwise_data_reponse_file = "../pairwise_date_response.csv"

	now = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

	dataset_file = open(dataset_dir + "dataset_werables__" + now +".csv", "a")
	print("INFO: dataset file: ",dataset_file.name)
	# print(dataset_file.name)
	# dataset_file.close()
	# exit()
	# dataset_file.write("date,avg_difangle,var_difangle,avgdis,vardis,decardis,movdistr_mean_a,movdistr_var_a,movdistr_mean_b,movdistr_var_b,motionsync_a_1,motionsync_a_2,motionsync_a_3,motionsync_a_4,motion_reaction_a_1,motion_reaction_a_2,motion_reaction_a_3,motion_reaction_a_4,motion_reaction_a_5,motion_reaction_a_6,motion_reaction_a_7,motion_reaction_a_8,motion_reaction_a_9,motion_reaction_a_10,motion_reaction_a_11,motion_reaction_a_12,motion_reaction_a_13,motion_reaction_a_14,motion_reaction_a_15,motion_reaction_a_16,motion_reaction_a_17,motion_reaction_a_18,motion_reaction_a_19,motion_reaction_a_20,motion_reaction_a_21,motion_reaction_a_22,motion_reaction_a_23,motion_reaction_a_24,motion_reaction_a_25,motion_reaction_a_26,motion_reaction_a_27,motion_reaction_a_28,motion_reaction_a_29,motion_reaction_a_30,varpos_a,varpos_b,M_1,M_2,M_3,M_4,M_5,M_6,PP_M_1,PP_M_2,PP_M_3,PP_M_4,PP_M_5,PP_M_6,"+"\n")

	list_of_days = {"day1": day_1_dates, "day2": day_2_dates, "day3": day_3_dates}
	# list_of_days = {"day2": day_2_dates}

	features_type_name = ["person_corr_mean","person_corr_std","person_corr_pds_0","person_corr_pds_1","person_corr_pds_2","person_corr_pds_3","person_corr_pds_4","person_corr_pds_5","mutual_info_mean","mutual_info_sdt","mutual_info_pds_0","mutual_info_pds_1","mutual_info_pds_2","mutual_info_pds_3","mutual_info_pds_4","mutual_info_pds_5","mimicry_min","mimicry_max","mimicry_mean","mimicry_std","lagged_correlation_mean","lagged_correlation_std","lagged_correlation_pds_0","lagged_correlation_pds_1","lagged_correlation_pds_2","lagged_correlation_pds_3","lagged_correlation_pds_4","lagged_correlation_pds_5","sym_conv_corr_b","asym_conv_corr_a","asym_conv_corr_b","global_conv"] 

	signal_types = ["x","y","z","abs_x","abs_y","abs_z","magnitud"]

	headers = []
	for sig_type in signal_types:
		for feat_name in features_type_name:
			headers.append(sig_type +"_"+ feat_name)


	# signal_types = ["x","y","z","abs_x","abs_y","abs_z","magnitud"]
	# low_features = ["mean", "std", "psd_0", "psd_1", "psd_2", "psd_3", "psd_4", "psd_5"]
	# complex_features = ["correlation", "mutual_info", "mimicry", "lagged_correlation", "sym_convergen", "aym_convergence", "global_convergence"]

	# headers = []
	# for sig in signal_types: 
	# 	for lf in low_features: 
	# 		for cf in complex_features: 
	# 			headers.append(sig+"_"+lf+"_"+cf) 

	dataset_file.write("date,"+",".join(headers)+",M_1,M_2,M_3,M_4,M_5,M_6,PP_M_1,PP_M_2,PP_M_3,PP_M_4,PP_M_5,PP_M_6\n")

	for key_day in list_of_days.keys():
		print("INFO: Working on day: ",key_day)
		for date in list_of_days.get(key_day):

			for file in vf.find_csv_filenames(date):
				
				# file = "../dataset/train/day_2/12242206/M_accel_Date_8_Participant_24.csv"
				go=False
				if fnmatch.fnmatch(file, '*M_accel*') and os.path.exists(file):

					date_number = str(file.split(sep="/")[4] + "_M")
					person_a = file

					if len(glob.glob(os.path.dirname(file)+"/F_accel*.csv")) > 0:
						person_b = glob.glob(os.path.dirname(file)+"/F_accel*.csv")[0]

						if os.path.exists(person_b):
							x_zscore_a,y_zscore_a,z_zscore_a,abs_x_zscore_a,abs_y_zscore_a,abs_z_zscore_a,magnitud_a = read_accel_from_csv(person_a)
							x_zscore_b,y_zscore_b,z_zscore_b,abs_x_zscore_b,abs_y_zscore_b,abs_z_zscore_b,magnitud_b = read_accel_from_csv(person_b)
							if np.linalg.norm(magnitud_a) > 0 and np.linalg.norm(magnitud_b) > 0:
								go = True

					ground_thr = os.popen("grep " + str(file.split(sep="/")[4]) + " " + str(pairwise_data_reponse_file) + "| awk -F',' '{if ($13 == 1) print $0}'" ).read()


				elif fnmatch.fnmatch(file, '*F_accel*') and os.path.exists(file):

					date_number = str(file.split(sep="/")[4] + "_F")
					person_a = file

					if len(glob.glob(os.path.dirname(file)+"/M_accel*.csv")) > 0:
						person_b = glob.glob(os.path.dirname(file)+"/M_accel*.csv")[0]

						if os.path.exists(person_b):
							x_zscore_a,y_zscore_a,z_zscore_a,abs_x_zscore_a,abs_y_zscore_a,abs_z_zscore_a,magnitud_a = read_accel_from_csv(person_a)
							x_zscore_b,y_zscore_b,z_zscore_b,abs_x_zscore_b,abs_y_zscore_b,abs_z_zscore_b,magnitud_b = read_accel_from_csv(person_b)
							if np.linalg.norm(magnitud_a) > 0 and np.linalg.norm(magnitud_b) > 0:
								go = True

					ground_thr = os.popen("grep " + str(file.split(sep="/")[4]) + " " + str(pairwise_data_reponse_file) + "| awk -F',' '{if ($13 == 2) print $0}'" ).read()


				if go:
					print(date_number,person_a)
					x_zscore_a,y_zscore_a,z_zscore_a,abs_x_zscore_a,abs_y_zscore_a,abs_z_zscore_a,magnitud_a = read_accel_from_csv(person_a)
					x_zscore_b,y_zscore_b,z_zscore_b,abs_x_zscore_b,abs_y_zscore_b,abs_z_zscore_b,magnitud_b = read_accel_from_csv(person_b)
					signals_a = [x_zscore_a,y_zscore_a,z_zscore_a,abs_x_zscore_a,abs_y_zscore_a,abs_z_zscore_a,magnitud_a]
					signals_b = [x_zscore_b,y_zscore_b,z_zscore_b,abs_x_zscore_b,abs_y_zscore_b,abs_z_zscore_b,magnitud_b]


					window_size = 10
					dataset_file.write(str(date_number)+",")
					all_features = []
					# print("INFO: Window size: ",window_size)
					for sig_a,sig_b in zip(signals_a,signals_b):
						low_features_a = get_low_level_features(sig_a,window_size)
						low_features_b = get_low_level_features(sig_b,window_size)

						all_features = get_complex_features(sig_a,sig_b,low_features_a,low_features_b)

						all_features_str = np.array2string(np.array(all_features,dtype=float), precision=4, separator=',', max_line_width=1000)
						dataset_file.write(str(all_features_str[1:-1])+",")

					# all_features_0 = np.array2string(np.array(all_features[0],dtype=float), precision=2, separator=',', max_line_width=1000)
					# all_features_1 = np.array2string(np.array(all_features[1],dtype=float), precision=2, separator=',', max_line_width=1000)
					# all_features_2 = np.array2string(np.array(all_features[2],dtype=float), precision=2, separator=',', max_line_width=1000)
					# all_features_3 = np.array2string(np.array(all_features[3],dtype=float), precision=2, separator=',', max_line_width=1000)
					

					ground_thr_a = ground_thr.split(sep=",")[6:12]
					ground_thr_b = ground_thr.split(sep=",")[14:20]
					ground_thr_a_string = np.array2string(np.array(ground_thr_a,dtype=int), precision=2, separator=',', max_line_width=1000)
					ground_thr_b_string = np.array2string(np.array(ground_thr_b,dtype=int), precision=2, separator=',', max_line_width=1000)
					
					dataset_file.write(str(ground_thr_a_string[1:-1])+","+str(ground_thr_b_string[1:-1])+"\n")

					# exit()
	print("INFO: dataset done: ",dataset_file.name)
	dataset_file.close()