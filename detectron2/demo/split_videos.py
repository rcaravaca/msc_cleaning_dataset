# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
import re

from progress.bar import FillingSquaresBar

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"

OFFSET = 170
FRAMES_TO_CHECK_CENTERS = 1800

def setup_cfg(args):
	# load config from file and command-line arguments
	cfg = get_cfg()
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	# Set score_threshold for builtin models
	cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
	cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
	cfg.freeze()
	return cfg

def get_parser():
	parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
	parser.add_argument(
		"--config-file",
		default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
		metavar="FILE",
		help="path to config file",
	)
	parser.add_argument("--video-input", help="Path to video file.")
	
	parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    
	parser.add_argument(
		"--output",
		help="A file or directory to save output visualizations. "
		"If not given, will show output in an OpenCV window.",
	)

	parser.add_argument(
		"--confidence-threshold",
		type=float,
		default=0.5,
		help="Minimum score for instance predictions to be shown",
	)
	parser.add_argument(
		"--opts",
		help="Modify config options using the command-line 'KEY VALUE' pairs",
		default=[],
		nargs=argparse.REMAINDER,
	)
	return parser

def get_bbox(frame):

	predictions, visualized_output = demo.run_on_image(frame)

	[rows,cols,channel] = frame.shape
	centers = []
	for box in predictions["instances"].pred_boxes:
		bbox = box.numpy()
		center = np.array( [int(bbox[1]+(bbox[3]-bbox[1])//2), int(bbox[0]+(bbox[2]-bbox[0])//2)] )
		
		if center[0] > OFFSET and center[0] < rows - OFFSET and center[1] > OFFSET and center[1] < cols - OFFSET:
			centers.append(center)
		
	return centers
	
def get_bbox_id(old_centers,new_centers):
	
	centers_id = old_centers
	
	if len(old_centers) == 0:
		
		centers_id = {}
		for ID in range(0,len(new_centers)):
			centers_id.update({ID: new_centers[ID]})
	else:
		
		for point in centers:
			dist_aux = 100
			for k in old_centers.keys():
				dist = np.linalg.norm(point - old_centers[k])

				if dist < dist_aux:
					centers_id.update({k: point})
					dist_aux = dist
	
	return centers_id

def cut_frames(img,centers,delta,counting_frames_split):
	
	label = 0
	image_list = {}
	for ID in centers.keys():
		
		delt = delta.get(ID)*counting_frames_split
		image_list.update({ID:img[centers[ID][0]-OFFSET+round(delt[0]):centers[ID][0]+OFFSET+round(delt[0]),centers[ID][1]-OFFSET+round(delt[1]):centers[ID][1]+OFFSET+round(delt[1])]})
	
	return image_list

def cut_frames2(img,centers):
	
	label = 0
	image_list = {}
	for ID in centers.keys():
		
		image_list.update({ID:img[centers[ID][0]-OFFSET:centers[ID][0]+OFFSET,centers[ID][1]-OFFSET:centers[ID][1]+OFFSET]})
	
	return image_list

def elapsed_time(start,end):
	hours, rem = divmod(end-start, 3600.0)
	minutes, seconds = divmod(rem, 60.0)
	print("INFO ==> Elapsed time : {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)
	args = get_parser().parse_args()
	setup_logger(name="fvcore")
	logger = setup_logger()
	logger.info("Arguments: " + str(args))

	cfg = setup_cfg(args)
	demo = VisualizationDemo(cfg)
	old_centers = []
	
	print(args.output)
	
	if not args.output:
		print("ERROR ==> Specify output dir")


	if args.input:

		m = re.search('Speed_date_part.*cam..', args.input[0])
		video_name = m.group(0)

		video = cv2.VideoCapture(args.input[0])
		width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
		frames_per_second = video.get(cv2.CAP_PROP_FPS)
		num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
		basename = os.path.basename(args.input[0])
		
		print("INFO ==> Frames count: ",num_frames)
		print("INFO ==> Frames rate: ",frames_per_second)
		print("INFO ==> Frames size: ",height,"x",width)

		##################################################################################################			
		# 	Getting centers
		##################################################################################################			
		counting_frames = 0
		counting_images = 0
		
		all_centers = []
		
		bar = FillingSquaresBar('INFO ==> Getting centers ...', max=num_frames)
		start_time = time.time()
		while(video.isOpened()):
			flag, frame = video.read()
			
			if flag and (counting_frames % FRAMES_TO_CHECK_CENTERS == 0 or counting_frames == num_frames-1):
				
				current_time = time.time()
				centers = get_bbox(frame)
					
				new_centers = get_bbox_id(old_centers,centers)
				all_centers.append(new_centers.copy())
				
				image_list = cut_frames2(frame,new_centers)
				
				for k in image_list.keys():
					cv2.imwrite(str(args.output) + "/" + str(video_name) + "_" + str(k) + "_" + str(counting_images) + ".png", image_list.get(k))
					
				counting_images = counting_images + 1
			else:
				if counting_frames >= video.get(cv2.CAP_PROP_FRAME_COUNT):
					break
					
			old_centers = new_centers
			counting_frames = counting_frames + 1
			bar.next()
		bar.next()
		finish_time = time.time()
		
		print("\nINFO ==> Getting centers done")
		elapsed_time(start_time,finish_time)
		video.release()

		##################################################################################################			
		# 	Getting delta to interpolate
		##################################################################################################	
		# ~ delta_zeros = {}
		# ~ delta = {}
		# ~ all_deltas = []
		# ~ for i in range(1,len(all_centers)):
			# ~ for k in all_centers[i].keys():
				# ~ delta_zeros.update({k: np.array([0,0])})
				# ~ delta.update({k:(all_centers[i].get(k)-all_centers[i-1].get(k))/FRAMES_TO_CHECK_CENTERS})
			
			# ~ all_deltas.append(delta.copy())
		
		# ~ ##################################################################################################			
		# ~ # 	Slpiting videos
		# ~ ##################################################################################################	
		
		# ~ output_video = [0]*10
		# ~ fourcc = cv2.VideoWriter_fourcc(*'XVID')
		# ~ for i in range(0,10):
			# ~ output_video[i] = cv2.VideoWriter("individual_videos_2/person_" + str(i) + ".avi",cv2.CAP_FFMPEG,fourcc, frames_per_second, (OFFSET*2,OFFSET*2),	1)

		# ~ counting_frames = 0
		# ~ counting_frames_split = 1
		# ~ delta_idx = -1
		# ~ bar = FillingSquaresBar('INFO ==> Making videos ...', max=num_frames)
		# ~ start_time = time.time()
		# ~ video = cv2.VideoCapture(args.input[0])
		# ~ while(video.isOpened()):
			# ~ flag, frame = video.read()
			
			# ~ if flag and counting_frames % FRAMES_TO_CHECK_CENTERS == 0:
				
				# ~ delta_idx = delta_idx + 1
				# ~ counting_frames_split = 1
				# ~ image_list = cut_frames(frame,all_centers[delta_idx],delta_zeros,counting_frames_split)
				
			# ~ else:
				# ~ if counting_frames >= video.get(cv2.CAP_PROP_FRAME_COUNT):
					# ~ break
			
				# ~ counting_frames_split = counting_frames_split + 1
				# ~ image_list = cut_frames(frame,all_centers[delta_idx],all_deltas[delta_idx],counting_frames_split)
				
			# ~ for image_id in image_list.keys():
				# ~ output_video[image_id].write(image_list[image_id])

			# ~ counting_frames = counting_frames + 1
			
			# ~ bar.next()
		# ~ bar.next()
		# ~ finish_time = time.time()
		
		# ~ print("\nINFO ==> Making videos done")
		# ~ elapsed_time(start_time,finish_time)
	
	# ~ video.release()
	# ~ for i in range(0,10):
		# ~ output_video[i].release()

		
			
		
			

