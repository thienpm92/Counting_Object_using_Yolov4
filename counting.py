import os
import sys
import cv2
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tool.yolo_model import Yolov4
from tool.utils import *
from tool.torch_utils import *







if __name__ == "__main__":
	#input parameter
	if len(sys.argv) == 5:
		vid_name = sys.argv[1]
		model_type = sys.argv[2]
		height = int(sys.argv[3])
		width = int(sys.argv[4])
	else:
		print('Usage: ')
		print('python counting.py vid_name model_type H W')


	#load and init model
	if model_type == 'yolov4':
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model_path = './models/yolov4/yolov4.pth'
	elif model_type == 'ssd':
		prototype_path = 'models/mobileNetSSD/MobileNetSSD_deploy.prototxt'
		model_path = 'models/mobileNetSSD/MobileNetSSD_deploy.caffemodel'

	try:
		if model_type == 'yolov4':
			print("[INFO] Loading model yolov4")
			model = Yolov4(yolov4conv137weight=None, n_classes=80, inference=True)
			pretrained_dict = torch.load(model_path, map_location=device)
			model.load_state_dict(pretrained_dict)
			if device == torch.device('cuda'):
				use_cuda = True
				model.cuda()
			else:
				use_cuda = False
		elif model_type == 'ssd':
			print("[INFO] Loading model ssd")
			model = cv2.dnn.readNetFromCaffe(prototype_path, model_path)
	except:
		print("Load model fail, please check the model_path")

	
	#init parameters
	object_type = 'car'
	tracking_mode = 0
	max_distance = 50
	laser_line = height - 100
	frame_count = 0
	object_number = 0
	obj_cnt = 0
	curr_trackers = []
	namesfile = 'data/coco.names'
	record_vid = False

	#init record video
	if record_vid:
		video_size = (width,height)
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		writer = cv2.VideoWriter("vid_res.avi",fourcc,30,video_size)
	vid = cv2.VideoCapture(vid_name)
	
	while vid.isOpened():
		_, frame = vid.read()
		if frame is None:
			break
		frame = cv2.resize(frame, (width, height))

		# Duyet qua cac doi tuong trong tracker
		old_trackers = curr_trackers
		curr_trackers = []
		laser_line_color = (0, 0, 255)
		track_boxes = []


		# Do object detection each 5 frame
		if frame_count % 5 == 0:
			if model_type == 'yolov4': 
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				class_names = load_class_names(namesfile)
				objects = detect_yolo(model, frame, 0.4, 0.6, object_type, class_names,use_cuda)
			else:
				objects = detect_ssd(model, frame, object_type, conf_threshold=0.5, h=height, w=width)


			for obj in objects:
				xd, yd, wd, hd, center_Xd, center_Yd = get_box_info(obj)

				if center_Yd < laser_line:
					#frame = draw_object(frame,xd, yd, wd, hd, cls_obj, class_names)
					#generate new tracker
					if tracking_mode == 0:
						tracker = cv2.TrackerKCF_create()
					elif tracking_mode == 1:
						tracker = cv2.TrackerCSRT_create()
					else:
						tracker = cv2.TrackerMOSSE_create()
					obj_cnt += 1
					new_obj = dict()
					bx = [xd, yd, wd, hd]
					tracker.init(frame, tuple(bx))

					new_obj['tracker_id'] = obj_cnt
					new_obj['tracker'] = tracker
					curr_trackers.append(new_obj)
				elif center_Yd >= laser_line and center_Yd <= laser_line:
					# If it across the line, not tracking it
					laser_line_color = (0, 255, 255)
					object_number += 1
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		else:
			for obj_tracker in old_trackers:
				# Update tracker
				tracker = obj_tracker['tracker']
				check, box = tracker.update(frame)

				if check == True:
					track_boxes.append(box)

					new_obj = dict()
					new_obj['tracker_id'] = obj_tracker['tracker_id']
					new_obj['tracker'] = tracker

					# Get obj info and draw
					x, y, w, h, center_X, center_Y = get_box_info(box)
					cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
					cv2.circle(frame, (center_X, center_Y), 4, (0, 255, 0), -1)

					# Compare obj with laser line
					if center_Y >= laser_line-10:
						# If it across the line, not tracking it
						laser_line_color = (0, 255, 255)
						object_number += 1
					else:
						# Otherwise continue track
						curr_trackers.append(new_obj)

		frame_count += 1
		# Display
		cv2.putText(frame, object_type + " number: " + str(object_number), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255 , 0), 2)
		cv2.putText(frame, "Press Esc to quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

		# Draw laser line
		cv2.line(frame, (0, laser_line), (width, laser_line), laser_line_color, 2)
		cv2.putText(frame, "Laser line", (10, laser_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, laser_line_color, 2)

		# Frame
		
		cv2.imshow("Image", frame)
		if record_vid:
			writer.write(frame)
		key = cv2.waitKey(1) & 0xFF
		if key == 27:
			break
	vid.release()
	cv2.destroyAllWindows


