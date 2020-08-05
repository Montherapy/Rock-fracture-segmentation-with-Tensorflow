# coding: utf-8

import os
import cv2

"""
Preprocessing the raw image in the image folder which has been made by using Labelme annotation tool.

	ref : https://github.com/wkentaro/labelme

"""

folder_names = "images/"
img_path = [folder_names+"img_label_%d/img.png" %i for i in range(1,24)]

for idx, im_path in enumerate(img_path):

	# read in image and input gridsize for clahe
	bgr = cv2.imread(im_path)
	if bgr.shape[0] != 1100 :
		bgr = cv2.resize(bgr, dsize=(1100,1100),  interpolation=cv2.INTER_AREA)

	grid = 20
	hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
	hsv_split = cv2.split(hsv)
	clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(grid, grid))
	hsv_split[2] = clahe.apply(hsv_split[2])
	hsv = cv2.merge(hsv_split)
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	new_path = folder_names+'/img_he_%d.png'%(idx+1)
	cv2.imwrite(new_path, bgr)
