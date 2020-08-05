# coding: utf-8

import numpy as np
import PIL.Image as im
import os
import cv2

data_path = './images/'
data_type = "test" # or "train"

size_raw = 1100
size_trim = 1024
size_crop = 256
size_randomcrop = 224

def flip(pic):
	return pic, np.flip(pic,0), np.flip(pic,1), np.flip(np.flip(pic,0),1)

def crop_and_put(im_1024, la_1024, n):
	for j in range(n):
		for k in range(n):
			# crop into 1024 size to get rid of the edge
			im_256 = im_1024[k * size_crop:(k + 1) * size_crop, j * size_crop:(j + 1) * size_crop, :]
			la_256 = la_1024[k * size_crop:(k + 1) * size_crop, j * size_crop:(j + 1) * size_crop, :]
			# randomly crop
			a,b,c,d =np.random.randint(size_crop - size_randomcrop, size=4)
			im_224_1 = im_256[a:a + size_randomcrop, b:b + size_randomcrop, :]
			im_224_2 = im_256[c:c + size_randomcrop, d:d + size_randomcrop, :]
			la_224_1 = la_256[a:a + size_randomcrop, b:b + size_randomcrop, :]
			la_224_2 = la_256[c:c + size_randomcrop, d:d + size_randomcrop, :]
			# put in the list
			imgset_224.append(im_224_1)
			imgset_224.append(im_224_2)
			labelset_224.append(la_224_1)
			labelset_224.append(la_224_2)

def get_raw_image_label(data_path):

	folder_names = [data_path + "img_label_%d"%i for i in range(1,24)]

	im_pathes = []
	la_pathes = []
	for idx, folder in enumerate(folder_names):
		img_index = folder.split("_")[-1]
		la_path = folder + '/label.png'
		im_path = data_path + 'img_he_%s.png' % img_index
		im_pathes.append(im_path)
		la_pathes.append(la_path)
	no_ph = len(folder_names)

	raw_im = []
	raw_la = []
	for i in range(no_ph):
		raw_im.append(np.array(im.open(im_pathes[i])))
		# making label has two classes (add background)
		la = np.array(im.open(la_pathes[i]))
		tmp = 1 - la
		la = np.reshape(la, [size_raw, size_raw, 1])
		tmp = np.reshape(tmp, [size_raw, size_raw, 1])
		lab = np.concatenate((la, tmp), axis=2)
		raw_la.append(lab)
	# shuffling the raw images
	perm = np.random.RandomState(seed=1).permutation(no_ph)
	raw_img = []
	raw_label = []
	for i in perm:
		raw_img.append(raw_im[i])
		raw_label.append(raw_la[i])


	return raw_label, raw_img

if __name__ == "__main__":
	"""
	Making train or test data after preprocessing by preprocess.py
	"""
	raw_label, raw_img = get_raw_image_label(data_path)

	# test set
	imgset_1024 = []
	labelset_1024 = []
	# train set
	imgset_224 = []
	labelset_224 = []
	edge = (size_raw - size_trim) // 2
	for i in range(len(raw_img)):
		im_1024 = raw_img[i][edge:-edge, edge:-edge, :]
		la_1024 = raw_label[i][edge:-edge, edge:-edge, :]
		n = size_trim // size_crop
		if data_type == "test" :
			imgset_1024.append(im_1024)
			labelset_1024.append(la_1024)
		elif data_type == "train":
			# Flip
			im1, im2, im3, im4 = flip(im_1024)
			la1, la2, la3, la4 = flip(la_1024)
			# crop and put in the train set
			crop_and_put(im1, la1, n)
			crop_and_put(im2, la2, n)
			crop_and_put(im3, la3, n)
			crop_and_put(im4, la4, n)
	# save data
	if data_type == "test":
		img_set_te = np.array(imgset_1024)
		label_set_te = np.array(labelset_1024)
		np.save('x_test.npy',img_set_te)
		np.save('y_test.npy',label_set_te)

	elif data_type == "train":
		img_set_tr = np.array(imgset_224)
		label_set_tr = np.array(labelset_224)
		np.save('x_train.npy', img_set_tr)
		np.save('y_train.npy',label_set_tr)
