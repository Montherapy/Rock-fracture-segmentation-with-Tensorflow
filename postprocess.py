# coding: utf-8

import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import matplotlib.pyplot as plt
import os

x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")
softmax = np.load("softmax_output.npy")
prediction = np.argmax(softmax, axis=3)

def fc_crf(softmax, Gsxy, Gcompat, Bsxy, Bsrgb, Bcompat, step, rgbim, savefigure=False):
	"""
	This code is based on PyDenseCRF code source:
		https://github.com/lucasb-eyer/pydensecrf
	"""

	output = []
	for i in range(softmax.shape[0]):
		im = softmax[i]
		w = im.shape[0]
		h = im.shape[1]
		n = im.shape[2]
		d = dcrf.DenseCRF2D(w, h, n)  # width, height, nlabels

		U = -np.log(im)  # Get the unary in some way.
		U = U.transpose(2, 0, 1).reshape((n, -1))  # Needs to be flat.
		U = np.ascontiguousarray(U)
		rgb = np.ascontiguousarray(rgbim[i])
		d.setUnaryEnergy(U)
		d.addPairwiseGaussian(sxy=Gsxy, compat=Gcompat)
		d.addPairwiseBilateral(sxy=Bsxy, srgb=Bsrgb, rgbim=rgb, compat=Bcompat)

		Q = d.inference(step)
		Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
		output.append(Q)
	output = np.array(output)

	if savefigure:
		try:
			os.makedirs("./result_fig/")
		except FileExistsError:
			pass
		for i in range(softmax.shape[0]):
			fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(12, 3))
			ax[0].imshow(x_test[i, ...], aspect="auto")
			ax[1].imshow(y_test[i, ..., 1], aspect="auto", cmap=plt.cm.gist_stern)
			ax[2].imshow(prediction[i], aspect="auto", cmap=plt.cm.gist_stern)
			ax[3].imshow(output[i], aspect="auto", cmap=plt.cm.gist_stern)
			ax[0].set_title("Network Input")
			ax[1].set_title("Ground truth")
			ax[2].set_title("Previous")
			ax[3].set_title("Postprocessing")
			ax[0].axis('Off')
			ax[1].axis('Off')
			ax[2].axis('Off')
			ax[3].axis('off')
			fig.tight_layout()
			# plt.show()
			fig.savefig("./result_fig/result_%d.jpg" % i, dpi=100, bbox_inches='tight',
						pad_inches=0)

	return output

def evaluate_result(prediction_result, ground_truth):
	"""

	:param prediction_result: Network output or CRF output
	:param ground_truth: Label
	:return:
	Calculate Recall, Precision and IoU

	"""
	union=(1 - np.multiply((1 - ground_truth), (1 - prediction_result))).sum(axis=(1, 2))
	intersection=np.multiply(ground_truth, prediction_result).sum(axis=(1, 2))
	recall = intersection / ground_truth.sum(axis=(1, 2))
	precision = intersection / prediction_result.sum(axis=(1, 2))

	iou = intersection / union
	mrecall = recall.mean()
	mprecision = precision.mean()
	miou = iou.mean()

	print("Mean Recall : ", mrecall)
	print("Mean Precision : ", mprecision)
	print("Mean IoU : ", miou)

if __name__ == "__main__":
	output =fc_crf(softmax, Gsxy=0.35, Gcompat=0.35, Bsxy=1.6, Bsrgb=300, Bcompat=10, step=3, rgbim=x_test, savefigure=True)

	# class "fractures"
	crf_result = 1 - output
	pred_result = 1 - prediction
	ground_truth = 1 - y_test[..., 1]

	# Network only
	evaluate_result(pred_result, ground_truth)

	# Network + CRF
	evaluate_result(crf_result, ground_truth)
