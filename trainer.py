import numpy as np
import tensorflow as tf

class Trainer():
	def __init__(self, net, x_tr=None, y_tr=None):

		self.net = net
		self.saver = tf.train.Saver()
		self.x_tr = x_tr
		self.y_tr = y_tr
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)

	def get_output(self, x_test, model_path=None):
		init = tf.global_variables_initializer()
		softmax_results = []
		prediction_results = []
		self.sess.run(init)
		if model_path is not None:
			self.saver.restore(self.sess, model_path)
			print("Model restored from file: %s" % model_path)

		for x in x_test:
			x = x[np.newaxis, :]
			output, softmax = self.sess.run([self.net.prediction, self.net.result],
											feed_dict={self.net.inputs: x, self.net.keep_prob: 1.,
													   self.net.is_training: False})
			softmax_results.append(softmax)
			prediction_results.append(output)

		return np.squeeze(np.array(softmax_results)), np.squeeze(np.array(prediction_results))

	def evaluate(self, x_test, y_test, model_path=None):
		init = tf.global_variables_initializer()
		self.sess.run(init)
		if model_path is not None:
			self.saver.restore(self.sess, model_path)
			print("Model restored from file: %s" % model_path)

			num_img = len(x_test)
			iou, num_iou = 0, 0
			# calculate metrics of each image
			for i in range(num_img):
				it, un, io = self.sess.run(
					[self.net.intersection, self.net.union, self.net.iou],
					feed_dict={self.net.inputs: x_test[i:i + 1], self.net.labels: y_test[i:i + 1],
							   self.net.keep_prob: 1.0, self.net.is_training: False})
				if it != 0 and un != 0:
					iou += io
					num_iou += 1

			print("number of image :", num_img)
			iou = iou / num_iou
			print("IOU : {}".format(iou))

	def get_optimizer(self):

		optimizer = tf.train.AdamOptimizer(learning_rate=self.net.learning_rate)
		gradients, variables = zip(*optimizer.compute_gradients(self.net.loss))
		gradients = [
			None if gradient is None else tf.clip_by_norm(gradient, 1.0)
			for gradient in gradients]
		operation = optimizer.apply_gradients(zip(gradients, variables))
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		operation = tf.group([operation, update_ops])

		return operation

	def training(self, epochs, learning_rate, batch_size, keep_prob_=1.,
				 restore_path=None, save_path=None, decrease_rate=1, decrease_epoch=1):

		optimizer = self.get_optimizer()
		self.sess.run(tf.global_variables_initializer())
		print("start training")

		if restore_path is not None:
			self.saver.restore(self.sess, restore_path)

		for epoch in range(epochs):
			print("\n epoch : ", epoch)
			print("learning rate : ", learning_rate * (decrease_rate ** int(epoch / decrease_epoch)))

			# making batch set for training
			batch_x, batch_y = [], []
			num_trset = self.x_tr.shape[0]
			# shuffling training data
			per = np.random.permutation(num_trset)
			self.x_tr = self.x_tr[per]
			self.y_tr = self.y_tr[per]

			# make batch set
			if num_trset % batch_size != 0:
				for i in range(num_trset // batch_size + 1):
					if i == num_trset // batch_size:
						batch_x.append(self.x_tr[batch_size * i:])
						batch_y.append(self.y_tr[batch_size * i:])
					else:
						batch_x.append(self.x_tr[batch_size * i:batch_size * (i + 1)])
						batch_y.append(self.y_tr[batch_size * i:batch_size * (i + 1)])

			if num_trset % batch_size == 0:
				for i in range(num_trset // batch_size):
					batch_x.append(self.x_tr[batch_size * i:batch_size * (i + 1)])
					batch_y.append(self.y_tr[batch_size * i:batch_size * (i + 1)])

			# start training
			iter_in_1epoch = len(batch_x)
			los = 0
			for i in range(iter_in_1epoch):
				l, _ = self.sess.run([self.net.loss, optimizer],
									 feed_dict={self.net.inputs: batch_x[i], self.net.labels: batch_y[i],
												self.net.keep_prob: keep_prob_, self.net.is_training: True,
												self.net.learning_rate: learning_rate * (
													decrease_rate ** int(epoch / decrease_epoch))})
				los = los + l
			print("epoch : {}, loss : {} ".format(epoch, los))

			if save_path is not None:
				save_path_epoch = save_path + '-ep_%d' % epoch
				self.saver.save(self.sess, save_path_epoch)
				print("saved")
		print("Training Finish !")
