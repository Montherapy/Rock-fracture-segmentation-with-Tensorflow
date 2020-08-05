import tensorflow as tf
from utils import *


class Net():
	def __init__(self, network_type=1, size=224, firstfilter=64, channels=3, n_class=2, class_weights=[0.5, 0.5],
				 norm_kwargs={}):
		tf.reset_default_graph()
		self.network_type = network_type
		self.n_class = n_class
		self.class_weights = class_weights
		norm_type = norm_kwargs.get("norm_type")
		group = norm_kwargs.get("group", 32)  ## used for group normalization
		self.is_training = tf.placeholder(dtype=tf.bool)
		self.learning_rate = tf.placeholder(dtype=tf.float32)

		self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, size, size, channels])
		self.labels = tf.placeholder(dtype=tf.float32, shape=[None, size, size, n_class])
		self.keep_prob = tf.placeholder(dtype=tf.float32)
		if network_type == 1:  # "Baseline network"
			self.logits = network1(self.inputs, firstfilter, channels, n_class, self.keep_prob, self.is_training,
								   norm_type, group)
		elif network_type == 2:  # "ASPP with 2 stages"
			self.logits = network2(self.inputs, firstfilter, channels, n_class, self.keep_prob, self.is_training, norm_type,
							  group)
		elif network_type == 3:  # "ASPP with 3 stages"
			self.logits = network3(self.inputs, firstfilter, channels, n_class, self.keep_prob, self.is_training,
								   norm_type, group)
		elif network_type == 4:  # "ASPP with 4 stages"
			self.logits = network4(self.inputs, firstfilter, channels, n_class, self.keep_prob, self.is_training,
								   norm_type, group)
		elif network_type == 5:  # "Additional inter-block connection"
			self.logits = network5(self.inputs, firstfilter, channels, n_class, self.keep_prob, self.is_training,
								   norm_type, group)
		elif network_type == 6:  # "Additional intra-block connection"
			self.logits = network6(self.inputs, firstfilter, channels, n_class, self.keep_prob, self.is_training,
								   norm_type, group)

		with tf.variable_scope("loss"):
			self.loss = self.weighted_cross_entropy()
		with tf.variable_scope("output"):
			self.result = tf.nn.softmax(self.logits)
			self.prediction = tf.argmax(self.result, axis=-1)
		with tf.variable_scope("metric"):
			labels = tf.argmax(self.labels, axis=-1)
			# intersection of union
			zeros = tf.zeros_like(tensor=labels)
			tf_sum = labels + self.prediction
			tf_mul = tf.math.multiply(labels, self.prediction)
			self.intersection = tf.reduce_sum(tf.cast(tf.equal(tf_sum, zeros), tf.float32))
			self.union = tf.reduce_sum(tf.cast(tf.equal(tf_mul, zeros), tf.float32))
			self.iou = self.intersection / (self.union + 1e-5)

	def weighted_cross_entropy(self):
		logits = tf.reshape(self.logits, [-1, self.n_class])
		labels = tf.reshape(self.labels, [-1, self.n_class])
		normal_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels, logits)
		class_weights = tf.constant(self.class_weights, dtype=tf.float32)

		weights = tf.reduce_sum(class_weights * labels, axis=1)
		weighted_loss = tf.reduce_mean(normal_loss * weights)

		return weighted_loss


