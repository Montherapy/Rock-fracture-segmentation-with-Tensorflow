import numpy as np
import tensorflow as tf

def make_w(shape):
	stddev = np.sqrt(2/(shape[0]*shape[1]*shape[2]))
	return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))

def make_b(shape):
	return tf.Variable(tf.constant(0.01, shape=shape))

def make_beta_and_gamma(num_featuremap):
	return tf.Variable(tf.zeros(shape=[num_featuremap])), tf.Variable(tf.ones(shape=[num_featuremap]))

def conv(inputs, filter):
	conv = tf.nn.conv2d(inputs, filter= filter, strides=[1, 1, 1, 1], padding='SAME')
	return conv

def pooling(inputs):
	return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def upconv(inputs, filter, channel_rate=2, at_bottom_of_network6=False):
	input_shape = tf.shape(inputs)
	batch = input_shape[0]
	input_height = input_shape[1]
	input_width = input_shape[2]
	input_channel = input_shape[3]

	output_height = input_height*2
	output_width = input_width*2
	output_channel = input_channel//channel_rate
	if at_bottom_of_network6:
		output_channel = 256
	return tf.nn.conv2d_transpose(inputs,filter= filter, output_shape=tf.stack([batch, output_height, output_width, output_channel]), strides=[1, 2, 2, 1],padding='SAME')

def batchnorm_activation(inputs, is_training, scope, dropout=False, keep_prob=1., norm_type='batch_norm', group=32):
	if norm_type =='batch_norm':
		norm = tf.contrib.layers.batch_norm(inputs, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, scope=scope)
	elif norm_type == 'group_norm':
		norm = tf.contrib.layers.group_norm(inputs, groups=group, scope=scope)
	elif norm_type == "instance_norm":
		norm = tf.contrib.layers.instance_norm(inputs, scope=scope)
	elif norm_type == 'None':
		norm = tf.nn.relu(inputs)
	ac_conv = tf.nn.relu(norm)
	output = tf.nn.dropout(ac_conv, keep_prob) if dropout else ac_conv

	return output

def depsep_conv(inputs, depthwise_filter, pointwise_filter):
	sepconv = tf.nn.separable_conv2d(inputs, depthwise_filter=depthwise_filter,strides=[1,1,1,1],
						   padding='SAME',pointwise_filter=pointwise_filter)
	return sepconv

def atrous_sep_conv(inputs,out_channels,rate=1):
	in_channels = inputs.get_shape().as_list()[3]
	rate_list = [rate,rate]
	df = make_w([3,3,in_channels,1])
	pf = make_w([1,1,in_channels, out_channels])
	sepconv = tf.nn.separable_conv2d(inputs, depthwise_filter=df,strides=[1,1,1,1],
						   padding='SAME',pointwise_filter=pf,rate=rate_list)
	return sepconv

def atrous_spatial_pyramid_pooling(inputs, is_training, keep_prob, atrous_rates, out_channels=256, norm_type ='batch_norm', group=32, scope="aspp"):
	with tf.variable_scope(scope):
		# (a) features extracted from various receptive fields
		with tf.name_scope("atrousconv"):
			inputs_size = tf.shape(inputs)[1:3]
			conv_1 = atrous_sep_conv(inputs, out_channels)
			conv_2 = atrous_sep_conv(inputs, out_channels, rate=atrous_rates[0])
			conv_2 = batchnorm_activation(conv_2, is_training,'rate_0',True, keep_prob, norm_type, group)
			conv_3 = atrous_sep_conv(inputs, out_channels, rate=atrous_rates[1])
			conv_3 = batchnorm_activation(conv_3, is_training, 'rate_1', True, keep_prob, norm_type, group)
			conv_4 = atrous_sep_conv(inputs, out_channels, rate=atrous_rates[2])
			conv_4 = batchnorm_activation(conv_4, is_training, 'rate_2', True, keep_prob, norm_type, group)

		# (b) the image-level features
		with tf.variable_scope("image_level_features"):
			image_level_features = tf.reduce_mean(inputs, [1, 2],  keepdims=True)
			image_level_features = atrous_sep_conv(image_level_features, out_channels)
			image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size)

		net = tf.concat([conv_1, conv_2, conv_3, conv_4, image_level_features], axis=3)
		net = atrous_sep_conv(net, out_channels)

		return net



def network6(x, firstfilter, channels, n_class, keep_prob, is_training, norm_type, group=32 ):
	x = x/127.5 - 1 ## norm
	f= firstfilter

	#goes down
	f1_1 = make_w(shape=[3,3,channels,f])  ### RGB scale
	xx = conv(x, filter=f1_1)
	batch1_1 = batchnorm_activation(xx, is_training, '1_1', True,  keep_prob, norm_type, group)

	f1_2 = make_w(shape=[3,3,f,f])
	xx = conv(batch1_1, filter=f1_2)
	batch1_2 = batchnorm_activation(xx, is_training, '1_2', norm_type=norm_type, group=group)

	mp2 = pooling(batch1_2)
	dpf2_1 = make_w(shape=[3,3,f,1])
	pwf2_1 = make_w(shape=[1,1,f,2*f])
	xx = depsep_conv(mp2, depthwise_filter=dpf2_1, pointwise_filter=pwf2_1)
	batch2_1 = batchnorm_activation(xx, is_training, '2_1', True, keep_prob, norm_type, group)
	dpf2_2 = make_w(shape=[3,3,2*f,1])
	pwf2_2 = make_w(shape=[1,1,2*f,2*f])
	xx = depsep_conv(batch2_1, depthwise_filter=dpf2_2, pointwise_filter=pwf2_2)
	batch2_2 = batchnorm_activation(xx, is_training, '2_2', norm_type=norm_type, group=group)
	xx = tf.concat([batch2_2,mp2],axis=-1)

	mp3 = pooling(xx)
	dpf3_1 = make_w(shape=[3,3,3*f,1])
	pwf3_1 = make_w(shape=[1,1,3*f,4*f])
	xx = depsep_conv(mp3, depthwise_filter=dpf3_1, pointwise_filter=pwf3_1)
	batch3_1 = batchnorm_activation(xx, is_training, '3_1', True,  keep_prob, norm_type, group)
	dpf3_2 = make_w(shape=[3,3,4*f,1])
	pwf3_2 = make_w(shape=[1,1,4*f,4*f])
	xx = depsep_conv(batch3_1, depthwise_filter=dpf3_2, pointwise_filter=pwf3_2)
	batch3_2 = batchnorm_activation(xx, is_training, '3_2', norm_type=norm_type,group=group)
	xx = tf.concat([batch3_2,mp3],axis=-1)

	mp4 = pooling(xx)
	xx = atrous_sep_conv(mp4, 8*f)
	xx = atrous_spatial_pyramid_pooling(xx, is_training, keep_prob, [2,4,8], 8*f, norm_type, group)
	xx = tf.concat([xx,mp4],axis=-1)

	## goes up
	f7_0 = make_w(shape=[3,3,4*f,15*f])
	uc7 = upconv(xx, filter=f7_0, at_bottom_of_network6=True)
	xx = tf.concat([uc7, batch3_2],axis=3)
	dpf7_1 = make_w(shape=[3,3,8*f,1])
	pwf7_1 = make_w(shape=[1,1,8*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf7_1, pointwise_filter=pwf7_1)
	xx = batchnorm_activation(xx, is_training, '7_1', True,  keep_prob, norm_type, group)
	dpf7_2 = make_w(shape=[3,3,4*f,1])
	pwf7_2 = make_w(shape=[1,1,4*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf7_2, pointwise_filter=pwf7_2)
	xx = batchnorm_activation(xx, is_training, '7_2', norm_type=norm_type, group=group)
	xx = tf.concat([uc7,xx],axis=-1)

	f8_0 = make_w(shape=[3,3,2*f,8*f])
	uc8 = upconv(xx, filter=f8_0, channel_rate=4)
	xx = tf.concat([uc8, batch2_2],axis=3)
	dpf8_1 = make_w(shape=[3,3,4*f,1])
	pwf8_1 = make_w(shape=[1,1,4*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf8_1, pointwise_filter=pwf8_1)
	xx = batchnorm_activation(xx, is_training, '8_1', True,  keep_prob, norm_type, group)
	dpf8_2 = make_w(shape=[3,3,2*f,1])
	pwf8_2 = make_w(shape=[1,1,2*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf8_2, pointwise_filter=pwf8_2)
	xx = batchnorm_activation(xx, is_training, '8_2', norm_type=norm_type, group=group)
	xx = tf.concat([uc8,xx],axis=-1)

	f9_0 = make_w(shape=[3,3,f,4*f])
	xx = upconv(xx, filter=f9_0, channel_rate=4)
	xx = tf.concat([xx, batch1_2],axis=3)
	dpf9_1 = make_w(shape=[3,3,2*f,1])
	pwf9_1 = make_w(shape=[1,1,2*f,f])
	xx = depsep_conv(xx, depthwise_filter=dpf9_1, pointwise_filter=pwf9_1)
	xx = batchnorm_activation(xx, is_training, '9_1', True, keep_prob, norm_type, group)
	dpf9_2 = make_w(shape=[3,3,f,1])
	pwf9_2 = make_w(shape=[1,1,f,f])
	xx = depsep_conv(xx, depthwise_filter=dpf9_2, pointwise_filter=pwf9_2)
	xx = batchnorm_activation(xx, is_training, '9_2', norm_type=norm_type, group=group)
	f9_3 = make_w(shape=[1,1,f,n_class])
	xx = conv(xx, filter=f9_3)

	b = make_b(shape=[n_class])
	xx = xx + b
	return xx

def network5(x, firstfilter, channels, n_class, keep_prob, is_training, norm_type, group=32 ):
	x = x/127.5 - 1 ## norm
	f= firstfilter

	#goes down
	f1_1 = make_w(shape=[3,3,channels,f])  ### RGB scale
	xx = conv(x, filter=f1_1)
	batch1_1 = batchnorm_activation(xx, is_training, '1_1', True,  keep_prob, norm_type, group)
	f1_2 = make_w(shape=[3,3,f,f])
	xx = conv(batch1_1, filter=f1_2)
	batch1_2 = batchnorm_activation(xx, is_training, '1_2', norm_type=norm_type, group=group)

	xx = pooling(batch1_2)
	dpf2_1 = make_w(shape=[3,3,f,1])
	pwf2_1 = make_w(shape=[1,1,f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf2_1, pointwise_filter=pwf2_1)
	batch2_1 = batchnorm_activation(xx, is_training, '2_1', True,  keep_prob, norm_type, group)
	dpf2_2 = make_w(shape=[3,3,2*f,1])
	pwf2_2 = make_w(shape=[1,1,2*f,2*f])
	xx = depsep_conv(batch2_1, depthwise_filter=dpf2_2, pointwise_filter=pwf2_2)
	batch2_2 = batchnorm_activation(xx, is_training, '2_2', norm_type=norm_type, group=group)

	xx = pooling(batch2_2)
	dpf3_1 = make_w(shape=[3,3,2*f,1])
	pwf3_1 = make_w(shape=[1,1,2*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf3_1, pointwise_filter=pwf3_1)
	batch3_1 = batchnorm_activation(xx, is_training, '3_1', True,  keep_prob, norm_type, group)
	dpf3_2 = make_w(shape=[3,3,4*f,1])
	pwf3_2 = make_w(shape=[1,1,4*f,4*f])
	xx = depsep_conv(batch3_1, depthwise_filter=dpf3_2, pointwise_filter=pwf3_2)
	batch3_2 = batchnorm_activation(xx, is_training, '3_2', norm_type=norm_type, group=group)

	xx = pooling(batch3_2)
	xx = atrous_sep_conv(xx, 8*f)
	xx = atrous_spatial_pyramid_pooling(xx, is_training, keep_prob, [2,4,8], 8*f, norm_type, group)

	## goes up
	f7_0 = make_w(shape=[3,3,4*f,8*f])
	xx = upconv(xx, filter=f7_0)
	xx = tf.concat([xx, batch3_2],axis=3)
	dpf7_1 = make_w(shape=[3,3,8*f,1])
	pwf7_1 = make_w(shape=[1,1,8*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf7_1, pointwise_filter=pwf7_1)
	xx = batchnorm_activation(xx, is_training, '7_1', True,  keep_prob, norm_type, group)
	xx = tf.concat([xx, batch3_1],axis=3)
	dpf7_2 = make_w(shape=[3,3,8*f,1])
	pwf7_2 = make_w(shape=[1,1,8*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf7_2, pointwise_filter=pwf7_2)
	xx = batchnorm_activation(xx, is_training, '7_2', norm_type=norm_type, group=group)

	f8_0 = make_w(shape=[3,3,2*f,4*f])
	xx = upconv(xx, filter=f8_0)
	xx = tf.concat([xx, batch2_2],axis=3)
	dpf8_1 = make_w(shape=[3,3,4*f,1])
	pwf8_1 = make_w(shape=[1,1,4*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf8_1, pointwise_filter=pwf8_1)
	xx = batchnorm_activation(xx, is_training, '8_1', True,  keep_prob, norm_type, group)
	xx = tf.concat([xx, batch2_1],axis=3)
	dpf8_2 = make_w(shape=[3,3,4*f,1])
	pwf8_2 = make_w(shape=[1,1,4*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf8_2, pointwise_filter=pwf8_2)
	xx = batchnorm_activation(xx, is_training, '8_2', norm_type=norm_type, group=group)

	f9_0 = make_w(shape=[3,3,f,2*f])
	xx = upconv(xx, filter=f9_0)
	xx = tf.concat([xx, batch1_2],axis=3)
	dpf9_1 = make_w(shape=[3,3,2*f,1])
	pwf9_1 = make_w(shape=[1,1,2*f,f])
	xx = depsep_conv(xx, depthwise_filter=dpf9_1, pointwise_filter=pwf9_1)
	xx = batchnorm_activation(xx, is_training, '9_1', True,  keep_prob, norm_type, group)
	xx = tf.concat([xx, batch1_1],axis=3)
	dpf9_2 = make_w(shape=[3,3,2*f,1])
	pwf9_2 = make_w(shape=[1,1,2*f,f])
	xx = depsep_conv(xx, depthwise_filter=dpf9_2, pointwise_filter=pwf9_2)
	xx = batchnorm_activation(xx, is_training, '9_2', norm_type=norm_type, group=group)
	f9_3 = make_w(shape=[1,1,f,n_class])
	xx = conv(xx, filter=f9_3)

	b = make_b(shape=[n_class])
	xx = xx + b
	return xx

def network4(x, firstfilter, channels, n_class, keep_prob, is_training, norm_type, group=32 ):
	x = x/127.5 - 1 ## norm
	f= firstfilter

	#goes down
	f1_1 = make_w(shape=[3,3,channels,f])  ### RGB scale
	xx = conv(x, filter=f1_1)
	xx = batchnorm_activation(xx, is_training, '1_1', True, keep_prob, norm_type, group)
	f1_2 = make_w(shape=[3,3,f,f])
	xx = conv(xx, filter=f1_2)
	batch1_2 = batchnorm_activation(xx, is_training, '1_2', norm_type=norm_type, group=group)

	xx = pooling(batch1_2)
	dpf2_1 = make_w(shape=[3,3,f,1])
	pwf2_1 = make_w(shape=[1,1,f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf2_1, pointwise_filter=pwf2_1)
	xx = batchnorm_activation(xx, is_training, '2_1', True,  keep_prob, norm_type, group)
	dpf2_2 = make_w(shape=[3,3,2*f,1])
	pwf2_2 = make_w(shape=[1,1,2*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf2_2, pointwise_filter=pwf2_2)
	batch2_2 = batchnorm_activation(xx, is_training, '2_2', norm_type=norm_type, group=group)

	xx = pooling(batch2_2)
	dpf3_1 = make_w(shape=[3,3,2*f,1])
	pwf3_1 = make_w(shape=[1,1,2*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf3_1, pointwise_filter=pwf3_1)
	xx = batchnorm_activation(xx, is_training, '3_1', True,  keep_prob, norm_type, group)
	dpf3_2 = make_w(shape=[3,3,4*f,1])
	pwf3_2 = make_w(shape=[1,1,4*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf3_2, pointwise_filter=pwf3_2)
	batch3_2 = batchnorm_activation(xx, is_training, '3_2', norm_type=norm_type, group=group)

	xx = pooling(batch3_2)
	dpf4_1 = make_w(shape=[3,3,4*f,1])
	pwf4_1 = make_w(shape=[1,1,4*f,8*f])
	xx = depsep_conv(xx, depthwise_filter=dpf4_1, pointwise_filter=pwf4_1)
	xx = batchnorm_activation(xx, is_training, '4_1', True, keep_prob, norm_type, group)
	dpf4_2 = make_w(shape=[3,3,8*f,1])
	pwf4_2 = make_w(shape=[1,1,8*f,8*f])
	xx = depsep_conv(xx, depthwise_filter=dpf4_2, pointwise_filter=pwf4_2)
	batch4_2 = batchnorm_activation(xx, is_training, '4_2', norm_type=norm_type, group=group)

	xx = pooling(batch4_2)
	xx = atrous_sep_conv(xx, 16*f)
	xx = atrous_spatial_pyramid_pooling(xx, is_training, keep_prob, [2,4,8], 16*f, norm_type, group)

	#goes up
	f6_0 = make_w(shape=[3,3,8*f,16*f])
	xx = upconv(xx, filter=f6_0)
	xx = tf.concat([xx, batch4_2],axis=3)
	dpf6_1 = make_w(shape=[3,3,16*f,1])
	pwf6_1 = make_w(shape=[1,1,16*f,8*f])
	xx = depsep_conv(xx, depthwise_filter=dpf6_1, pointwise_filter=pwf6_1)
	xx = batchnorm_activation(xx, is_training, '6_1', True,  keep_prob, norm_type, group)
	dpf6_2 = make_w(shape=[3,3,8*f,1])
	pwf6_2 = make_w(shape=[1,1,8*f,8*f])
	xx = depsep_conv(xx, depthwise_filter=dpf6_2, pointwise_filter=pwf6_2)
	xx = batchnorm_activation(xx, is_training, '6_2', norm_type=norm_type, group=group)

	f7_0 = make_w(shape=[3,3,4*f,8*f])
	xx = upconv(xx, filter=f7_0)
	xx = tf.concat([xx, batch3_2],axis=3)
	dpf7_1 = make_w(shape=[3,3,8*f,1])
	pwf7_1 = make_w(shape=[1,1,8*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf7_1, pointwise_filter=pwf7_1)
	xx = batchnorm_activation(xx, is_training, '7_1', True,  keep_prob, norm_type, group)
	dpf7_2 = make_w(shape=[3,3,4*f,1])
	pwf7_2 = make_w(shape=[1,1,4*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf7_2, pointwise_filter=pwf7_2)
	xx = batchnorm_activation(xx, is_training, '7_2', norm_type=norm_type, group=group)

	f8_0 = make_w(shape=[3,3,2*f,4*f])
	xx = upconv(xx, filter=f8_0)
	xx = tf.concat([xx, batch2_2],axis=3)
	dpf8_1 = make_w(shape=[3,3,4*f,1])
	pwf8_1 = make_w(shape=[1,1,4*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf8_1, pointwise_filter=pwf8_1)
	xx = batchnorm_activation(xx, is_training, '8_1', True,  keep_prob, norm_type, group)
	dpf8_2 = make_w(shape=[3,3,2*f,1])
	pwf8_2 = make_w(shape=[1,1,2*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf8_2, pointwise_filter=pwf8_2)
	xx = batchnorm_activation(xx, is_training, '8_2', norm_type=norm_type, group=group)

	f9_0 = make_w(shape=[3,3,f,2*f])
	xx = upconv(xx, filter=f9_0)
	xx = tf.concat([xx, batch1_2],axis=3)
	dpf9_1 = make_w(shape=[3,3,2*f,1])
	pwf9_1 = make_w(shape=[1,1,2*f,f])
	xx = depsep_conv(xx, depthwise_filter=dpf9_1, pointwise_filter=pwf9_1)
	xx = batchnorm_activation(xx, is_training, '9_1', True, keep_prob, norm_type, group)
	dpf9_2 = make_w(shape=[3,3,f,1])
	pwf9_2 = make_w(shape=[1,1,f,f])
	xx = depsep_conv(xx, depthwise_filter=dpf9_2, pointwise_filter=pwf9_2)
	xx = batchnorm_activation(xx, is_training, '9_2', norm_type=norm_type, group=group)
	f9_3 = make_w(shape=[1,1,f,n_class])
	xx = conv(xx, filter=f9_3)

	b = make_b(shape=[n_class])
	xx =xx + b
	return xx

def network3(x, firstfilter, channels, n_class, keep_prob, is_training, norm_type, group=32 ):
	x = x/127.5 - 1 ## norm
	f= firstfilter

	#goes down
	f1_1 = make_w(shape=[3,3,channels,f])  ### RGB scale
	xx = conv(x, filter=f1_1)
	xx = batchnorm_activation(xx, is_training, '1_1', True,  keep_prob, norm_type, group)
	f1_2 = make_w(shape=[3,3,f,f])
	xx = conv(xx, filter=f1_2)
	batch1_2 = batchnorm_activation(xx, is_training, '1_2', norm_type=norm_type, group=group)

	xx = pooling(batch1_2)
	dpf2_1 = make_w(shape=[3,3,f,1])
	pwf2_1 = make_w(shape=[1,1,f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf2_1, pointwise_filter=pwf2_1)
	xx = batchnorm_activation(xx, is_training, '2_1', True,  keep_prob, norm_type, group)
	dpf2_2 = make_w(shape=[3,3,2*f,1])
	pwf2_2 = make_w(shape=[1,1,2*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf2_2, pointwise_filter=pwf2_2)
	batch2_2 = batchnorm_activation(xx, is_training, '2_2', norm_type=norm_type, group=group)

	xx = pooling(batch2_2)
	dpf3_1 = make_w(shape=[3,3,2*f,1])
	pwf3_1 = make_w(shape=[1,1,2*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf3_1, pointwise_filter=pwf3_1)
	xx = batchnorm_activation(xx, is_training, '3_1', True,  keep_prob, norm_type, group)
	dpf3_2 = make_w(shape=[3,3,4*f,1])
	pwf3_2 = make_w(shape=[1,1,4*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf3_2, pointwise_filter=pwf3_2)
	batch3_2 = batchnorm_activation(xx, is_training, '3_2', norm_type=norm_type, group=group)

	xx = pooling(batch3_2)
	xx = atrous_sep_conv(xx, 8*f)
	xx = atrous_spatial_pyramid_pooling(xx, is_training, keep_prob, [2,4,8], 8*f, norm_type, group)

	## goes up
	f7_0 = make_w(shape=[3,3,4*f,8*f])
	xx = upconv(xx, filter=f7_0)
	xx = tf.concat([xx, batch3_2],axis=3)
	dpf7_1 = make_w(shape=[3,3,8*f,1])
	pwf7_1 = make_w(shape=[1,1,8*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf7_1, pointwise_filter=pwf7_1)
	xx = batchnorm_activation(xx, is_training, '7_1', True,  keep_prob, norm_type, group)
	dpf7_2 = make_w(shape=[3,3,4*f,1])
	pwf7_2 = make_w(shape=[1,1,4*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf7_2, pointwise_filter=pwf7_2)
	xx = batchnorm_activation(xx, is_training, '7_2', norm_type=norm_type, group=group)

	f8_0 = make_w(shape=[3,3,2*f,4*f])
	xx = upconv(xx, filter=f8_0)
	xx = tf.concat([xx, batch2_2],axis=3)
	dpf8_1 = make_w(shape=[3,3,4*f,1])
	pwf8_1 = make_w(shape=[1,1,4*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf8_1, pointwise_filter=pwf8_1)
	xx = batchnorm_activation(xx, is_training, '8_1', True,  keep_prob, norm_type, group)
	dpf8_2 = make_w(shape=[3,3,2*f,1])
	pwf8_2 = make_w(shape=[1,1,2*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf8_2, pointwise_filter=pwf8_2)
	xx = batchnorm_activation(xx, is_training, '8_2', norm_type=norm_type, group=group)

	f9_0 = make_w(shape=[3,3,f,2*f])
	xx = upconv(xx, filter=f9_0)
	xx = tf.concat([xx, batch1_2],axis=3)
	dpf9_1 = make_w(shape=[3,3,2*f,1])
	pwf9_1 = make_w(shape=[1,1,2*f,f])
	xx = depsep_conv(xx, depthwise_filter=dpf9_1, pointwise_filter=pwf9_1)
	xx = batchnorm_activation(xx, is_training, '9_1', True,  keep_prob, norm_type, group)
	dpf9_2 = make_w(shape=[3,3,f,1])
	pwf9_2 = make_w(shape=[1,1,f,f])
	xx = depsep_conv(xx, depthwise_filter=dpf9_2, pointwise_filter=pwf9_2)
	xx = batchnorm_activation(xx, is_training, '9_2', norm_type=norm_type, group=group)
	f9_3 = make_w(shape=[1,1,f,n_class])
	xx = conv(xx, filter=f9_3)

	b = make_b(shape=[n_class])
	xx = xx + b
	return xx


def network2(x, firstfilter, channels, n_class, keep_prob, is_training, norm_type, group=32 ):
	x = x/127.5 - 1 ## norm
	f= firstfilter

	#goes down
	f1_1 = make_w(shape=[3,3,channels,f])  ### RGB scale
	xx = conv(x, filter=f1_1)
	xx = batchnorm_activation(xx, is_training, '1_1', True,  keep_prob, norm_type, group)
	f1_2 = make_w(shape=[3,3,f,f])
	xx = conv(xx, filter=f1_2)
	batch1_2 = batchnorm_activation(xx, is_training, '1_2', norm_type=norm_type, group=group)

	xx = pooling(batch1_2)
	xx = atrous_sep_conv(xx,out_channels=2*f)
	xx = batchnorm_activation(xx, is_training, '2_1', True,  keep_prob, norm_type, group)
	xx = atrous_sep_conv(xx,out_channels=2*f)
	batch2_2 = batchnorm_activation(xx, is_training, '2_2', norm_type=norm_type, group=group)

	xx = pooling(batch2_2)
	xx = atrous_spatial_pyramid_pooling(xx, is_training, keep_prob, [4,8,16], 4*f, norm_type, group)

	## goes up
	f8_0 = make_w(shape=[3,3,2*f,4*f])
	xx = upconv(xx, filter=f8_0)
	xx = tf.concat([xx, batch2_2],axis=3)
	dpf8_1 = make_w(shape=[3,3,4*f,1])
	pwf8_1 = make_w(shape=[1,1,4*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf8_1, pointwise_filter=pwf8_1)
	xx = batchnorm_activation(xx, is_training, '8_1', True,  keep_prob, norm_type, group)
	dpf8_2 = make_w(shape=[3,3,2*f,1])
	pwf8_2 = make_w(shape=[1,1,2*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf8_2, pointwise_filter=pwf8_2)
	xx = batchnorm_activation(xx, is_training, '8_2', norm_type=norm_type,group=group)

	f9_0 = make_w(shape=[3,3,f,2*f])
	xx = upconv(xx, filter=f9_0)
	xx = tf.concat([xx, batch1_2],axis=3)
	dpf9_1 = make_w(shape=[3,3,2*f,1])
	pwf9_1 = make_w(shape=[1,1,2*f,f])
	xx = depsep_conv(xx, depthwise_filter=dpf9_1, pointwise_filter=pwf9_1)
	xx = batchnorm_activation(xx, is_training, '9_1', True,  keep_prob, norm_type, group)
	dpf9_2 = make_w(shape=[3,3,f,1])
	pwf9_2 = make_w(shape=[1,1,f,f])
	xx = depsep_conv(xx, depthwise_filter=dpf9_2, pointwise_filter=pwf9_2)
	xx = batchnorm_activation(xx, is_training, '9_2', norm_type=norm_type, group=group)
	f9_3 = make_w(shape=[1,1,f,n_class])
	xx = conv(xx, filter=f9_3)

	b = make_b(shape=[n_class])
	xx = xx + b
	return xx

def network1(x, firstfilter, channels, n_class, keep_prob, is_training, norm_type, group=32 ):
	x = x/127.5 - 1 ## norm
	f= firstfilter

	#goes down
	f1_1 = make_w(shape=[3,3,channels,f])  ### RGB scale
	xx = conv(x, filter=f1_1)
	xx = batchnorm_activation(xx, is_training, '1_1', True,  keep_prob, norm_type, group)
	f1_2 = make_w(shape=[3,3,f,f])
	xx = conv(xx, filter=f1_2)
	batch1_2 = batchnorm_activation(xx, is_training, '1_2', norm_type=norm_type, group=group)

	xx = pooling(batch1_2)
	dpf2_1 = make_w(shape=[3,3,f,1])
	pwf2_1 = make_w(shape=[1,1,f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf2_1, pointwise_filter=pwf2_1)
	xx = batchnorm_activation(xx, is_training, '2_1', True, keep_prob, norm_type, group)
	dpf2_2 = make_w(shape=[3,3,2*f,1])
	pwf2_2 = make_w(shape=[1,1,2*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf2_2, pointwise_filter=pwf2_2)
	batch2_2 = batchnorm_activation(xx, is_training, '2_2', norm_type=norm_type, group=group)

	xx = pooling(batch2_2)
	dpf3_1 = make_w(shape=[3,3,2*f,1])
	pwf3_1 = make_w(shape=[1,1,2*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf3_1, pointwise_filter=pwf3_1)
	xx = batchnorm_activation(xx, is_training, '3_1', True,  keep_prob, norm_type, group)
	dpf3_2 = make_w(shape=[3,3,4*f,1])
	pwf3_2 = make_w(shape=[1,1,4*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf3_2, pointwise_filter=pwf3_2)
	batch3_2 = batchnorm_activation(xx, is_training, '3_2', norm_type=norm_type, group=group)

	xx = pooling(batch3_2)
	dpf4_1 = make_w(shape=[3,3,4*f,1])
	pwf4_1 = make_w(shape=[1,1,4*f,8*f])
	xx = depsep_conv(xx, depthwise_filter=dpf4_1, pointwise_filter=pwf4_1)
	xx = batchnorm_activation(xx, is_training, '4_1', True, keep_prob, norm_type, group)
	dpf4_2 = make_w(shape=[3,3,8*f,1])
	pwf4_2 = make_w(shape=[1,1,8*f,8*f])
	xx = depsep_conv(xx, depthwise_filter=dpf4_2, pointwise_filter=pwf4_2)
	batch4_2 = batchnorm_activation(xx, is_training, '4_2', norm_type=norm_type, group=group)

	xx = pooling(batch4_2)
	dpf5_1 = make_w(shape=[3,3,8*f,1])
	pwf5_1 = make_w(shape=[1,1,8*f,16*f])
	xx = depsep_conv(xx, depthwise_filter=dpf5_1, pointwise_filter=pwf5_1)
	xx = batchnorm_activation(xx, is_training, '5_1', True,  keep_prob, norm_type, group)
	dpf5_2 = make_w(shape=[3,3,16*f,1])
	pwf5_2 = make_w(shape=[1,1,16*f,16*f])
	xx = depsep_conv(xx, depthwise_filter=dpf5_2, pointwise_filter=pwf5_2)
	xx = batchnorm_activation(xx, is_training, '5_2', norm_type=norm_type, group=group)

	#goes up
	f6_0 = make_w(shape=[3,3,8*f,16*f])
	xx = upconv(xx, filter=f6_0)
	xx = tf.concat([xx, batch4_2],axis=3)
	dpf6_1 = make_w(shape=[3,3,16*f,1])
	pwf6_1 = make_w(shape=[1,1,16*f,8*f])
	xx = depsep_conv(xx, depthwise_filter=dpf6_1, pointwise_filter=pwf6_1)
	xx = batchnorm_activation(xx, is_training, '6_1', True,  keep_prob, norm_type, group)
	dpf6_2 = make_w(shape=[3,3,8*f,1])
	pwf6_2 = make_w(shape=[1,1,8*f,8*f])
	xx = depsep_conv(xx, depthwise_filter=dpf6_2, pointwise_filter=pwf6_2)
	xx = batchnorm_activation(xx, is_training, '6_2', norm_type=norm_type, group=group)

	f7_0 = make_w(shape=[3,3,4*f,8*f])
	xx = upconv(xx, filter=f7_0)
	xx = tf.concat([xx, batch3_2],axis=3)
	dpf7_1 = make_w(shape=[3,3,8*f,1])
	pwf7_1 = make_w(shape=[1,1,8*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf7_1, pointwise_filter=pwf7_1)
	xx = batchnorm_activation(xx, is_training, '7_1', True,  keep_prob, norm_type, group)
	dpf7_2 = make_w(shape=[3,3,4*f,1])
	pwf7_2 = make_w(shape=[1,1,4*f,4*f])
	xx = depsep_conv(xx, depthwise_filter=dpf7_2, pointwise_filter=pwf7_2)
	xx = batchnorm_activation(xx, is_training, '7_2', norm_type=norm_type, group=group)

	f8_0 = make_w(shape=[3,3,2*f,4*f])
	xx = upconv(xx, filter=f8_0)
	xx = tf.concat([xx, batch2_2],axis=3)
	dpf8_1 = make_w(shape=[3,3,4*f,1])
	pwf8_1 = make_w(shape=[1,1,4*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf8_1, pointwise_filter=pwf8_1)
	xx = batchnorm_activation(xx, is_training, '8_1', True, keep_prob, norm_type, group)
	dpf8_2 = make_w(shape=[3,3,2*f,1])
	pwf8_2 = make_w(shape=[1,1,2*f,2*f])
	xx = depsep_conv(xx, depthwise_filter=dpf8_2, pointwise_filter=pwf8_2)
	xx = batchnorm_activation(xx, is_training, '8_2', norm_type=norm_type, group=group)

	f9_0 = make_w(shape=[3,3,f,2*f])
	xx = upconv(xx, filter=f9_0)
	xx = tf.concat([xx, batch1_2],axis=3)
	dpf9_1 = make_w(shape=[3,3,2*f,1])
	pwf9_1 = make_w(shape=[1,1,2*f,f])
	xx = depsep_conv(xx, depthwise_filter=dpf9_1, pointwise_filter=pwf9_1)
	xx = batchnorm_activation(xx, is_training, '9_1', True,  keep_prob, norm_type, group)
	dpf9_2 = make_w(shape=[3,3,f,1])
	pwf9_2 = make_w(shape=[1,1,f,f])
	xx = depsep_conv(xx, depthwise_filter=dpf9_2, pointwise_filter=pwf9_2)
	xx = batchnorm_activation(xx, is_training, '9_2', norm_type=norm_type, group=group)
	f9_3 = make_w(shape=[1,1,f,n_class])
	xx = conv(xx, filter=f9_3)

	b = make_b(shape=[n_class])
	xx = xx + b
	return xx
