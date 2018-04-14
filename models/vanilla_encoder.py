import tensorflow as tf
import tensorlayer as tl

def model(inputs, z_dim=100, is_train=True, reuse=False):
	df_dim = 64
	w_init = tf.random_normal_initializer(stddev=0.02)
	gamma_init = tf.random_normal_initializer(1., 0.02)
	with tf.variable_scope('encoder', reuse=reuse):
		#tl.layers.set_name_reuse(reuse)
		network_in = tl.layers.InputLayer(inputs, name='d/in')
		network_h0 = tl.layers.Conv2dLayer(network_in, act=lambda x: tl.act.lrelu(x, 0.2),
											W_init=w_init,
											shape=[5, 5, 3, 64], 
											strides=[1, 2, 2, 1],
											padding='SAME',
											name='enc/h0/conv2d')
		network_h1 = tl.layers.Conv2dLayer(network_h0, act=tf.identity,
											W_init=w_init,
											shape=[5, 5, 64, 128],
											strides=[1, 2, 2, 1],
											padding='SAME',
											name='enc/h1/conv2d')
		network_h1 = tl.layers.BatchNormLayer(network_h1, act=lambda x: tl.act.lrelu(x, 0.2),
											is_train=is_train,
											gamma_init=gamma_init,
											name='enc/h1/bn')
		network_h2 = tl.layers.Conv2dLayer(network_h1, act=tf.identity,
											W_init=w_init,
											shape=[5, 5, 128, 256],
											strides=[1, 2, 2, 1],
											padding='SAME',
											name='enc/h2/conv2d')
		network_h2 = tl.layers.BatchNormLayer(network_h2, act=lambda x: tl.act.lrelu(x, 0.2),
											is_train=is_train,
											gamma_init=gamma_init,
											name='enc/h2/bn')
		network_h3 = tl.layers.Conv2dLayer(network_h2, act=tf.identity,
											W_init=w_init,
											shape=[5, 5, 256, 512],
											strides=[1, 2, 2, 1],
											padding='SAME',
											name='enc/h3/conv2d')
		network_h3 = tl.layers.BatchNormLayer(network_h3, act=lambda x:tl.act.lrelu(x, 0.2),
											is_train=is_train,
											gamma_init=gamma_init,
											name='enc/h3/bn')
		network_h4 = tl.layers.FlattenLayer(network_h3, name='enc/h4/flatten')
		network_h4 = tl.layers.DenseLayer(network_h4, n_units=z_dim, act=tf.identity,
											W_init=w_init, name='enc/h4/lin')
		logits = network_h4.outputs
		network_h4.outputs = tf.nn.sigmoid(network_h4.outputs)

	return network_h4, logits

