import tensorflow as tf
import tensorlayer as tl

def model(inputs, image_size=64, c_dim=3, batch_size=64, is_train=False, reuse=False):
	s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16)
	gf_dim = 64
	w_init = tf.random_normal_initializer(stddev=0.02)
	gamma_init = tf.random_normal_initializer(1., stddev=0.02)
	with tf.variable_scope('generator', reuse=reuse):
		tl.layers.set_name_reuse(reuse)

		network_in = tl.layers.InputLayer(inputs, name='dec/in')
		network_h0 = tl.layers.DenseLayer(network_in, n_units=gf_dim*8*s16*s16,
											W_init=w_init,
											act=tf.identity,
											name='dec/h0/lin')
		network_h0 = tl.layers.ReshapeLayer(network_h0, 
											shape=[-1, s16, s16, gf_dim*8],
											name='dec/h0/reshape')
		network_h0 = tl.layers.BatchNormLayer(network_h0, act=tf.nn.relu,
											is_train=is_train,
											gamma_init=gamma_init,
											name='dec/h0/bn')
		network_h1 = tl.layers.DeConv2dLayer(network_h0, act=tf.identity,
											shape=[5, 5, gf_dim*4, gf_dim*8],
											output_shape=[batch_size, s8, s8, gf_dim*4],
											strides=[1, 2, 2, 1],
											padding='SAME',
											name='dec/h1/deconv')
		network_h1 = tl.layers.BatchNormLayer(network_h1, act=tf.nn.relu,
											is_train=is_train,
											gamma_init=gamma_init,
											name='dec/h1/bn')
		network_h2 = tl.layers.DeConv2dLayer(network_h1, act=tf.identity,
											shape=[5, 5, gf_dim*2, gf_dim*4],
											output_shape=[batch_size, s4, s4, gf_dim*2],
											strides=[1, 2, 2, 1],
											padding='SAME',
											name='dec/h2/deconv')
		network_h2 = tl.layers.BatchNormLayer(network_h2, act=tf.nn.relu,
											is_train=is_train,
											gamma_init=gamma_init,
											name='dec/h2/bn')
		network_h3 = tl.layers.DeConv2dLayer(network_h2, act=tf.identity,
											shape=[5, 5, gf_dim, gf_dim*2],
											output_shape=[batch_size, s2, s2, gf_dim],
											strides=[1, 2, 2, 1],
											padding='SAME',
											name='dec/h3/deconv')
		network_h3 = tl.layers.BatchNormLayer(network_h3, act=tf.nn.relu,
											is_train=is_train,
											gamma_init=gamma_init,
											name='dec/h3/bn')
		network_h4 = tl.layers.DeConv2dLayer(network_h3, act=tf.identity,
											shape=[5, 5, c_dim, gf_dim],
											output_shape=[batch_size, image_size, image_size, c_dim],
											strides=[1, 2, 2, 1],
											padding='SAME',
											name='dec/h4/deconv')
		logits = network_h4.outputs
		network_h4.outputs = tf.nn.tanh(network_h4.outputs)
	return network_h4, logits