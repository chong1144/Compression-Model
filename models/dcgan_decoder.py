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
		
