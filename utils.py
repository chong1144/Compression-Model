import tensorflow as tf
import numpy as np
from functools import reduce

class LossCalculator:
	def __init__(self, vgg, stylized_image):
		self.vgg = vgg
		self.transform_loss_net = vgg.net(vgg.preprocess(stylized_image))

	def content_loss(self, content_input_patch, content_layer, content_weight):
		content_loss_net = self.vgg.net(self.vgg.preprocess(content_input_patch))
		return tf.nn.l2_loss(content_loss_net[content_layer] \
			- self.transform_loss_net[content_layer]) / (_tensor_size(content_loss_net[content_layer]))


def _tensor_size(tensor):
	from operator import mul
	return reduce(mul, (d.value for d in tensor.get_shape()), 1)