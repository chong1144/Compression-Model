import tensorflow as tf
from random import shuffle
import scipy.misc
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


def center_crop(x, crop_h, crop_w=None, resize_w=64):
	if crop_w is None:
		crop_w = crop_h
	h, w = x.shape[:2]
	j = int(round((h - crop_h) / 2.))
	i = int(round((w - crop_w) / 2.))
	return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
	if is_crop:
		cropped_image = center_crop(image, npx, resize_w=resize_w)
	else:
		cropped_image = image
	return np.array(cropped_image) / 127.5 - 1

def imread(path, is_grayscale=False):
	if is_grayscale:
		return scipy.misc.imread(path, flatten=True).astype(np.float)
	else:
		return scipy.misc.imread(path).astype(np.float)

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):
	return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)