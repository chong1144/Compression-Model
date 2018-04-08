import tensorflow as tf
import numpy as np
from functools import reduce

class LossCalculator:
	def __init__(self, vgg, stylized_image):
		self.vgg = vgg
		self.transform_loss_net = vgg.net(vgg.preprocess(stylized_image))

	def content_loss(self, content_input_patch, content_layer, content_weight):
		