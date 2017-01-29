import tensorflow as tf, numpy as np

def create(config, scope = 'embedder', embedding = None):
	dim_v, dim_i = config.getint('vocab'), config.getint('wvec')
	model = dict()

	with tf.name_scope(scope):
		model['We'] = tf.Variable(tf.random_uniform([dim_v, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), collections = [tf.GraphKeys.VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'We') if embedding is None else tf.Variable(embedding, collections = [tf.GraphKeys.VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'We')
		model['Be'] = tf.Variable(tf.random_uniform([1, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), name = 'Be')

	return model
