import tensorflow as tf

def create(encoder1, encoder2, config, scope = 'combiner'):
	dim_i, dim_d, dim_t, dim_b, nonlinear = config.getint('wvec'), config.getint('depth'), config.getint('_steps_'), config.getint('batch'), getattr(tf.nn, config.get('nonlinear'))
	model = dict()

	with tf.name_scope(scope):
		for i in xrange(dim_d):
			with tf.name_scope('layer_%i' %i):
				model['cW_%i' %i] = tf.Variable(tf.truncated_normal([2 * dim_i, dim_i], stddev = 1.0 / dim_i), name = 'cW_%i' %i) if i == 0 else tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), collections = [tf.GraphKeys.REGULARIZATION_LOSSES], name = 'cW_%i' %i)
				model['cB_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'cB_%i' %i)
				model['cx_%i' %i] = tf.concat(1, [encoder1['eh_%i' %(dim_t - 1)], encoder2['eh_%i' %(dim_t - 1)]], name = 'cx_%i' %i) if i == 0 else model['cy_%i' %(i - 1)]
				model['cy_%i' %i] = tf.nn.sigmoid(tf.add(tf.matmul(model['cx_%i' %i], model['cW_%i' %i]), model['cB_%i' %i]), name = 'cy_%i' %i)

	return model
