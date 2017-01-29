import tensorflow as tf, numpy as np

def create(model, config):
	dim_v, dim_i, dim_d, dim_t, dim_b, dim_p = config.getint('vocab'), config.getint('wvec'), config.getint('depth'), config.getint('steps'), config.getint('batch'), config.getint('predictions')
	biencoder, samp, lrate, dstep, drate, optim, rfact, reg = config.getboolean('biencoder'), config.getint('samples'), config.getfloat('lrate'), config.getint('dstep'), config.getfloat('drate'), getattr(tf.train, config.get('optim')), config.getfloat('rfact'), getattr(tf.contrib.layers, config.get('reg'))

	with tf.name_scope('decoder'):
		with tf.name_scope('input'):
			model['dh_%i_%i' %(dim_d - 1, -1)] = tf.Variable(tf.random_uniform([1, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), name = 'dh_%i_%i' %(dim_d -1, -1)) # consider starting with all zeros

		with tf.name_scope('label'):
			for ii in xrange(dim_t):
				model['dyi_%i' %ii] = tf.placeholder(tf.int32, [dim_b], name = 'dyi_%i' %ii)
				model['dy_%i' %ii] = tf.add(tf.nn.embedding_lookup(model['We'], model['dyi_%i' %ii]), model['Be'], name = 'dy_%i' %ii)

		for i in xrange(dim_d):
			with tf.name_scope('inputgate_%i' %i):
				model['dWi_%i' %i] = tf.Variable(tf.random_uniform([dim_i, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), collections = [tf.GraphKeys.VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'dWi_%i' %i)
				model['dBi_%i' %i] = tf.Variable(tf.random_uniform([1, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), name = 'dBi_%i' %i)

			with tf.name_scope('forgetgate_%i' %i):
				model['dWf_%i' %i] = tf.Variable(tf.random_uniform([dim_i, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), collections = [tf.GraphKeys.VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'dWf_%i' %i)
				model['dBf_%i' %i] = tf.Variable(tf.random_uniform([1, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), name = 'dBf_%i' %i)

			with tf.name_scope('outputgate_%i' %i):
				model['dWo_%i' %i] = tf.Variable(tf.random_uniform([dim_i, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), collections = [tf.GraphKeys.VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'dWo_%i' %i)
				model['dBo_%i' %i] = tf.Variable(tf.random_uniform([1, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), name = 'dBo_%i' %i)

			with tf.name_scope('cellstate_%i' %i):
				model['dWc_%i' %i] = tf.Variable(tf.random_uniform([dim_i, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), collections = [tf.GraphKeys.VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'dWc_%i' %i)
				model['dBc_%i' %i] = tf.Variable(tf.random_uniform([1, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), name = 'dBc_%i' %i)

			if biencoder:
				with tf.name_scope('transferstate_%i' %i):
					model['dWt_%i' %i] = tf.Variable(tf.random_uniform([2 * dim_i, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), collections = [tf.GraphKeys.VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'dWt_%i' %i)
					model['dBt_%i' %i] = tf.Variable(tf.random_uniform([1, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), name = 'dBt_%i' %i)

			with tf.name_scope('hidden_%i' %i):
				model['dWz_%i' %i] = tf.Variable(tf.random_uniform([dim_i, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), collections = [tf.GraphKeys.VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'dWz_%i' %i)
				model['dBz_%i' %i] = tf.Variable(tf.random_uniform([1, dim_i], - np.sqrt(6. / dim_i), np.sqrt(6. / dim_i)), name = 'dBz_%i' %i)

			for ii in xrange(dim_t):
				with tf.name_scope('transfer_%i_%i' %(i, ii)):
					model['ect_%i_%i' %(i, ii)] = model['ec_%i_%i' %(i, ii)] if not biencoder else tf.add(tf.matmul(model['ec_%i_%i' %(i, ii)], model['dWt_%i' %i]), model['dBt_%i' %i], 'ect_%i_%i' %(i, ii))

		for ii in xrange(dim_t):
			for i in xrange(dim_d):
				with tf.name_scope('input_%i_%i' %(i, ii)):
					model['dx_%i_%i' %(i, ii)] = model['dh_%i_%i' %(dim_d - 1, ii - 1)] if i == 0 else model['dh_%i_%i' %(i - 1, ii)]

				with tf.name_scope('inputgate_%i_%i' %(i, ii)):
					model['di_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['dx_%i_%i' %(i, ii)], model['dWi_%i' %i]), model['dBi_%i' %i]), name = 'di_%i_%i' %(i, ii))

				with tf.name_scope('forgetgate_%i_%i' %(i, ii)):
					model['df_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['dx_%i_%i' %(i, ii)], model['dWf_%i' %i]), model['dBf_%i' %i]), name = 'df_%i_%i' %(i, ii))
	
				with tf.name_scope('outputgate_%i_%i' %(i, ii)):
					model['do_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['dx_%i_%i' %(i, ii)], model['dWo_%i' %i]), model['dBo_%i' %i]), name = 'do_%i_%i' %(i, ii))

				with tf.name_scope('cellstate_%i_%i' %(i, ii)):
					model['dcc_%i_%i' %(i, ii)] = model['ect_%i_%i' %(i, dim_t - 1)] if ii == 0 else model['dc_%i_%i' %(i, ii - 1)] # consider starting with all zeros
					model['dc_%i_%i' %(i, ii)] = tf.add(tf.mul(model['df_%i_%i' %(i, ii)], model['dcc_%i_%i' %(i, ii)]), tf.mul(model['di_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['dx_%i_%i' %(i, ii)], model['dWc_%i' %i]), model['dBc_%i' %i]))), name = 'dc_%i_%i' %(i, ii))

				with tf.name_scope('hidden_%i_%i' %(i, ii)):
					model['dz_%i_%i' %(i, ii)] = tf.add(tf.matmul(model['dc_%i_%i' %(i, ii)], model['dWz_%i' %i]), model['dBz_%i' %i], name = 'dz_%i_%i' %(i, ii))

				with tf.name_scope('output_%i_%i' %(i, ii)):
					model['dh_%i_%i' %(i, ii)] = tf.mul(model['do_%i_%i' %(i, ii)], tf.nn.tanh(model['dz_%i_%i' %(i, ii)]), name = 'dh_%i_%i' %(i, ii))

		with tf.name_scope('output'):
			for ii in xrange(dim_t):
				model['dh_%i' %ii] = model['dh_%i_%i' %(dim_d - 1, ii)]

		with tf.name_scope('meansquared'):
			for ii in xrange(dim_t):
				model['dmses_%i' %ii] = tf.select(tf.equal(model['dyi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.sub(model['dy_%i' %ii], model['dh_%i' %ii])), [1]), name = 'dmses_%i' %ii)
			model['dmses'] = tf.reduce_sum(tf.add_n([model['dmses_%i' %ii] for ii in xrange(dim_t)]), name = 'dmses')
			model['sdmses'] = tf.scalar_summary(model['dmses'].name, model['dmses'])

		with tf.name_scope('negativeloglikelihood'):
			for ii in xrange(dim_t):
				model['dnlls_%i' %ii] = tf.select(tf.equal(model['dyi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.nn.sampled_softmax_loss(model['We'], tf.zeros([dim_v], tf.float32), model['dh_%i' %ii], tf.reshape(model['dyi_%i' %ii], [dim_b, 1]), samp, dim_v), name = 'dnlls_%i' %ii)
			model['dnlls'] = tf.reduce_sum(tf.add_n([model['dnlls_%i' %ii] for ii in xrange(dim_t)]), name = 'dnlls')
			model['sdnlls'] = tf.scalar_summary(model['dnlls'].name, model['dnlls'])

		with tf.name_scope('predict'):
			for ii in xrange(dim_t):
				model['dp_%i' %ii] = tf.nn.top_k(tf.matmul(model['dh_%i' %ii], model['We'], transpose_b = True), dim_p, name = 'dp_%i' %ii)

	model['gsd'] = tf.Variable(0, trainable = False, name = 'gsd')
	model['lrd'] = tf.train.exponential_decay(lrate, model['gsd'], dstep, drate, staircase = False, name = 'lrd')
	model['reg'] = tf.contrib.layers.apply_regularization(reg(rfact), tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	model['tdmses'] = optim(model['lrd']).minimize(model['dmses'] + model['reg'], global_step = model['gsd'], name = 'tdmses')
	model['tdnlls'] = optim(model['lrd']).minimize(model['dnlls'] + model['reg'], global_step = model['gsd'], name = 'tdnlls')

	return model
