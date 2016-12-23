import tensorflow as tf

def create(model, config):
	dim_v, dim_i, dim_d, dim_t, dim_b = config.getint('vocab'), config.getint('wvec'), config.getint('depth'), config.getint('steps'), config.getint('batch')
	lrate_ms, dstep_ms, drate_ms, optim_ms = config.getfloat('lrate'), config.getint('dstep'), config.getfloat('drate'), getattr(tf.train, config.get('optim'))

	with tf.name_scope('embedding'):
		model['We'] = tf.Variable(tf.truncated_normal([dim_v, dim_i], stddev = 1.0 / dim_i), name = 'We')
		model['Be'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'Be')

	with tf.name_scope('encoder'):
		with tf.name_scope('input'):
			for ii in xrange(dim_t):
				model['exi_%i' %ii] = tf.placeholder(tf.int32, [dim_b], name = 'exi_%i' %ii)
				model['ex_%i' %ii] = tf.add(tf.nn.embedding_lookup(model['We'], model['exi_%i' %ii]), model['Be'], name = 'ex_%i' %ii)

		with tf.name_scope('label'):
			for ii in xrange(dim_t):
				model['eyi_%i' %ii] = tf.placeholder(tf.int32, [dim_b], name = 'eyi_%i' %ii)
				model['ey_%i' %ii] = tf.add(tf.nn.embedding_lookup(model['We'], model['eyi_%i' %ii]), model['Be'], name = 'ey_%i' %ii)

		for i in xrange(dim_d):
			with tf.name_scope('input_%i' %i):
				for ii in xrange(dim_t):
					model['ex_%i_%i' %(i, ii)] = model['ex_%i' %ii] if i == 0 else model['eh_%i_%i' %(i - 1, ii)]

			with tf.name_scope('inputgate_%i' %i):
				model['eWi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'eWi_%i' %i)
				model['eBi_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'eBi_%i' %i)
				for ii in xrange(dim_t):
					model['ei_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['ex_%i_%i' %(i, ii)], model['eWi_%i' %i]), model['eBi_%i' %i]), name = 'ei_%i_%i' %(i, ii))

			with tf.name_scope('forgetgate_%i' %i):
				model['eWf_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'eWf_%i' %i)
				model['eBf_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'eBf_%i' %i)
				for ii in xrange(dim_t):
					model['ef_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['ex_%i_%i' %(i, ii)], model['eWf_%i' %i]), model['eBf_%i' %i]), name = 'ef_%i_%i' %(i, ii))

			with tf.name_scope('outputgate_%i' %i):
				model['eWo_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'eWo_%i' %i)
				model['eBo_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'eBo_%i' %i)
				for ii in xrange(dim_t):
					model['eo_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['ex_%i_%i' %(i, ii)], model['eWo_%i' %i]), model['eBo_%i' %i]), name = 'eo_%i_%i' %(i, ii))

			with tf.name_scope('cellstate_%i' %i):
				model['eWc_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'eWc_' + str(i))
				model['eBc_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'eBc_' + str(i))
				for ii in xrange(dim_t):
					model['ecc_%i_%i' %(i, ii)] = tf.Variable(tf.truncated_normal([dim_b, dim_i], stddev = 1.0 / dim_i), name = 'ecc_%i_%i' %(i, ii)) if ii == 0 else model['ec_%i_%i' %(i, ii - 1)] # consider starting with all zeros
					model['ec_%i_%i' %(i, ii)] = tf.add(tf.mul(model['ef_%i_%i' %(i, ii)], model['ecc_%i_%i' %(i, ii)]), tf.mul(model['ei_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['ex_%i_%i' %(i, ii)], model['eWc_%i' %i]), model['eBc_%i' %i]))), name = 'ec_%i_%i' %(i, ii))

			with tf.name_scope('hidden_%i' %i):
				model['eWz_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'eWz_%i' %i)
				model['eBz_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'eBz_%i' %i)
				for ii in xrange(dim_t):
					model['ez_%i_%i' %(i, ii)] = tf.add(tf.matmul(model['ec_%i_%i' %(i, ii)], model['eWz_%i' %i]), model['eBz_%i' %i], name = 'ez_%i_%i' %(i, ii))

			with tf.name_scope('output_%i' %i):
				for ii in xrange(dim_t):
					model['eh_%i_%i' %(i, ii)] = tf.mul(model['eo_%i_%i' %(i, ii)], tf.nn.tanh(model['ez_%i_%i' %(i, ii)]), name = 'eh_%i_%i' %(i, ii))

		with tf.name_scope('output'):
			for ii in xrange(dim_t):
				model['eh_%i' %ii] = model['eh_%i_%i' %(dim_d - 1, ii)]

		with tf.name_scope('meansquared'):
			for ii in xrange(dim_t):
				model['ems_%i' %ii] = tf.select(tf.equal(model['exi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.sub(model['ey_%i' %ii], model['eh_%i' %ii])), [1]), name = 'ems_%i' %ii)
			model['ems'] = tf.reduce_sum(tf.add_n([model['ems_%i' %ii] for ii in xrange(dim_t)]), name = 'ems')
			model['sems'] = tf.scalar_summary(model['ems'].name, model['ems'])

	model['gsems'] = tf.Variable(0, trainable = False, name = 'gsems')
	model['lrems'] = tf.train.exponential_decay(lrate_ms, model['gsems'], dstep_ms, drate_ms, staircase = False, name = 'lrems')
	model['tems'] = optim_ms(model['lrems']).minimize(model['ems'], global_step = model['gsems'], name = 'tems')

	return model
