import tensorflow as tf

def create(model, config, embedding = None):
	dim_v, dim_i, dim_d, dim_t, dim_b = config.getint('vocab'), config.getint('wvec'), config.getint('depth'), config.getint('steps'), config.getint('batch')
	lrate_ms, dstep_ms, drate_ms, optim_ms = config.getfloat('lrate'), config.getint('dstep'), config.getfloat('drate'), getattr(tf.train, config.get('optim'))

	with tf.name_scope('embedding'):
		model['We'] = tf.Variable(tf.truncated_normal([dim_v, dim_i], stddev = 1.0 / dim_i), name = 'We') if embedding is None else tf.Variable(embedding, name = 'We')
		model['Be'] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'Be')

	with tf.name_scope('bicoder'):
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
				model['eFWi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'eFWi_%i' %i)
				model['eFBi_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'eFBi_%i' %i)
				model['eBWi_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'eBWi_%i' %i)
				model['eBBi_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'eBBi_%i' %i)
				for ii in xrange(dim_t):
					model['eFi_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['ex_%i_%i' %(i, ii)], model['eFWi_%i' %i]), model['eFBi_%i' %i]), name = 'eFi_%i_%i' %(i, ii))
					model['eBi_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['ex_%i_%i' %(i, ii)], model['eBWi_%i' %i]), model['eBBi_%i' %i]), name = 'eBi_%i_%i' %(i, ii))

			with tf.name_scope('forgetgate_%i' %i):
				model['eFWf_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'eFWf_%i' %i)
				model['eFBf_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'eFBf_%i' %i)
				model['eBWf_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'eBWf_%i' %i)
				model['eBBf_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'eBBf_%i' %i)
				for ii in xrange(dim_t):
					model['eFf_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['ex_%i_%i' %(i, ii)], model['eFWf_%i' %i]), model['eFBf_%i' %i]), name = 'eFf_%i_%i' %(i, ii))
					model['eBf_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['ex_%i_%i' %(i, ii)], model['eBWf_%i' %i]), model['eBBf_%i' %i]), name = 'eBf_%i_%i' %(i, ii))

			with tf.name_scope('outputgate_%i' %i):
				model['eWo_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'eWo_%i' %i)
				model['eBo_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'eBo_%i' %i)
				for ii in xrange(dim_t):
					model['eo_%i_%i' %(i, ii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['ex_%i_%i' %(i, ii)], model['eWo_%i' %i]), model['eBo_%i' %i]), name = 'eo_%i_%i' %(i, ii))

			with tf.name_scope('cellstate_%i' %i):
				model['eFWc_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'eFWc_' + str(i))
				model['eFBc_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'eFBc_' + str(i))
				model['eBWc_%i' %i] = tf.Variable(tf.truncated_normal([dim_i, dim_i], stddev = 1.0 / dim_i), name = 'eBWc_' + str(i))
				model['eBBc_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'eBBc_' + str(i))
				for ii in xrange(dim_t):
					model['eFcc_%i_%i' %(i, ii)] = tf.Variable(tf.truncated_normal([dim_b, dim_i], stddev = 1.0 / dim_i), name = 'eFcc_%i_%i' %(i, ii)) if ii == 0 else model['eFc_%i_%i' %(i, ii - 1)] # consider starting with all zeros
					model['eFc_%i_%i' %(i, ii)] = tf.select(tf.equal(model['exi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['eFcc_%i_%i' %(i, ii)], tf.add(tf.mul(model['eFf_%i_%i' %(i, ii)], model['eFcc_%i_%i' %(i, ii)]), tf.mul(model['eFi_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['ex_%i_%i' %(i, ii)], model['eFWc_%i' %i]), model['eFBc_%i' %i])))), name = 'eFc_%i_%i' %(i, ii))
				for ii in reversed(xrange(dim_t)):
					model['eBcc_%i_%i' %(i, ii)] = tf.Variable(tf.truncated_normal([dim_b, dim_i], stddev = 1.0 / dim_i), name = 'eBcc_%i_%i' %(i, ii)) if ii == dim_t - 1 else model['eBc_%i_%i' %(i, ii + 1)] # consider starting with all zeros
					model['eBc_%i_%i' %(i, ii)] = tf.select(tf.equal(model['exi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['eBcc_%i_%i' %(i, ii)], tf.add(tf.mul(model['eBf_%i_%i' %(i, ii)], model['eBcc_%i_%i' %(i, ii)]), tf.mul(model['eBi_%i_%i' %(i, ii)], tf.nn.tanh(tf.add(tf.matmul(model['ex_%i_%i' %(i, ii)], model['eBWc_%i' %i]), model['eBBc_%i' %i])))), name = 'eBc_%i_%i' %(i, ii))
				for ii in xrange(dim_t):
					model['ec_%i_%i' %(i, ii)] = tf.concat(1, [model['eFc_%i_%i' %(i, ii)], model['eBc_%i_%i' %(i, ii)]], 'ec_%i_%i' %(i, ii))

			with tf.name_scope('hidden_%i' %i):
				model['eWz_%i' %i] = tf.Variable(tf.truncated_normal([2 * dim_i, dim_i], stddev = 1.0 / dim_i), name = 'eFWz_%i' %i)
				model['eBz_%i' %i] = tf.Variable(tf.truncated_normal([1, dim_i], stddev = 1.0 / dim_i), name = 'eFBz_%i' %i)
				for ii in xrange(dim_t):
					model['ez_%i_%i' %(i, ii)] = tf.add(tf.matmul(model['ec_%i_%i' %(i, ii)], model['eWz_%i' %i]), model['eBz_%i' %i], name = 'ez_%i_%i' %(i, ii))

			with tf.name_scope('output_%i' %i):
				for ii in xrange(dim_t):
					model['eh_%i_%i' %(i, ii)] = tf.mul(model['eo_%i_%i' %(i, ii)], tf.nn.tanh(model['ez_%i_%i' %(i, ii)]), name = 'eh_%i_%i' %(i, ii))
				model['eh_%i_%i' %(dim_d - 1, -1)] = tf.zeros([dim_b, dim_i], tf.float32)

		with tf.name_scope('output'):
			for ii in xrange(dim_t):
				model['eh_%i' %ii] = tf.select(tf.equal(model['exi_%i' %ii], tf.zeros([dim_b], tf.int32)), model['eh_%i_%i' %(dim_d - 1, ii - 1)], model['eh_%i_%i' %(dim_d - 1, ii)], name = 'eh_%i' %ii)

		with tf.name_scope('meansquared'):
			for ii in xrange(dim_t):
				model['ems_%i' %ii] = tf.select(tf.equal(model['exi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.sub(model['ey_%i' %ii], model['eh_%i' %ii])), [1]), name = 'ems_%i' %ii)
			model['ems'] = tf.reduce_sum(tf.add_n([model['ems_%i' %ii] for ii in xrange(dim_t)]), name = 'ems')
			model['sems'] = tf.scalar_summary(model['ems'].name, model['ems'])

	model['gsems'] = tf.Variable(0, trainable = False, name = 'gsems')
	model['lrems'] = tf.train.exponential_decay(lrate_ms, model['gsems'], dstep_ms, drate_ms, staircase = False, name = 'lrems')
	model['tems'] = optim_ms(model['lrems']).minimize(model['ems'], global_step = model['gsems'], name = 'tems')

	return model
