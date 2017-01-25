import tensorflow as tf

def create(embedder, config, scope = 'encoder'):
	dim_v, dim_i, dim_d, dim_t, dim_b = config.getint('vocab'), config.getint('wvec'), config.getint('depth'), config.getint('steps'), config.getint('batch')
	samp, lrate, dstep, drate, optim = config.getint('samples'), config.getfloat('lrate'), config.getint('dstep'), config.getfloat('drate'), getattr(tf.train, config.get('optim'))
	model = dict()

	with tf.name_scope(scope):
		with tf.name_scope('input'):
			for ii in xrange(dim_t):
				model['exi_%i' %ii] = tf.placeholder(tf.int32, [dim_b], name = 'exi_%i' %ii)
				model['ex_%i' %ii] = tf.add(tf.nn.embedding_lookup(embedder['We'], model['exi_%i' %ii]), embedder['Be'], name = 'ex_%i' %ii)

		with tf.name_scope('label'):
			for ii in xrange(dim_t):
				model['eyi_%i' %ii] = tf.placeholder(tf.int32, [dim_b], name = 'eyi_%i' %ii)
				model['ey_%i' %ii] = tf.add(tf.nn.embedding_lookup(embedder['We'], model['eyi_%i' %ii]), embedder['Be'], name = 'ey_%i' %ii)

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
				model['emse_%i' %ii] = tf.select(tf.equal(model['exi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.reduce_sum(tf.square(tf.sub(model['ey_%i' %ii], model['eh_%i' %ii])), [1]), name = 'emse_%i' %ii)
			model['emse'] = tf.reduce_sum(tf.add_n([model['emse_%i' %ii] for ii in xrange(dim_t)]), name = 'emse')
			model['semse'] = tf.scalar_summary(model['emse'].name, model['emse'])

		with tf.name_scope('negativeloglikelihood'):
			for ii in xrange(dim_t):
				model['enll_%i' %ii] = tf.select(tf.equal(model['exi_%i' %ii], tf.zeros([dim_b], tf.int32)), tf.zeros([dim_b], tf.float32), tf.nn.sampled_softmax_loss(embedder['We'], tf.zeros([dim_v], tf.float32), model['eh_%i' %ii], tf.reshape(model['eyi_%i' %ii], [dim_b, 1]), samp, dim_v), name = 'enll_%i' %ii)
			model['enll'] = tf.reduce_sum(tf.add_n([model['enll_%i' %ii] for ii in xrange(dim_t)]), name = 'enll')
			model['senll'] = tf.scalar_summary(model['enll'].name, model['enll'])

	model['gse'] = tf.Variable(0, trainable = False, name = 'gse')
	model['lre'] = tf.train.exponential_decay(lrate, model['gse'], dstep, drate, staircase = False, name = 'lre')
	model['temse'] = optim(model['lre']).minimize(model['emse'], global_step = model['gse'], name = 'temse')
	model['tenll'] = optim(model['lre']).minimize(model['enll'], global_step = model['gse'], name = 'tenll')

	return model
