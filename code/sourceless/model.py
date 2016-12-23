import sys, configparser, datetime
import encoder, bicoder, decoder, atcoder
import tensorflow as tf

def prepad(unpadded, pad, size):
	if len(unpadded) == size:
		return unpadded
	return [pad] * (size - len(unpadded)) + unpadded

def postpad(unpadded, pad, size):
	if len(unpadded) == size:
		return unpadded
	return unpadded + [pad] * (size - len(unpadded))

def feed(model, config, filename):
	batch, length = config.getint('global', 'batchsize'), config.getint('global', 'timesize')
	prelist, postlist = list(), list()
	for line in open(filename):
		iniedit, preedit, postedit = line.split('\t')
#		preedit, postedit = line.split('\t')
		preedit, postedit = [int(pre) for pre in preedit.split()], [int(post) for post in postedit.split()]
		prelist.append(prepad(preedit, 0, length))
		postlist.append(postpad(postedit, 0, length))
		if len(prelist) == batch:
			feeddict = dict()
			feeddict.update({model['exi_%i' %i]: [prelist[ii][i] for ii in xrange(batch)] for i in xrange(length)})
			feeddict.update({model['dyi_%i' %i]: [postlist[ii][i] for ii in xrange(batch)] for i in xrange(length)})
			yield feeddict
			prelist, postlist = list(), list()

def run(model, config, session, summary, filename, train):
	iters, freq, time, total = config.getint('global', 'iterations') if train else 1, config.getint('global', 'frequency'), config.getint('global', 'timesize'), 0.
	for i in xrange(iters):
		for ii, feeddict in enumerate(feed(model, config, filename)):
			if train:
				val, t = session.run([model['dms'], model['tdms']], feed_dict = feeddict)
				total += val
				if (ii + 1) % freq == 0:
					summ = session.run(model['sdms'], feed_dict = feeddict)
					summary.add_summary(summ, model['gsdms'].eval())
					print datetime.datetime.now(), 'iteration', i, 'batch', ii, 'loss', total
			else:
				val = session.run([model['dp_%i' %iii] for iii in xrange(time)], feed_dict = feeddict)
				exps, outs = [feeddict[model['dyi_%i' %iii]] for iii in xrange(time)], [x[1] for x in val]
				print >> sys.stderr, outs
				for bexp, bout in zip(exps, outs):
					for exp, out in zip(bexp, bout):
						if exp == 0: continue
						if exp in out: total += 1

	return total

if __name__ == '__main__':
	config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
	config.read(sys.argv[1])

	model = dict()
#	model = encoder.create(model, config['encoder'])
	model = bicoder.create(model, config['bicoder'])
#	model = decoder.create(model, config['decoder'])
	model = atcoder.create(model, config['atcoder'])

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
#		tf.train.Saver().restore(sess, config.get('global', 'path'))
		summary = tf.train.SummaryWriter(config.get('global', 'logs'), sess.graph)

		print datetime.datetime.now(), 'training model'
		trainingloss = run(model, config, sess, summary, sys.argv[2], True)
		print datetime.datetime.now(), 'training loss', trainingloss
		print datetime.datetime.now(), 'testing model'
		testingaccuracy = run(model, config, sess, summary, sys.argv[3], False)
		print datetime.datetime.now(), 'testing accuracy', testingaccuracy

		tf.train.Saver().save(sess, config.get('global', 'path'))
