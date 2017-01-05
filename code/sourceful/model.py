import sys, configparser, datetime
import embedder, encoder, bicoder, combiner, decoder, atcoder, alcoder
import tensorflow as tf, numpy as np

def prepad(unpadded, pad, size):
	if len(unpadded) == size:
		return unpadded
	return [pad] * (size - len(unpadded)) + unpadded

def postpad(unpadded, pad, size):
	if len(unpadded) == size:
		return unpadded
	return unpadded + [pad] * (size - len(unpadded))

def feed(encoder1, encoder2, decoder, config, filename):
	batch, length, align = config.getint('global', 'batchsize'), config.getint('global', 'timesize'), config.getboolean('global', 'align')
	inilist, prelist, postlist, alignlist = list(), list(), list(), list()
	for line in open(filename):
		iniedit, preedit, postedit, alignment = line.split('\t')
		lengths = [len(preedit.split()), len(postedit.split())]
		if align:
			completealign, calign = postpad(alignment.split(';'), '', length), list()
			for i in xrange(length):
				calign.append(prepad(list(), -1., 2 * length))
				if not completealign[i].strip(): continue
				completealign[i] = completealign[i].split(',')
				for ii in xrange(2):
					if not completealign[i][ii].strip(): continue
					indexalign = [int(inalign) for inalign in completealign[i][ii].split(' ') if inalign]
					for idx in indexalign:
						calign[-1][ii * length + idx + length - lengths[ii]] = 1. / len(indexalign)
			alignlist.append(calign)
		inilist.append(prepad([int(ini) for ini in iniedit.split()], 0, length))
		prelist.append(prepad([int(pre) for pre in preedit.split()], 0, length))
		postlist.append(postpad([int(post) for post in postedit.split()], 0, length))
		if len(prelist) == batch:
			feeddict = dict()
			feeddict.update({encoder1['exi_%i' %i]: [inilist[ii][i] for ii in xrange(batch)] for i in xrange(length)})
			feeddict.update({encoder2['exi_%i' %i]: [prelist[ii][i] for ii in xrange(batch)] for i in xrange(length)})
			feeddict.update({decoder['dyi_%i' %i]: [postlist[ii][i] for ii in xrange(batch)] for i in xrange(length)})
			if align:
				feeddict.update({decoder['dai_%i' %i]: [[alignlist[ii][i][iii] for ii in xrange(batch)] for iii in xrange(2 * length)] for i in xrange(length)})
			yield feeddict
			inilist, prelist, postlist, alignlist = list(), list(), list(), list()

def run(encoder1, encoder2, decoder, config, session, summary, filename, train):
	iters, freq, time, saves, align, total, totaldmss, totaldmsa = config.getint('global', 'iterations') if train else 1, config.getint('global', 'frequency'), config.getint('global', 'timesize'), config.get('global', 'output'), config.getboolean('global', 'align'), 0., 0., 0.
	for i in xrange(iters):
		if train:
			for ii, feeddict in enumerate(feed(encoder1, encoder2, decoder, config, filename)):
				if align:
					valdmss, valdmsa, t = session.run([decoder['dmss'], decoder['dmsa'], decoder['tdms']], feed_dict = feeddict)
					totaldmss += valdmss
					totaldmsa += valdmsa
					if (ii + 1) % freq == 0:
						summs, summa = session.run([decoder['sdmss'], decoder['sdmsa']], feed_dict = feeddict)
						summary.add_summary(summs, decoder['gsdms'].eval())
						summary.add_summary(summa, decoder['gsdms'].eval())
						print datetime.datetime.now(), 'iteration', i, 'batch', ii, 'loss:', 'seq', totaldmss, 'att', totaldmsa
				else:
					val, t = session.run([decoder['dms'], decoder['tdms']], feed_dict = feeddict)
					total += val
					if (ii + 1) % freq == 0:
						summ = session.run(decoder['sdms'], feed_dict = feeddict)
						summary.add_summary(summ, decoder['gsdms'].eval())
						print datetime.datetime.now(), 'iteration', i, 'batch', ii, 'loss', total
		else:
			for ii, feeddict in enumerate(feed(encoder1, encoder2, decoder, config, filename)):
				val = session.run([decoder['dp_%i' %iii] for iii in xrange(time)], feed_dict = feeddict)
				exps, vals, outs = [feeddict[decoder['dyi_%i' %iii]] for iii in xrange(time)], [x[0] for x in val], [x[1] for x in val]
				np.savez(open('%s/%i' %(saves, ii), 'w'), values = np.transpose(np.array(vals), [1, 0, 2]), outputs = np.transpose(np.array(outs), [1, 0, 2]))
				for bexp, bout in zip(exps, outs):
					for exp, out in zip(bexp, bout):
						if exp == 0: continue
						if exp in out: total += 1

	if train and align:
		return totaldmss, totaldmsa
	else:
		return total

if __name__ == '__main__':
	config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
	config.read(sys.argv[1])

	print datetime.datetime.now(), 'creating model'
	embedder = embedder.create(config['embedder'])
#	encoder1 = encoder.create(embedder, config['encoder'])
#	encoder2 = encoder.create(embedder, config['encoder'])
	encoder1 = bicoder.create(embedder, config['bicoder'])
	encoder2 = bicoder.create(embedder, config['bicoder'])
	combiner = combiner.create(encoder1, encoder2, config['thinker'])
#	decoder_ = decoder.create(embedder, encoder1, encoder2, combiner, config['decoder'])
#	decoder_ = atcoder.create(embedder, encoder1, encoder2, combiner, config['atcoder'])
	decoder_ = alcoder.create(embedder, encoder1, encoder2, combiner, config['alcoder'])

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
#		tf.train.Saver().restore(sess, config.get('global', 'load'))
		summary = tf.train.SummaryWriter(config.get('global', 'logs'), sess.graph)

		print datetime.datetime.now(), 'training model'
		trainingloss = run(encoder1, encoder2, decoder_, config, sess, summary, sys.argv[2], True)
		print datetime.datetime.now(), 'training loss', trainingloss
		print datetime.datetime.now(), 'saving model'
#		tf.train.Saver().save(sess, config.get('global', 'save'))
		print datetime.datetime.now(), 'testing model'
		testingaccuracy = run(encoder1, encoder2, decoder_, config, sess, summary, sys.argv[3], False)
		print datetime.datetime.now(), 'testing accuracy', testingaccuracy
