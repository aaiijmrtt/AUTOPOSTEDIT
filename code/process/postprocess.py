import sys, configparser
import numpy as np
import preprocess

def translate(config, filenumber):
	saves, batch, time = config.get('global', 'output'), config.getint('global', 'batchsize'), config.getint('global', 'timesize')
	loaded = np.load(open('%s/%i' %(saves, filenumber)))
	maximized, maximizer = np.sum(np.max(loaded['values'], 2), 1), np.argmax(loaded['values'], 2)
	return maximized, [[loaded['outputs'][i][ii][maximizer[i][ii]] for ii in xrange(time)] for i in xrange(batch)]

def ensemble(configs, filenumber):
	scores, sequences = list(), list()
	for config in configs:
		score, sequence = translate(config, filenumber)
		scores.append(score)
		sequences.append(sequence)
	return [sequences[select][ii] for ii, select in enumerate(np.argmax(scores, 0))]

if __name__ == '__main__':
	configs = list()
	for configfile in open(sys.argv[1]):
		config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
		config.read(configfile.strip())
		configs.append(config)

	vocab, reversevocab, maximum = preprocess.readindexfile(sys.argv[2])
	with open(sys.argv[3], 'w') as fileout:
		for filenumber in xrange(int(sys.argv[4])):
			for line in ensemble(configs, filenumber):
				fileout.write(' '.join([reversevocab[word] for word in line]) + '\n')
