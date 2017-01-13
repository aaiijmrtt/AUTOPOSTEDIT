import sys, os, datetime, configparser
import numpy as np

def maptext(char):
	if char.isdigit(): return 'NUM'
	elif char.isalpha(): return char.lower()
	return char

def count(filename, vocab = dict()):
	for line in open(filename):
		for word in map(maptext, line.split()):
			if word in vocab: vocab[word] += 1
			else: vocab[word] = 1
	return vocab

def writetofile(filenamein, filenameout, vocab):
	with open(filenameout, 'w') as fileout:
		for line in open(filenamein):
			for word in map(maptext, line.split()):
				fileout.write(str(vocab.index(word) + 1) + ' ')
			fileout.write('\n')

def readindexfile(filename):
	maximum, vocabulary, reverse = 0, dict(), dict()
	for line in open(filename):
		index, token = line.split('\t')
		vocabulary[token.strip()] = int(index)
		reverse[int(index)] = token.strip()
		maximum = max(maximum, index)
	return vocabulary, reverse, maximum

def readalignfile(file1, file2, lengths):
	alignments1, alignments2 = list(), list()
	alignments = list()
	for line1, line2, length in zip(open(file1), open(file2), lengths):
		alignment1, alignment2 = [list() for _ in xrange(length)], [list() for _ in xrange(length)]
		for align in line1.strip().split(' '):
			left, right = align.split('-')
			alignment1[int(right)].append(left)
		for align in line2.strip().split(' '):
			left, right = align.split('-')
			alignment2[int(right)].append(left)
		alignments.append(';'.join([' '.join(align1) + ',' + ' '.join(align2) for align1, align2 in zip(alignment1, alignment2)]))
	return alignments

def readword2vec(filename, vocabulary = dict()):
	with open(filename) as filein:
		next(filein)
		for line in filein:
			tokens = line.split(' ')
			vocabulary[''.join(map(maptext, tokens[0].strip()))] = [float(dimension) for dimension in tokens[1: ]]
	return vocabulary

if __name__ == '__main__':
	config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
	config.read(sys.argv[1])

	directory, vocab = config.get('global', 'data'), dict()
	for folder in ['train', 'dev', 'test']:
		for filename in ['src', 'mt', 'pe']:
			vocab = count('%s/%s.%s' %(directory, folder, filename), vocab)
			print datetime.datetime.now(), 'read %s.%s' %(folder, filename)

	vocab = map(lambda x: x[0], sorted(vocab.items(), key = lambda x: x[1], reverse = True))
	with open('%s/map.dict' %directory, 'w') as fileout:
		for key, word in enumerate(vocab):
			fileout.write('%i\t%s\n' %(key + 1, word.strip()))
	print datetime.datetime.now(), 'write mapped dict'
	
	for folder in ['train', 'dev', 'test']:
		for filename in ['src', 'mt', 'pe']:
			writetofile('%s/%s.%s' %(directory, folder, filename), '%s/mapped.%s.%s' %(directory, folder, filename), vocab)
		print datetime.datetime.now(), 'write src mt pe', folder

	for folder in ['train', 'dev', 'test']:
		with open('%s/merged.%s' %(directory, folder), 'w') as fileout:
			for src, mt, pe in zip(*[open('%s/mapped.%s.%s' %(directory, folder, filename)) for filename in ['src', 'mt', 'pe']]):
				fileout.write('%s\t%s\t%s\n' %(src.strip(), mt.strip(), pe.strip()))
		print datetime.datetime.now(), 'write merged', folder

	time, vocab = 0, 0
	for filename in ['%s/merged.%s' %(directory, name) for name in ['train', 'test', 'dev']]:
		for line in open(filename):
			for words in line.split('\t'):
				if not words: continue
				wordlist = [int(word) for word in words.split()]
				vocab = max(vocab, max(wordlist or [0]))
				time = max(time, len(wordlist))
	print datetime.datetime.now(), 'time', time, 'vocab', vocab

	pelengths = [len(line.split()) for line in open('%s/mapped.train.pe' %directory).readlines()]
	alignments = readalignfile('%s/train.src-pe' %directory, '%s/train.mt-pe' %directory, pelengths)
	with open('%s/train' %directory, 'w') as fileout:
		for line, alignment in zip(open('%s/merged.train' %directory), alignments):
			fileout.write('%s\t%s\n' %(line.strip(), alignment.strip()))
	print datetime.datetime.now(), 'write train'

	for folder in ['dev', 'test']:
		with open('%s/%s' %(directory, folder), 'w') as fileout:
			for line in open('%s/merged.%s' %(directory, folder)):
				fileout.write('%s\t\n' %line.strip())
		print datetime.datetime.now(), 'write', folder

	word2vec, dimension = dict(), 300
	for language in ['english', 'german']:
		word2vec = readword2vec('%s/%s.model' %(directory, language), word2vec)
		print datetime.datetime.now(), 'read model', language

	indexvocab, reversevocab, indexmax = readindexfile('%s/map.dict' %directory)
	vocab = [np.zeros(dimension, float)]
	for word in indexvocab:
		if word in word2vec: vocab.append(word2vec[word])
		else: vocab.append(np.random.normal(0., np.sqrt(3. / dimension), dimension))
	np.savetxt('%s/model' %directory, np.vstack(vocab))
	print datetime.datetime.now(), 'write model'
