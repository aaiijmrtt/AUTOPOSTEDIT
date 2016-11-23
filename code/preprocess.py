import argparse
import datetime
import sys
import os

def filtertext(char):
	return char.isdigit() or char.isalpha() or char == ' '

def maptext(char):
	if char == '0':   return 'zero '
	elif char == '1': return 'one '
	elif char == '2': return 'two '
	elif char == '3': return 'three '
	elif char == '4': return 'four '
	elif char == '5': return 'five '
	elif char == '6': return 'six '
	elif char == '7': return 'seven '
	elif char == '8': return 'eight '
	elif char == '9': return 'nine '
	return char.lower()

def count(filename, vocab = dict()):
	for line in open(filename):
		for word in map(maptext, filter(filtertext, line.split())):
			if word in vocab: vocab[word] += 1
			else: vocab[word] = 1
	return vocab

def writetofile(filenamein, filenameout, vocab):
	with open(filenameout, 'w') as fileout:
		for line in open(filenamein):
			for word in map(maptext, filter(filtertext, line.split())):
				fileout.write(str(vocab.index(word) + 1) + ' ')
			fileout.write('\n')

def main():

	data_path = os.path.abspath('../code/data')
	if not os.path.exists(data_path):
		os.makedirs(data_path)
		
	parser = argparse.ArgumentParser(description = 'Build vocabulary of preprocessed files')
	parser.add_argument('-d', '--dir', action = 'store', default = data_path, type = str, 
				help = 'Root directory of preprocessed splits')
	args = parser.parse_args()
	
	dir_path = args.dir
	
	vocab = count(os.path.join(args.dir, 'dev/dev.src'))
	print datetime.datetime.now(), 'Read dev.src'
	vocab = count(os.path.join(args.dir, 'dev/dev.mt'))
	print datetime.datetime.now(), 'Read dev.mt'
	vocab = count(os.path.join(args.dir, 'dev/dev.pe'))
	print datetime.datetime.now(), 'Read dev.pe'
	vocab = count(os.path.join(args.dir, 'test/test.src'))
	print datetime.datetime.now(), 'Read test.src'
	vocab = count(os.path.join(args.dir, 'test/test.mt'), vocab)
	print datetime.datetime.now(), 'Read test.mt'
	vocab = count(os.path.join(args.dir, 'test/test.pe'), vocab)
	print datetime.datetime.now(), 'Read test.pe'
	vocab = count(os.path.join(args.dir, 'train/train.src'), vocab)
	print datetime.datetime.now(), 'Read train.src'
	vocab = count(os.path.join(args.dir, 'train/train.mt'), vocab)
	print datetime.datetime.now(), 'Read train.mt'
	vocab = count(os.path.join(args.dir, 'train/train.pe'), vocab)
	print datetime.datetime.now(), 'Read train.pe'
		
	vocab = map(lambda x: x[0], sorted(vocab.items(), key = lambda x: x[1], reverse = True))
	
	writetofile(os.path.join(args.dir, 'train/train.src'), 
		    os.path.join(args.dir, 'mapped_train.src'), vocab)
	writetofile(os.path.join(args.dir, 'train/train.mt'), 
		    os.path.join(args.dir, 'mapped_train.mt'), vocab)
	writetofile(os.path.join(args.dir, 'train/train.pe'), 
		    os.path.join(args.dir, 'mapped_train.pe'), vocab)
	print datetime.datetime.now(), 'Write mapped_train'
	writetofile(os.path.join(args.dir, 'dev/dev.src'), 
		    os.path.join(args.dir, 'mapped_dev.src'), vocab)
	writetofile(os.path.join(args.dir, 'dev/dev.mt'), 
		    os.path.join(args.dir, 'mapped_dev.mt'), vocab)
	writetofile(os.path.join(args.dir, 'dev/dev.pe'), 
		    os.path.join(args.dir, 'mapped_dev.pe'), vocab)
	print datetime.datetime.now(), 'Write mapped_dev'
	writetofile(os.path.join(args.dir, 'test/test.src'), 
		    os.path.join(args.dir, 'mapped_test.src'), vocab)
	writetofile(os.path.join(args.dir, 'test/test.mt'), 
		    os.path.join(args.dir, 'mapped_test.mt'), vocab)
	writetofile(os.path.join(args.dir, 'test/test.pe'), 
		    os.path.join(args.dir, 'mapped_test.pe'), vocab)
	print datetime.datetime.now(), 'Write mapped_test'
	
	# Merging the mapped files
	
	train_src = os.path.join(dir_path, 'mapped_train.src')
	train_mt = os.path.join(dir_path, 'mapped_train.mt')
	train_pe = os.path.join(dir_path, 'mapped_train.pe')
	merged_train = os.path.join(dir_path, 'merged_train.txt')
	
	dev_src = os.path.join(dir_path, 'mapped_dev.src')
	dev_mt = os.path.join(dir_path, 'mapped_dev.mt')
	dev_pe = os.path.join(dir_path, 'mapped_dev.pe')
	merged_dev = os.path.join(dir_path, 'merged_dev.txt')
	
	test_src = os.path.join(dir_path, 'mapped_test.src')
	test_mt = os.path.join(dir_path, 'mapped_test.mt')
	test_pe = os.path.join(dir_path, 'mapped_test.pe')
	merged_test = os.path.join(dir_path, 'merged_test.txt')


	tr_src = open(train_src, 'r').readlines()
	tr_mt = open(train_mt, 'r').readlines()
	tr_pe = open(train_pe, 'r').readlines()
	
	dv_src = open(dev_src, 'r').readlines()
	dv_mt = open(dev_mt, 'r').readlines()
	dv_pe = open(dev_pe, 'r').readlines()
	
	ts_src = open(test_src, 'r').readlines()
	ts_mt = open(test_src, 'r').readlines()
	ts_pe = open(test_src, 'r').readlines()
	
	f1 = open(merged_train, 'w')
	f2 = open(merged_dev, 'w')
	f3 = open(merged_test, 'w')
	
	for (l_src, l_mt, l_pe) in zip(tr_src, tr_mt, tr_pe):
		sentence = l_src.rstrip() + str('\t') + l_mt.rstrip() + str('\t') + l_pe
		f1.write(sentence)

	print datetime.datetime.now(), 'Write merged_train'

	for (l_src, l_mt, l_pe) in zip(dv_src, dv_mt, dv_pe):
		sentence = l_src.rstrip() + str('\t') + l_mt.rstrip() + str('\t') + l_pe
		f2.write(sentence)
	
	print datetime.datetime.now(), 'Write merged_dev'
		
	for (l_src, l_mt, l_pe) in zip(ts_src, ts_mt, ts_pe):
		sentence = l_src.rstrip() + str('\t') + l_mt.rstrip() + str('\t') + l_pe
		f3.write(sentence)

	print datetime.datetime.now(), 'Write merged_test'
	
	
	f1.close()
	f2.close()
	f3.close()
	
	time, vocab = 0, 0
	for filename in ['data/merged_train.txt', 'data/merged_test.txt', 'data/merged_dev.txt']:
		for line in open(filename, 'r'):
			src, mt, pe = line.split('\t')
			for words in [src, mt, pe]:
				wordlist = [int(word) for word in words.split()]
				if max(wordlist) > vocab:
					vocab = max(wordlist)
				if len(wordlist) > time:
					time = len(wordlist)

	print 'time', time, 'vocab', vocab

if __name__ == '__main__':
	main()
