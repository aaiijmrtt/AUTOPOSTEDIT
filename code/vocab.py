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
	parser = argparse.ArgumentParser(description = 'Build vocabulary of preprocessed files')
	parser.add_argument('-d', '--dir', action = 'store', default = data_path, type = str, 
				help = 'Root directory of preprocessed splits')
	args = parser.parse_args()
	
	dir_path = args.dir
	vocab = count(os.path.join(args.dir, 'dev/dev.mt'))
	print datetime.datetime.now(), 'Read dev.mt'
	vocab = count(os.path.join(args.dir, 'dev/dev.pe'))
	print datetime.datetime.now(), 'Read dev.pe'
	vocab = count(os.path.join(args.dir, 'test/test.mt'), vocab)
	print datetime.datetime.now(), 'Read test.mt'
	vocab = count(os.path.join(args.dir, 'train/train.mt'), vocab)
	print datetime.datetime.now(), 'Read train.mt'
	vocab = count(os.path.join(args.dir, 'train/train.pe'), vocab)
	print datetime.datetime.now(), 'Read train.pe'
		
	vocab = map(lambda x: x[0], sorted(vocab.items(), key = lambda x: x[1], reverse = True))
	
	writetofile(os.path.join(args.dir, 'train/train.mt'), 
		    os.path.join(args.dir, 'mapped_train.mt'), vocab)
	writetofile(os.path.join(args.dir, 'train/train.pe'), 
		    os.path.join(args.dir, 'mapped_train.pe'), vocab)
	print datetime.datetime.now(), 'Write train'
	writetofile(os.path.join(args.dir, 'dev/dev.mt'), 
		    os.path.join(args.dir, 'mapped_dev.mt'), vocab)
	writetofile(os.path.join(args.dir, 'dev/dev.pe'), 
		    os.path.join(args.dir, 'mapped_dev.pe'), vocab)
	print datetime.datetime.now(), 'Write dev'
	writetofile(os.path.join(args.dir, 'test/test.mt'), 
		    os.path.join(args.dir, 'mapped_test.mt'), vocab)
	print datetime.datetime.now(), 'Write test'

if __name__ == '__main__':
	main()
