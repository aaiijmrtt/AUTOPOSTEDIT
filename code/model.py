import sys, configparser
import encoder, decoder, atcoder
import tensorflow as tf

if __name__ == '__main__':
	config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
	config.read(sys.argv[1])

	model = dict()
	model = encoder.create(model, config['encoder'])
	model = decoder.create(model, config['atcoder'])
