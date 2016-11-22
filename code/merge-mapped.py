import argparse
import os

data_path = os.path.abspath('../code/data')
parser = argparse.ArgumentParser(description = 'Merge the mapped files to tab separated files')
parser.add_argument('-d', '--dir', action = 'store', default = data_path, type = str, 
			help = 'Root directory of mapped files')
args = parser.parse_args()
dir_path = args.dir

train_mt = os.path.join(dir_path, 'mapped_train.mt')
train_pe = os.path.join(dir_path, 'mapped_train.pe')
merged_train = os.path.join(dir_path, 'merged_train.txt')

dev_mt = os.path.join(dir_path, 'mapped_dev.mt')
dev_pe = os.path.join(dir_path, 'mapped_dev.pe')
merged_dev = os.path.join(dir_path, 'merged_dev.txt')

t_mt = open(train_mt, 'r').readlines()
t_pe = open(train_pe, 'r').readlines()

d_mt = open(dev_mt, 'r').readlines()
d_pe = open(dev_pe, 'r').readlines()

f1 = open(merged_train, 'w')
f2 = open(merged_dev, 'w')

for (l_mt, l_pe) in zip(t_mt, t_pe):
	sentence = l_mt.rstrip() + str('\t') + l_pe
	f1.write(sentence)

for (l_mt, l_pe) in zip(d_mt, d_pe):
	sentence = l_mt.rstrip() + str('\t') + l_pe
	f2.write(sentence)
	
f1.close()
f2.close()
