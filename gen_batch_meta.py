import glob
import numpy as np
import sys
import random
import os

if __name__ == '__main__':
	files = glob.glob('505videos_cropped/*/*.npy')
	print len(files)
	for it in files[:5]:
		print it
	random.shuffle(files)

	num = len(files)
	ntrain = int(0.5*num)
	nval = int(0.1*num)
	ntest = num - ntrain - nval

	ftrains = [os.path.abspath(it) for it in files[:ntrain]]
	fvals = [os.path.abspath(it) for it in files[ntrain:ntrain+nval]]
	ftests = [os.path.abspath(it) for it in files[-ntest:]]

	labels = {}
	with open('video_classes.txt') as f:
		for line in f:
			vid, lbl = line.strip().split('\t')
			labels[vid] = float(lbl)
	meta = {}
	meta['train'] = ftrains
	meta['val'] = fvals
	meta['test'] = ftests
	meta['labels'] = labels

	np.save('meta.npy', [meta])