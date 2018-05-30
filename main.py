from model import RelationNetwork
from dataset import SortOfCLEVRDataset
from generator import show_sample

import torch
from torch.utils.data import DataLoader

import matplotlib as mpl
mpl.use('TkAgg') # use TkAgg backend to avoid segmentation fault
import matplotlib.pyplot as plt

import os
import sys


def main():
	# hyperparameters for RN model
	hyper = {	'batch_size': 64,
				'lr': 0.005		}

	model = RelationNetwork(hyper)
	# params = model.parameters()
	# for i, p in enumerate(params):
	# 	print(i, p.shape)
	# sys.exit()


	# get full path to HDF5 file of data
	curr_dir = os.path.dirname(os.path.realpath(__file__))
	data_dir = os.path.join(curr_dir, '..', 'data')
	data_fname = 'sort-of-clevr.h5'
	full_data_path = os.path.join(data_dir, data_fname)


	dataset = SortOfCLEVRDataset(full_data_path)	# create pytorch Dataset object

	# create pytorch DataLoader object with proper batch size
	loader = DataLoader(dataset, batch_size=hyper['batch_size'], shuffle=True, num_workers=1)
	batch = next(iter(loader))	# grab next batch

	# fig, ax = show_sample(batch, size=4)
	# plt.show()

	imgs = batch['images'].float() # convert from double to float
	qs = batch['questions'].float()
	lbs = batch['labels']
	lbs = torch.argmax(lbs, 2) # convert from one-hot vectors to class indices (required by torch's cross entropy loss)


	rel_acc = 0
	nonrel_acc = 0
	for image in imgs:
		for q in range(20):
			loss, acc = model.train(imgs, qs[:,q,:], lbs[:,q])
			if q%2==0:	rel_acc+=acc/10.
			else:		nonrel_acc+=acc/10.
		print('relational accuracy:\t{}%'.format(rel_acc))
		rel_acc = 0
		print('nonrelational accuracy:\t{}%'.format(nonrel_acc))
		nonrel_acc = 0


if __name__ == '__main__':
	main()