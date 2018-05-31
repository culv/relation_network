import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import h5py as h5

import os
import sys

class ToTensor(object):
	def __call__(self, sample):
		for data in sample.keys():
			if data == 'images':
				sample[data] = sample[data].transpose((2,0,1))
			sample[data] = torch.from_numpy(sample[data])

		return sample

class SortOfCLEVRDataset(Dataset):

	def __init__(self, data_full_path, transform=ToTensor(), train=True):
		self.data_full_path = data_full_path
		self.train = train
		self.transform = transform


	def __len__(self):
		d = h5.File(self.data_full_path, 'r') # load HDF5 dataset
		# get length of dataset
		if self.train:
			for dataset in d['train']:
				length = d['train'][dataset].shape[0]
				break
		else:
			for dataset in d['test']:
				length = d['test'][dataset].shape[0]
				break

		d.close()
		return length


	def __getitem__(self, idx):
		d = h5.File(self.data_full_path, 'r')
		
		if self.train:		group = d['train']
		else:				group = d['test']


		sample = {}
		for data in group.keys():
			sample[data] = group[data][idx]


		d.close()

		return self.transform(sample)

############################################################################

def main():
	curr_dir = os.path.dirname(os.path.realpath(__file__))
	data_path = os.path.join(curr_dir, 'data', 'sort-of-clevr.h5')

	dataset = SortOfCLEVRDataset(data_path)


	dl = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)


	for i_batch, sample_batch in enumerate(dl):
		print(i_batch, sample_batch['images'].shape)
		break


if __name__ == '__main__':
	main()