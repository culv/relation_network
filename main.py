# import model, dataset, and data generator
from model import RelationNetwork, CNN_MLP
from dataset import SortOfCLEVRDataset
from generator import SortOfCLEVRGenerator, show_sample
from visdom_utils import VisdomLinePlotter

# import PyTorch & NumPy
import numpy as np
import torch
from torch.utils.data import DataLoader

# import Visdom to examine training
from visdom import Visdom
from tqdm import tqdm # progress meter for loops!

# import MatPlotLib for viewing dataset samples
import matplotlib as mpl
mpl.use('TkAgg') # use TkAgg backend to avoid segmentation fault
import matplotlib.pyplot as plt

import os
import sys
import time



############################################################################################################


def main():
	PORT = 7777
	CUDA = torch.cuda.is_available()
	EPOCHS = 4

	if CUDA:
		batch = 64
	else:
		batch = 2 # for debugging/dummy training on CPU

	# hyperparameters for RN model
	hyper = {	'batch_size': batch,
				'lr': 0.005		}


	# get full path to HDF5 file of data
	curr_dir = os.path.dirname(os.path.realpath(__file__))
	data_dir = os.path.join(curr_dir, 'data')
	data_fname = 'sort-of-clevr.h5'


	# create dataset of it doesn't already exist
	if not os.path.exists(data_dir):
		print('Creating dataset...')
		gen = SortOfCLEVRGenerator()
		if CUDA:		made_data = gen.create_dataset() # if on GPU make the whole 10k image set
		else:			made_data = gen.create_dataset(train_size=8, test_size=0) # small dummy dataset for CPU

		print('Saving dataset...')
		os.makedirs(data_dir)
		gen.save_dataset(made_data, data_dir, data_fname)


	full_data_path = os.path.join(data_dir, data_fname)
	dataset = SortOfCLEVRDataset(full_data_path)	# create pytorch Dataset object
	print('Dataset loaded')



	# create pytorch DataLoader object with proper batch size
	loader = DataLoader(dataset, batch_size=hyper['batch_size'], shuffle=True, num_workers=1)
	batch = next(iter(loader))	# grab next batch


	num_questions = batch['questions'].shape[1] # total number of questions per image


	iters = int(dataset.__len__() / hyper['batch_size']) # of batches per epoch


	model = RelationNetwork(hyper) # create CNN+RN
	model_MLP = CNN_MLP(hyper) # create CNN+MLP
	
	# use GPU if available
	if CUDA:
		model.cuda()
		model.MLP()
		print('GPU available - will default to using GPU')
	else:
		print('GPU unavailable - will default to using CPU')



	# start Visdom
	vis = Visdom(port=PORT)
	# check if Visdom server is available
	if vis.check_connection():
		print('Visdom server is online - will log data ')
	else:
		print('Visdom server is offline - will not log data')


	# create Visdom line plot for training loss
	loss_log = VisdomLinePlotter(vis, color='orange', title='Training Loss', ylabel='loss', xlabel='iters', linelabel='CNN+RN')
	loss_log.add_new(color='blue', linelabel='CNN+MLP')

	# create Visdom line plot for training accuracy
	acc_log = VisdomLinePlotter(vis, color='red', title='Training Accuracy', ylabel='accuracy (%)',
								xlabel='iters', linelabel='CNN+RN relational')
	acc_log.add_new(color='blue', linelabel='CNN+RN non-relational')
	acc_log.add_new(color='orange', linelabel='CNN+MLP relational')
	acc_log.add_new(color='green', linelabel='CNN+MLP non-relational')


	# training loop
	for epoch in range(EPOCHS):

		for it in tqdm(range(iters)):

			batch = next(iter(loader)) # load next batch

			imgs = batch['images'].float() # convert imgs to floats
			questions = batch['questions'].float() # convert questions to floats
			labels = torch.argmax(batch['labels'], 2) 	# convert from one-hot labels to class index
														# (required by torch's cross entropy loss)

			if CUDA:
				imgs = imgs.cuda()
				questions = questions.cuda()
				labels = labels.cuda()

			# every iteration, reset relational and nonrelational accuracy trackers
			rel_acc = 0
			nonrel_acc = 0
			MLP_rel_acc = 0
			MLP_nonrel_acc = 0

			for q in range(num_questions): # forward and backward pass of model on a batch of images and SINGLE question
				loss, acc = model.train(imgs, questions[:,q,:], labels[:,q])

				MLP_loss, MLP_acc = model_MLP.train(imgs, questions[:,q,:], labels[:,q])


				if q%2==0:
					rel_acc+=float(acc)/(num_questions/2)
					MLP_rel_acc+=float(MLP_acc)/(num_questions/2)
				else:
					nonrel_acc+=float(acc)/(num_questions/2)
					MLP_nonrel_acc+=float(MLP_acc)/(num_questions/2)

			if vis.check_connection():
				loss_log.update(it+epoch*iters, [float(loss), float(MLP_loss)])
				acc_log.update(it+epoch*iters, [rel_acc, nonrel_acc, MLP_rel_acc, MLP_nonrel_acc])

		print('[Epoch {:d}] loss={:.2f}, rel acc={:.2f}%, nonrel acc={:.2f}%'.format(epoch, float(loss), rel_acc, nonrel_acc))
		model.save_model(epoch)
		model_MLP.save_model(epoch)
		print('Saved model\n')


if __name__ == '__main__':
	main()