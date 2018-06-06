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


import os
import sys
import time



def show_acc_table(table, new_row):
	pass


############################################################################################################



def main():
	# port for Visdom server, cuda availability, number of epochs
	PORT = 7777
	CUDA = torch.cuda.is_available()
	EPOCHS = 1


	if CUDA:
		batch_size = 64
		log_freq = 50
	else:
		batch_size = 2 # for debugging/dummy training on CPU
		log_freq = 5

	# hyperparameters for RN model (from paper)
	hyper = {	'batch_size': batch_size,
				'lr': 1e-4		}


	# get full path to HDF5 file of data
	curr_dir = os.path.dirname(os.path.realpath(__file__))
	data_dir = os.path.join(curr_dir, 'data')
	data_fname = 'sort-of-clevr.h5'


	# create dataset of it doesn't already exist
	if not os.path.exists(data_dir):
		print('Creating dataset...')
		gen = SortOfCLEVRGenerator()
		if CUDA:		made_data = gen.create_dataset() # if on GPU make the whole 10k image set
		else:			made_data = gen.create_dataset(train_size=2, test_size=2) # small dummy dataset for CPU

		print('Saving dataset...')
		os.makedirs(data_dir)
		gen.save_dataset(made_data, data_dir, data_fname)


	# get dataset
	full_data_path = os.path.join(data_dir, data_fname)
	train_dataset = SortOfCLEVRDataset(full_data_path)	# create pytorch Dataset object for training set
	test_dataset = SortOfCLEVRDataset(full_data_path, train=False) # for testing set
	print('Dataset loaded')



	# create pytorch DataLoader object with proper batch size
	train_loader = DataLoader(train_dataset, batch_size=hyper['batch_size'], shuffle=True, num_workers=1)
	test_loader = DataLoader(test_dataset, batch_size=hyper['batch_size'], shuffle=True, num_workers=1)


	# since test batch will be reused, just need to load and send to GPU once outside of training loop
	test_batch = next(iter(test_loader))

	test_imgs = test_batch['images'].float() # convert imgs to floats
	test_questions = test_batch['questions'].float() # convert questions to floats
	test_labels = torch.argmax(test_batch['labels'], 2) 	# convert from one-hot labels to class index
												# (required by torch's cross entropy loss)

	if CUDA:
		test_imgs = test_imgs.cuda()
		test_questions = test_questions.cuda()
		test_labels = test_labels.cuda()


	num_questions = test_batch['questions'].shape[1] # total number of questions per image


	iters = int(train_dataset.__len__() / hyper['batch_size']) # of batches per epoch



	RN = RelationNetwork(hyper) # create CNN+RN
	MLP = CNN_MLP(hyper) # create CNN+MLP
	
	# use GPU if available
	if CUDA:
		RN.cuda()
		MLP.cuda()
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
	train_acc_log = VisdomLinePlotter(vis, color='red', title='Training Accuracy', ylabel='accuracy (%)',
								xlabel='iters', linelabel='RN rel')
	train_acc_log.add_new(color='blue', linelabel='RN nonrel')
	train_acc_log.add_new(color='orange', linelabel='MLP rel')
	train_acc_log.add_new(color='green', linelabel='MLP nonrel')

	# for testing accuracy
	test_acc_log = VisdomLinePlotter(vis, color='red', title='Testing Accuracy', ylabel='accuracy (%)',
								xlabel='iters', linelabel='RN rel')
	test_acc_log.add_new(color='blue', linelabel='RN nonrel')
	test_acc_log.add_new(color='orange', linelabel='MLP rel')
	test_acc_log.add_new(color='green', linelabel='MLP nonrel')

	# training loop
	start = time.time()
	for epoch in range(EPOCHS):

		for it in tqdm(range(iters)):

			train_batch = next(iter(train_loader)) # load next batch

			imgs = train_batch['images'].float() # convert imgs to floats
			questions = train_batch['questions'].float() # convert questions to floats
			labels = torch.argmax(train_batch['labels'], 2) 	# convert from one-hot labels to class index
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
				loss, acc = RN.train(imgs, questions[:,q,:], labels[:,q])

				MLP_loss, MLP_acc = MLP.train(imgs, questions[:,q,:], labels[:,q])

				# every (log_freq) batches or epoch, check train accuracy
				if (it+epoch*iters)%log_freq==0 or iters-it==1:
					if q%2==0:
						rel_acc+=float(acc)/(num_questions/2)
						MLP_rel_acc+=float(MLP_acc)/(num_questions/2)
					else:
						nonrel_acc+=float(acc)/(num_questions/2)
						MLP_nonrel_acc+=float(MLP_acc)/(num_questions/2)

			# every 50 batches, check test accuracy and log
			if (it+epoch*iters)%log_freq==0:
				# test set accuracy
				test_rel_acc = 0
				test_nonrel_acc = 0
				test_MLP_rel_acc = 0
				test_MLP_nonrel_acc = 0

				for q in range(num_questions):
					test_acc = RN.test(test_imgs, test_questions[:,q,:], test_labels[:,q])
					MLP_test_acc = MLP.test(test_imgs, test_questions[:,q,:], test_labels[:,q])

					if q%2==0:
						test_rel_acc+=float(test_acc)/(num_questions/2)
						test_MLP_rel_acc+=float(MLP_test_acc)/(num_questions/2)
					else:
						test_nonrel_acc+=float(test_acc)/(num_questions/2)
						test_MLP_nonrel_acc+=float(MLP_test_acc)/(num_questions/2)					

				if vis.check_connection():
					loss_log.update(it+epoch*iters, [float(loss), float(MLP_loss)])
					train_acc_log.update(it+epoch*iters, [rel_acc, nonrel_acc, MLP_rel_acc, MLP_nonrel_acc])
					test_acc_log.update(it+epoch*iters, [test_rel_acc, test_nonrel_acc, test_MLP_rel_acc, test_MLP_nonrel_acc])

		print('\n[       Epoch {:2d}        |      Training Accuracy      |      Testing Accuracy       ]'.format(epoch))
		print('[-----------------------|-----------------------------|-----------------------------]')
		print('[ CNN+RN  | Loss: {:5.2f} | rel: {:5.2f}%, nonrel: {:5.2f}% | rel: {:5.2f}%, nonrel: {:5.2f}% ]'.format(
			float(loss), rel_acc, nonrel_acc, test_rel_acc, test_nonrel_acc))
		print('[ CNN+MLP | Loss: {:5.2f} | rel: {:5.2f}%, nonrel: {:5.2f}% | rel: {:5.2f}%, nonrel: {:5.2f}% ]'.format(
			float(MLP_loss), MLP_rel_acc, MLP_nonrel_acc, test_MLP_rel_acc, test_MLP_nonrel_acc))

		RN.save_model(epoch)
		MLP.save_model(epoch)
		print('Saved model(s)\n')

	delta_t = time.time() - start
	print('Time to train: ', time.strftime('%H:%M:%S', time.gmtime(delta_t)))

if __name__ == '__main__':
	main()