import torch
print('Using PyTorch version '+torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

import numpy as np

import sys
import os

# check if GPU is available
if torch.cuda.is_available():
	print('GPU available: will run on GPU')
	cuda = True
else:
	print('GPU unavailable: will run on CPU')
	cuda = False

# define CNN module (input to both RN and MLP)
class CNN_Module(nn.Module):
	def __init__(self):
		super(CNN_Module, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(3,24,3,stride=2,padding=1),
			nn.BatchNorm2d(24))
		self.layer2 = nn.Sequential(
			nn.Conv2d(24,24,3,stride=2,padding=1),
			nn.BatchNorm2d(24))
		self.layer3 = nn.Sequential(
			nn.Conv2d(24,24,3,stride=2,padding=1),
			nn.BatchNorm2d(24))
		self.layer4 = nn.Sequential(
			nn.Conv2d(24,24,3,stride=2,padding=1),
			nn.BatchNorm2d(24))

		# self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1) # 3 input channels(RGB), 24 3x3 kernels (output 24 38x38 feature maps)
		# self.batchnorm1 = nn.BatchNorm2d(24) # batch normalization on 24 feature maps
		# self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1) # (output 24 51x51 feature maps)
		# self.batchnorm2 = nn.BatchNorm2d(24)
		# self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1) # (output 24 5x55 feature maps)
		# self.batchnorm3 = nn.BatchNorm2d(24)
		# self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1) # (output 24 16x16 feature maps)
		# self.batchnorm4 = nn.BatchNorm2d(24)

	# forward pass of conv net
	def forward(self, im):
		x = self.layer1(im)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		# print('debug4')
		# x = self.conv1(im) # first convolution on image
		# print('debug5')
		# x = self.batchnorm1(x)
		# x = self.conv2(x)
		# x = self.batchnorm2(x)
		# x = self.conv3(x)
		# x = self.batchnorm3(x)
		# x = self.conv4(x)
		# x = self.batchnorm4(x)
		return x

# parent class for CNN+RN and CNN+MLP architectures
class Generic_Model(nn.Module):
	# takes name and set of hyperparameters as input
	def __init__(self, hyp, name):
		super(Generic_Model, self).__init__()
		self.name = name

	# method for training model, takes an input image, question, and label
	def train(self, im_in, q_in, label):
		self.optimizer.zero_grad()
		out = self.forward(im_in, q_in) # runs forward pass of child class
		criterion = nn.CrossEntropyLoss()
		loss = criterion(out, label) # compute cross entropy loss
		loss.backward() # compute gradients
		self.optimizer.step() # backpropagate
		pred = out.data.max(1)[1] # predicted answer
		correct = pred.eq(label.data).cpu().sum() # determine if model was correct (on the CPU)
		acc = correct * 100. / label.shape[0] # calculate accuracy
		return loss, acc

	# test model
	def test(self, im_in, q_in, label):
		output = self.forward(im_in, q_in) # run forward pass of child class
		pred = output.data.max(1)[1]
		correct = pred.eq(label.data).cpu().sum()
		acc = correct * 100. / label.shape[0]
		return acc

	# save model during training
	def save_model(self, epoch):
		torch.save(self.state_dict(), 'models/{}_epoch_{:02d}'.format(self.name, epoch))


# define relation network (RN)
# (inherits from parent class, Generic_Model)
class RelationNetwork(Generic_Model):
	def __init__(self, hyp):
		super(RelationNetwork, self).__init__(hyp, 'RN')

		self.conv = CNN_Module() # initialize conv net input module

		# define g function of relation network
		# input length is (#_filters_per_object + coord_of_object)*2 + question_vec
		self.g_fc1 = nn.Linear((24+2)*2+11, 256)

		self.g_fc2 = nn.Linear(256, 256)
		self.g_fc3 = nn.Linear(256, 256)
		self.g_fc4 = nn.Linear(256, 256)

		# define f function of relation network
		self.f_fc1 = nn.Linear(256, 256)
		self.f_fc2 = nn.Linear(256, 256)
		self.f_fc3 = nn.Linear(256, 10)



		self.coord_oi = torch.FloatTensor(hyp['batch_size'], 2) # FloatTensor to hold (batch_size) 2-dim coordinates of object o_i
		self.coord_oj = torch.FloatTensor(hyp['batch_size'], 2) # coordinates of object o_j

		# send coords to GPU if available
		if cuda:
			self.coord_oi = self.coord_oi.cuda()
			self.coord_oj = self.coord_oj.cuda()
		self.coord_oi = Variable(self.coord_oi)
		self.coord_oj = Variable(self.coord_oj)

		# create coordinate tensor
		def cvt_coord(i):
			return[(i/5-2)/2., (i%5-2)/2.]

		self.coord_tensor = torch.FloatTensor(hyp['batch_size'], 25, 2)

		# send coord_tensor to GPU if available
		if cuda:
			self.coord_tensor = self.coord_tensor.cuda()
		self.coord_tensor = Variable(self.coord_tensor)

		# initialize coordinate tensor with numpy
		np_coord_tensor = np.zeros((hyp['batch_size'], 25, 2))
		for i in range(25):
			np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
		self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor)) # copy numpy array over to torch Variable


		self.optimizer = optim.Adam(self.parameters(), lr=hyp['lr']) # use Adam gradient descent


	# forward pass of RN, takes batch of images and batch of SINGLE questions (not all) as input
	def forward(self, im, q):
		x = self.conv(im) # output shape (batch_size)x24x5x5

		# reshape (flatten) conv filters for FC layers
		batches = x.shape[0] # number of batches
		channels = x.shape[1] # number of channels
		d = x.shape[2] # size of filters is (d)x(d)


		# flatten filters to be 1d, then switch the 2nd and 3rd dimensions of reshaped x
		x_flat = x.view(batches, channels, d**2).permute(0,2,1) # shape (batch_size)x25x24

		# concatenate 2-d coordinates along last dimension
		x_flat = torch.cat([x_flat, self.coord_tensor],2) # shape (batch_size)x25x26


		q = torch.unsqueeze(q, 1) # insert dimension of size 1 after batch_size dimension
								  # (wraps question), shape (batch_size)x1x20x11
		q = q.repeat(1,25,1) # copy question for all 25 dims of kernel, shape (batch_size)x25x20x11
		q = torch.unsqueeze(q, 2) # wrap question, shape (batch_size)x25x1x11

		# pair up all possible o_i, o_j object pairs

		x_i = torch.unsqueeze(x_flat, 1) # insert dimension of 1 at axis=1, shape (batch_size)x1x25x26
		x_i = x_i.repeat(1,25,1,1) # repeat 25 times, shape (batch_size)x25x25x26

		x_j = torch.unsqueeze(x_flat, 2) # insert dimension of 1 at axis=2, shape (batch_size)x25x1x26
		x_j = torch.cat([x_j, q], 3) # concatenate question encoding along last axis, shape (batch_size)x25x1x37
		x_j = x_j.repeat(1,1,25,1) # repeat each object 25 times, to be matched up with all 25 other objects
								   # from x_i, shape (batch_size)x25x25x37


		x_full = torch.cat([x_i, x_j], 3) # concatenate object pairs, shape (batch_size)x25x25x63

		x_ = x_full.view(batches*(d**4), 63) # flatten into 1d vector
		x_ = self.g_fc1(x_)
		x_ = F.relu(x_)
		x_ = self.g_fc2(x_)
		x_ = F.relu(x_)
		x_ = self.g_fc3(x_)
		x_ = F.relu(x_)
		x_ = self.g_fc4(x_)
		x_ = F.relu(x_)

		x_g = x_.view(batches, d**4, 256)
		x_g = x_g.sum(1).squeeze()

		x_f = self.f_fc1(x_g)
		x_f = F.relu(x_f)
		x_f = self.f_fc2(x_f)
		x_f = F.relu(x_f)
		x_f = F.dropout(x_f)
		x_f = self.f_fc3(x_f)
		return x_f

def main():
	hyper = {	'batch_size': 2,
				'lr': 0.005		}

	model = RelationNetwork(hyper)
	print(model)

if __name__ == '__main__':
	main()