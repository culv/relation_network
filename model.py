import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

import numpy as np

from params import conv_out_size

import sys
import os


# define CNN module (input to both RN and MLP)
class CNN_Module(nn.Module):
	def __init__(self, conv_params):
		super(CNN_Module, self).__init__()


		c2 = conv_params[1]
		c3 = conv_params[2]
		c4 = conv_params[3]

		c = conv_params[0]
		self.layer1 = nn.Sequential(
			nn.Conv2d(c[0],c[1],c[2],stride=c[3],padding=c[4]),
			nn.ReLU(),
			nn.BatchNorm2d(c[1]))

		c = conv_params[1]
		self.layer2 = nn.Sequential(
			nn.Conv2d(c[0],c[1],c[2],stride=c[3],padding=c[4]),
			nn.ReLU(),
			nn.BatchNorm2d(c[1]))

		c = conv_params[2]
		self.layer3 = nn.Sequential(
			nn.Conv2d(c[0],c[1],c[2],stride=c[3],padding=c[4]),
			nn.ReLU(),
			nn.BatchNorm2d(c[1]))

		c = conv_params[3]
		self.layer4 = nn.Sequential(
			nn.Conv2d(c[0],c[1],c[2],stride=c[3],padding=c[4]),
			nn.ReLU(),
			nn.BatchNorm2d(c[1]))

	# forward pass of conv net
	def forward(self, im):
		x = self.layer1(im)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

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
		if not os.path.exists('./models'):
			os.makedirs('./models')
			print('Created models dir')
		torch.save(self.state_dict(), './models/{}_epoch_{:02d}'.format(self.name, epoch))

# define relation network (RN)
# (inherits from parent class, Generic_Model)
class RelationNetwork(Generic_Model):
	def __init__(self, hyp, conv_params, g_params, f_params):
		super(RelationNetwork, self).__init__(hyp, 'CNN+RN')

		self.conv = CNN_Module(conv_params) # initialize conv net input module

		gp = g_params
		fp = f_params

		# define g function of relation network
		# input length is (#_filters_per_object + coord_of_object)*2 + question_vec
		self.g = nn.Sequential(
			nn.Linear(gp[0][0], gp[0][1]),
			nn.ReLU(),
			nn.Linear(gp[1][0], gp[1][1]),
			nn.ReLU(),
			nn.Linear(gp[2][0], gp[2][1]),
			nn.ReLU(),
			nn.Linear(gp[3][0], gp[3][1]))

		# define f function of relation network
		self.f = nn.Sequential(
			nn.Linear(fp[0][0], fp[0][1]),
			nn.ReLU(),
			nn.Linear(fp[1][0], fp[1][1]),
			nn.Dropout(),
			nn.ReLU(),
			nn.Linear(fp[2][0], fp[2][1]))


		self.coord_oi = torch.FloatTensor(hyp['batch_size'], 2) # FloatTensor to hold (batch_size) 2-dim coordinates of object o_i
		self.coord_oj = torch.FloatTensor(hyp['batch_size'], 2) # coordinates of object o_j

		# send coords to GPU if available
		cuda = torch.cuda.is_available()
		if cuda:
			self.coord_oi = self.coord_oi.cuda()
			self.coord_oj = self.coord_oj.cuda()
		self.coord_oi = Variable(self.coord_oi)
		self.coord_oj = Variable(self.coord_oj)

		d = conv_out_size(conv_params) # size of output kernels from conv net

		# create coordinate tensor
		def cvt_coord(i):
			return[(i/d-2)/2., (i%d-2)/2.]

		self.coord_tensor = torch.FloatTensor(hyp['batch_size'], d**2, 2)

		# send coord_tensor to GPU if available
		if cuda:
			self.coord_tensor = self.coord_tensor.cuda()
		self.coord_tensor = Variable(self.coord_tensor)

		# initialize coordinate tensor with numpy
		np_coord_tensor = np.zeros((hyp['batch_size'], d**2, 2))
		for i in range(d**2):
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
		q = q.repeat(1,d**2,1) # copy question for all 25 dims of kernel, shape (batch_size)x25x20x11
		q = torch.unsqueeze(q, 2) # wrap question, shape (batch_size)x25x1x11

		# pair up all possible o_i, o_j object pairs

		x_i = torch.unsqueeze(x_flat, 1) # insert dimension of 1 at axis=1, shape (batch_size)x1x25x26
		x_i = x_i.repeat(1,d**2,1,1) # repeat 25 times, shape (batch_size)x25x25x26

		x_j = torch.unsqueeze(x_flat, 2) # insert dimension of 1 at axis=2, shape (batch_size)x25x1x26
		x_j = torch.cat([x_j, q], 3) # concatenate question encoding along last axis, shape (batch_size)x25x1x37
		x_j = x_j.repeat(1,1,d**2,1) # repeat each object 25 times, to be matched up with all 25 other objects
								   # from x_i, shape (batch_size)x25x25x37


		x_full = torch.cat([x_i, x_j], 3) # concatenate object pairs, shape (batch_size)x25x25x63

		x_g = self.g(x_full)


		x_g = x_g.view(batches, d**4, -1)
		x_g = x_g.sum(1).squeeze()

		x_f = self.f(x_g)

		return x_f


class CNN_MLP(Generic_Model):
	def __init__(self, hyp, conv_params, mlp_params):
		super(CNN_MLP, self).__init__(hyp, 'CNN+MLP')

		self.conv = CNN_Module(conv_params)

		p = mlp_params

		# MLP w/ same number of layers and neurons-per-layer as RN
		# input to MLP is flattened output of CNN
		# (batch_size)*(24 filters)*5*5
		self.MLP = nn.Sequential(
			nn.Linear(p[0][0], p[0][1]),
			nn.ReLU(),
			nn.Linear(p[1][0],p[1][1]),
			nn.ReLU(),
			nn.Linear(p[2][0],p[2][1]),
			nn.ReLU(),
			nn.Linear(p[3][0],p[3][1]),
			nn.ReLU(),
			nn.Linear(p[4][0],p[4][1]),
			nn.ReLU(),
			nn.Linear(p[5][0],p[5][1]),
			nn.ReLU(),
			nn.Linear(p[6][0],p[6][1]))

		self.optimizer = optim.Adam(self.parameters(), lr=hyp['lr']) # use Adam gradient descent

	def forward(self, im, q):
		x = self.conv(im)

		x_flat = x.view(im.shape[0], -1) # flatten conv kernels

		x_flat_and_q = torch.cat((x_flat, q), 1) # append question

		x_out = self.MLP(x_flat_and_q)

		return x_out

# calculate the number of trainable parameters in a model
def get_num_params(params):
	# filter out params that won't be trained
	params = filter(lambda p: p.requires_grad, params)
	num = sum(np.prod(p.shape) for p in params)
	return num

##############################################################################################

def main():
	print('Using PyTorch version '+torch.__version__)

	CUDA = torch.cuda.is_available()

	# check if GPU is available
	if CUDA:
		print('GPU available: will run on GPU')
	else:
		print('GPU unavailable: will run on CPU')


	hyper = {	'batch_size': 2,
				'lr': 0.005		}

	RN_model = RelationNetwork(hyper)
	print(RN_model)

	MLP_model = CNN_MLP(hyper)
	print(MLP_model)

	RN_params = RN_model.parameters()

	RN_num = get_num_params(RN_params)
	print('CNN+RN has {} parameters'.format(RN_num))

	MLP_params = MLP_model.parameters()
	MLP_num = get_num_params(MLP_params)
	print('CNN+MLP has {} parameters'.format(MLP_num))



if __name__ == '__main__':
	main()