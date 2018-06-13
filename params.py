import torch

class SmallParams(object):

	def __init__(self, img_size=75):
		# check if GPU is available to determine if dummy batches will be used
		CUDA = torch.cuda.is_available()

		if CUDA:
			batch_size = 64
		else: # for debugging/dummy training on CPU
			batch_size = 2

		# hyperparameters for training
		self.hyper_params = {	'batch_size':	batch_size,
								'lr': 			1e-4,		}

		# parameters for convolutional layers	
		conv1 = [3, 24, 3, 2, 1] # Conv2d params, [in_channels, out_channels, kernel_size, stride, pad]
		conv2 = [24, 24, 3, 2, 1]
		conv3 = [24, 24, 3, 2, 1]
		conv4 = [24, 24, 3, 2, 1]

		self.conv_params = [conv1, conv2, conv3, conv4]


		# parameters for g_theta()
		g1 = [(conv4[1]+2)*2+11, 256] # FC params, [in, out]
		g2 = [256, 256]
		g3 = [256, 256]
		g4 = [256, 256]

		self.g_params = [g1, g2, g3, g4]


		# parameters for f_phi()
		f1 = [256, 256] # FC params, [in, out]
		f2 = [256, 256]
		f3 = [256, 10]

		self.f_params = [f1, f2, f3]


		# parameters for MLP
		s = conv_out_size(self.conv_params, img_size)

		m1 = [conv4[1]*s**2+11, 256] # FC params, [in, out]
		m2 = [256, 256]
		m3 = [256, 256]
		m4 = [256, 256]
		m5 = [256, 256]
		m6 = [256, 256]
		m7 = [256, 10]

		self.mlp_params = [m1, m2, m3, m4, m5, m6, m7]


class BigParams(object):

	def __init__(self, img_size=75):
		# check if GPU is available to determine if dummy batches will be used
		CUDA = torch.cuda.is_available()

		if CUDA:
			batch_size = 64
		else: # for debugging/dummy training on CPU
			batch_size = 2

		# hyperparameters for training
		self.hyper_params = {	'batch_size':	batch_size,
								'lr': 			1e-4,		}

		# parameters for convolutional layers	
		conv1 = [3, 32, 3, 2, 1] # Conv2d params, [in_channels, out_channels, kernel_size, stride, pad]
		conv2 = [32, 64, 3, 2, 1]
		conv3 = [64, 128, 3, 2, 1]
		conv4 = [128, 256, 3, 2, 1]

		self.conv_params = [conv1, conv2, conv3, conv4]


		# parameters for g_theta()
		g1 = [(conv4[1]+2)*2+11, 2000] # FC params, [in, out]
		g2 = [2000, 2000]
		g3 = [2000, 2000]
		g4 = [2000, 2000]

		self.g_params = [g1, g2, g3, g4]


		# parameters for f_phi()
		f1 = [2000, 1000] # FC params, [in, out]
		f2 = [1000, 500]
		f3 = [500, 10]

		self.f_params = [f1, f2, f3]


		# parameters for MLP
		s = conv_out_size(self.conv_params, img_size)
		print(s)

		m1 = [conv4[1]*s**2+11, 2000] # FC params, [in, out]
		m2 = [2000, 2000]
		m3 = [2000, 2000]
		m4 = [2000, 2000]
		m5 = [2000, 1000]
		m6 = [1000, 500]
		m7 = [500, 10]

		self.mlp_params = [m1, m2, m3, m4, m5, m6, m7]

# iteratively calculate the size of output kernels for each
# convolutional layer and return the last one
def conv_out_size(conv_params, size=75):
	for c in conv_params:
		size = int( (size - c[2] + 2*c[4])/c[3] + 1 ) # out_size = floor((in_size - K + 2P)/S + 1)
	return size

######################################################################################################

def main():
	params = BigParams()
	print(params.hyper_params)
	print(params.conv_params)
	print(params.g_params)
	print(params.f_params)
	print(params.mlp_params)


if __name__ == '__main__':
	main()