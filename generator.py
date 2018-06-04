import numpy as np
from PIL import Image, ImageDraw
import h5py as h5 # for efficient database creation/access

import sys
import os

import matplotlib as mpl
mpl.use('TkAgg') # use TkAgg backend to avoid segmentation fault
import matplotlib.pyplot as plt

# generates and saves sort-of-CLEVR dataset
class SortOfCLEVRGenerator(object):
	colors = [
		(255,0,0),      # red
		(0,0,255),      # blue
		(0,255,0),      # green
		(255,165,0),    # orange
		(255,255,0),    # yellow
		(100,100,100)   # grey
	]

	color_lookup = ['red', 'blue', 'green', 'orange', 'yellow', 'grey']

	shapes = [
		's',    # square
		'c'    # circle
	]


	question_vector_size = 11 # size of encoded question vector
	answer_vector_size = 10 # one-hot vector of possible answers

	answer_lookup = ['yes', 'no', 'square', 'circle', '1', '2', '3', '4', '5', '6']

	def __init__(self, img_size=75, number_questions=10, number_shapes=6):
		self.img_size = img_size
		
		# shape size is rounded down img_size/7.5
		self.shape_size = int(img_size/5.)

		self.number_questions = number_questions # number of questions per category (relational, non-relational)
		self.number_shapes = number_shapes

	# make centers of shapes at random locations, non-overlapping
	def create_centers(self):
		centers = []
		for n in range(self.number_shapes):
			# flag to make sure shapes don't overlap
			collision = True
			while collision:
				# sample from range [shape_size, img_size-shape_size] to avoid shapes falling outside box
				center = np.random.randint(self.shape_size, self.img_size - self.shape_size, 2)
				# check squared distances for overlap
				collision = False
				for c in centers:
					if ((center-c)**2).sum() < (self.shape_size)**2:
						collision = True
			# once non-overlapping center is found, add it to the list
			centers.append(center)
		return centers

	# create numpy array of the image
	def create_image(self):
		# randomly create centers for non-overlapping shapes
		centers = self.create_centers()

		# randomly choose each shape
		choices = np.random.randint(len(self.shapes), size=self.number_shapes)

		# generate background and make PIL drawer
		img = Image.new('RGB', (self.img_size,)*2, color=(200,)*3)
		draw = ImageDraw.Draw(img)

		# state list holds information about state of image (center and shape type for each shape)
		state = []

		# generate a shape at each center
		for i, choice in enumerate(choices):
			center = centers[i]
			color = self.colors[i]
			# calculate bounding box for squares and circles (bottom left and top right corners)
			r = int(self.shape_size/2.*np.cos(np.pi/4.))
			bbox = tuple(center-r) + tuple(center+r)
			print(bbox)
			# if c==1, draw rectangle
			if choice:
				draw.rectangle(bbox, fill=color)
			# otherwise draw circle
			else:
				draw.ellipse(bbox, fill=color)

			state.append([center, choice]) # append shape information
		return np.array(img), state

	# create 11-bit question representation of the following form:
	# [red, blue, green, orange, yellow, gray, relational, non-relational, question 1, question 2, question 3]
	def create_questions(self):
		questions = []
		for q in range(self.number_questions):
			for r in range(2): # generate relational and non-relational for each iteration
				question = [0] * self.question_vector_size # list of 11 zeros
				color = np.random.randint(len(self.colors)) # choose random color
				question[color] = 1
				question[len(self.colors) + r] = 1 # choose relational/non-relational
				question_type = np.random.randint(3) # choose random question
				question[len(self.colors) + 2 + question_type] = 1
				questions.append(question)
		return questions

	# get one-hot vector of answers to each question
	# one-hot vector corresponds to: [yes, no, square, circle, 1, 2, 3, 4, 5, 6]
	# where 1-6 are the number of shapes with same shape
	def create_answers(self, state, questions):
		answers = []
		for question in questions:
			answer = [0] * self.answer_vector_size # list of 10 zeros
			color = question[:6].index(1) # grab the color of shape the question is asking about

			if question[6]: # if the question is relational

				if question[8]: # "What is the shape of the nearest object?"						
					# calculate squared distances to all shapes
					dist = [( (state[color][0]-obj[0]) **2).sum() for obj in state] 
					dist[dist.index(0)] = np.inf # since we're checking smallest distance, replace 0 with np.inf to ignore query shape
					closest = dist.index(min(dist))

					if state[closest][1]: # state[i][1] = 1 for square, 0 for circle
						answer[2] = 1 # answer[2] = 1 => square
					else:
						answer[3] = 1 # answer[3] = 1 => circle

				elif question[9]: # "What is the shape of the farthest object?"
					dist = [( (state[color][0]-obj[0]) **2).sum() for obj in state] 
					furthest = dist.index(max(dist))
					if state[furthest][1]:
						answer[2] = 1
					else:
						answer[3] = 1

				else: # "How many objects have the same shape?"

					count = 0 # initialize count
					shape = state[color][1] # get shape type

					for obj in state: # count number of those shapes
						if obj[1] == shape:
							count += 1
					answer[count + 3] = 1 # one-hot will get set to 1 at count+3 (+3 skips past [yes,no,square,circle])
				
				
			else: # if question is nonrelational

				if question[8]: # Is the shape a circle or a rectangle?
					if state[color][1]:
						answer[2] = 1
					else:
						answer[3] = 1

				elif question[9]: # Is the shape on the bottom of the image?
					if state[color][0][1] > self.img_size/2:
						answer[0] = 1 # answer[0] = 1 => yes
					else:
						answer[1] = 1 # answer[1] = 1 => no

				else: # Is the shape on the left of the image?
					if state[color][0][0] > self.img_size/2:
						answer[1] = 1
					else:
						answer[0] = 1
			answers.append(answer)
		return answers

	# create an (image, questions, labels) set of specified size
	def create_batch(self, size):
		# basic structure for set
		batch = {	'images': [],
					'questions': [],
					'labels': []}


		# create images, questions, labels
		for i in range(size):
			imgs, representation = self.create_image()
			batch['images'].append(imgs/255.) # convert images from [0, 255] int to [0,1] float
			batch['questions'].append(self.create_questions())
			batch['labels'].append(self.create_answers(representation, batch['questions'][i]))

		# convert lists to numpy arrays
		for key in batch.keys():
			batch[key] = np.array(batch[key])

		return batch

	# create a whole dataset
	# pass sizes of train, test, validation sets
	def create_dataset(self, train_size=9800, test_size=200, val_size=0):
		dataset = {'train': self.create_batch(train_size)}
		if test_size != 0:	dataset['test'] = self.create_batch(test_size)
		if val_size != 0: dataset['validation'] = self.create_batch(val_size)
		return dataset


	# save dataset efficiently in HDF5 binary format with h5py library
	def save_dataset(self, dataset_dict, data_dir='./data', fname='sort-of-clevr.h5'):
		# if data directory doesn't exist, make it
		if not os.path.exists(data_dir):
			os.makedirs(data_dir)

		# full path of file + HDF5 file extension
		full_fname = os.path.join(data_dir, fname)

		f = h5.File(full_fname, 'w') # open (data_dir)/(fname).hdf5 for writing

		# loop over sets (train, test, validation)		
		for key in dataset_dict.keys():
			group = f.create_group(key) # create group for each set
			# loop over data (images, questions, labels)
			for subkey in dataset_dict[key].keys():
				dataset = group.create_dataset(subkey, data=dataset_dict[key][subkey]) # save data

		f.close() # close file

# converts 11-bit question encoding and 10-bit one-hot answer vector to strings for human consumption
def bit2string(question, answer):
	# grab color of query shape
	c = SortOfCLEVRGenerator.color_lookup[question[:6].tolist().index(1)]

	# grab answer to question
	a = SortOfCLEVRGenerator.answer_lookup[answer.tolist().index(1)]

	# decode question
	if question[6]: # if question is relational
		if question[8]: 	q = 'What shape is the object nearest to the {} object?'.format(c)
		elif question[9]: 	q = 'What shape is the object farthest from the {} object?'.format(c)
		else: 				q = 'How many objects have the same shape as the {} object?'.format(c)
	else: # if question is non-relational
		if question[8]: 	q = 'What is the shape of the {} object?'.format(c)
		elif question[9]: 	q = 'Is the {} object on the bottom of the image?'.format(c)
		else: 				q = 'Is the {} object on the left of the image?'.format(c)

	return q, a

# check the structure of the HDF5 dataset
def check_h5_structure(h5_file):
	for grp in h5_file:
		print(h5_file[grp])
		for dset in h5_file[grp]:
			print('\t'+str(h5_file[grp][dset]))


# display a (size)x(size) grid of sample images labeled with a sample question and answer
def show_sample(sample, size=4):

	imgs = sample['images']
	questions = sample['questions']
	answers = sample['labels']

	# if size**2 is larger than number of samples, reduce the size
	num = imgs.shape[0]
	if size**2 > num:
		size = int(num**0.5)

	fig, axs = plt.subplots(size, size)
	axs = axs.reshape(-1) # flatten np.array of Axes objects

	for i, ax in enumerate(axs):
		# remove ticks and tick labels
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.tick_params(axis='both', which='both', length=0)

		# pick random relational and non-relational question/answer (alternate between)
		if np.mod(i, 2): # even plots (2nd, 4th, 6th, ...) pick random relational question
			rand_start = 0
		else: # odd plots (1st, 3rd, 5th, ...) pick random non-relational
			rand_start = 1


		rand_i = np.random.choice( np.arange(rand_start, questions.shape[1], 2) )
		q, a = bit2string(questions[i][rand_i], answers[i][rand_i])
		
		# display questions, answers, and images
		ax.set_title('Q: {}'.format(q))
		ax.set_xlabel('A: {}'.format(a))


		# if image is in pytorch format (CxHxW) convert it back to HxWxC
		try: 		img = imgs[i].numpy().transpose(((1,2,0)))
		except: 	img = imgs[i]

		ax.imshow(img)

	return fig, axs

##########################################################################################

def main():
	generator = SortOfCLEVRGenerator()#img_size=128)

	curr_dir = os.path.dirname(os.path.realpath(__file__))
	data_dir = os.path.join(curr_dir, 'data')

	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	make = True

	if make:
		dset = generator.create_dataset(train_size=32, test_size=0, val_size=0)

		generator.save_dataset(dset, data_dir=data_dir)


	file = os.path.join(data_dir, 'sort-of-clevr.h5')

	d = h5.File(file, 'r')

	check_h5_structure(d)


	figure, axes = show_sample(d['train'], size=4)

	d.close()

	plt.show()

if __name__ == '__main__':
	main()