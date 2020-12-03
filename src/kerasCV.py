# Testing with keras. Unrelated to my implementation
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
import numpy as numpy
from keras.utils import to_categorical
from keras import optimizers

"""
Some helpful links:
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
https://keras.io/getting-started/sequential-model-guide/
"""

class keras:

	def __init__(self):
		self.load_data()
		self.load_model()
		self.train()
		self.accuracy()

	def accuracy(self):
		scores = self.model.evaluate(self.pixelsT, self.labelsT)
		print ("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

	def train(self):
		self.model.fit(self.pixels, self.labels, epochs=10)

	def load_model(self):
		# sgd performs lot worse than adam?
		sgd = optimizers.SGD(lr = 0.1, momentum = 0.0, decay = 0.0, nesterov = False)

		model = Sequential()
		model.add(Dense(30, input_dim = 784, activation = 'sigmoid'))
		model.add(Dense(10, activation = 'sigmoid'))
		# some reason binaryCE does better than categoricalCE and MSE
		# adam performs BOUNDS better than sgd.
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
		# BIG NOTE! SGD with MSE (basically error) performs about exactly as mine. slightly better

		self.model = model

	def load_data(self):
		print 'Loading data...'
		data = pd.read_csv('../data/mnist_train.csv')
		test = pd.read_csv('../data/mnist_test.csv')

		values = data.values
		labels = values[:,0]
		self.labels = to_categorical(labels)
		self.pixels = values[:,1:]

		testValues = test.values
		labelsT = testValues[:,0]
		self.labelsT = to_categorical(labelsT)
		self.pixelsT = testValues[:,1:]

		print 'Finished loading data.'

runProgram = keras()