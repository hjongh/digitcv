# Actual implementation of NN
import pandas as pd
import numpy as np
from math import exp
from sklearn.model_selection import train_test_split
import json

class digitcv:
	# learning rate
	lr0 = 0.0001 # with 0.01 weight initiliazation
	lr  = 0.0001 # for 5
	lr2 = 0.00001 # for 5-10
	lr3 = 0.000001 # for 10-15
	# number of hidden nodes
	hiddenNodes = 30 #115 for 0.0001 BIG network
	# width and height of images
	width = 28
	# input size
	inputSize = 784
	# output size
	outputSize = 10
	# epochs
	epochs = 5 #10 is good
	# saved nn
	savedNN = '../models/oldNN.txt'
	# save location
	saveLoc = '../models/save.txt'

	def __init__(self):
		self.load_data()

		self.initialiseWeights()
		#self.loadNN(self.savedNN)

		self.accuracy(self.pixelsT, self.labelsT, 0)
		self.train(self.labels, self.pixels, self.epochs)
		self.save(self.saveLoc)

	def initialiseWeights(self):
		self.weights1 = self.initWeights(self.inputSize, self.hiddenNodes)
		self.weights2 = self.initWeights(self.hiddenNodes, self.outputSize)
		self.bias1 = self.initWeights(1, self.hiddenNodes)
		self.bias2 = self.initWeights(1, self.outputSize)

	def analyseTrainData(self):
		counts = [0 for x in range(10)]
		for label in self.labels:
			counts[int(label)] += 1
		for count in counts: print count

	def train(self, labels, pixels, epochs):
		print "starting training..."
		for epoch in range(epochs):
		#	if epoch == 10:
		#		self.lr = self.lr2
		#	elif epoch == 15:
		#		self.lr == self.lr3
			for i in range(len(labels)):
				# going through the network
				digit = np.mat(pixels[i])

				output1 = np.add(np.dot(digit, self.weights1), self.bias1)
				hiddenLayer = self.sigmoid(output1)
				output2 = np.add(np.dot(hiddenLayer, self.weights2), self.bias2)

				prediction = self.sigmoid(output2)
				actual = self.actual(labels[i])
				error = np.subtract(actual, prediction)

				# this should be negative, but we'll add it to the weights/biases to cancel that
				#errorPrime = self.sigmoidPrime(output2) #new
				#dweights2 = self.lr * np.dot(hiddenLayer.T, errorPrime)
				#dbias2 = self.lr * errorPrime
				#dHiddenLayer = np.multiply(self.sigmoidPrime(output1), np.dot(errorPrime, self.weights2.T))
				#dweights1 = self.lr * np.dot(digit.T, dHiddenLayer)
				#dbias1 = self.lr * dHiddenLayer

				# backpropagation!
				dweights2 = self.lr * np.dot(hiddenLayer.T, error)
				dbias2 = self.lr * error
				dHiddenLayer = np.multiply(self.sigmoidPrime(output1) , np.dot(error, self.weights2.T))
				dweights1 = self.lr * np.dot(digit.T, dHiddenLayer) #im confused
				dbias1 = self.lr * dHiddenLayer

				# updating the weights
				self.weights2 = np.add(self.weights2, dweights2)
				self.bias2 = np.add(self.bias2, dbias2)
				self.weights1 = np.add(self.weights1, dweights1)
				self.bias1 = np.add(self.bias1, dbias1)

				#print "Predicted: %s Actual: %s" % (self.evalPredict(prediction), labels[i])
			self.accuracy(self.pixelsT, self.labelsT, epoch + 1)
			#print "Epoch %s" % epoch
		# ok so at this point all our weights are trained
		print 'done training!'

	def save(self, location):
		print 'Saving neural network...'
		neural_net = {
			"w1": self.weights1.tolist(),
			"w2": self.weights2.tolist(),
			"b1": self.bias1[0].tolist(),
			"b2": self.bias2[0].tolist()
		}
		with open(location, 'w') as save_file:
			json.dump(neural_net, save_file)

	def loadNN(self, fileName):
		print 'Loading saved neural network...'
		with open(fileName) as save_file:
			nn = json.load(save_file)
		self.weights1 = nn['w1']
		self.weights2 = nn['w2']
		self.bias1 = nn['b1']
		self.bias2 = nn['b2']

	def printWeights(self):
		print self.weights1
		print self.weights2
		print self.bias1
		print self.bias2

	def actual(self, n):
		n = int(n)
		actual = np.zeros((1,10,))
		actual[0][n] = 1
		return actual

	def evalPredict(self, predictionVector):
		index, maxi = 0, predictionVector.item(0, 0)
		for i in range(10):
			if predictionVector.item(0, i) > maxi:
				index = i
				maxi = predictionVector.item(0, i)
		return index
		#prediction = np.nonzero(predictionVector[0] == max(predictionVector[0]))
		#return prediction[0][0]

	def accuracy(self, digits, labels, epochNum):
		counts = [0 for x in range(10)]
		totalCounts = [0 for y in range(10)]
		count = len(labels)
		true = 0
		for i in range(len(labels)):
			if (( self.predict( np.mat(digits[i]) ) + 0.0) == labels[i]):
				#print labels[i]
				true += 1
				counts[int(labels[i])] += 1
			totalCounts[int(labels[i])] += 1
		print "Epoch %s has accuracy of %.5f" % (epochNum, (true / float(count)))
		#for x in range(10):
		#	print "%s: %s / %s" % (x, counts[x], totalCounts[x])

	def predict(self, digit):
		output1 = np.add(np.dot(digit, self.weights1), self.bias1)
		hiddenLayer = self.sigmoid(output1)
		output2 = np.add(np.dot(hiddenLayer, self.weights2), self.bias2)
		prediction = self.sigmoid(output2)

		# print prediction
		return self.evalPredict(prediction)

	def normalize(self, digits):
		normalizedDigits = [[(digits[r][c] / 255.0) for c in range(len(digits[0]))] for r in range(len(digits))]
		return normalizedDigits

	def load_data(self):
		print 'Loading data...'
		data = pd.read_csv('../data/mnist_train.csv')
		test = pd.read_csv('../data/mnist_test.csv')

		values = data.values
		self.labels = values[:,0]
		self.pixels = values[:,1:]

		testValues = test.values
		self.labelsT = testValues[:,0]
		self.pixelsT = testValues[:,1:]
		#pixels = self.normalize(pixels)

		print 'Finished loading data.'

	def initWeights(self, rows, columns):
		weights = 0.1 * np.random.randn(rows, columns)
		weights2 = np.random.randn(rows, columns) / ((rows*columns)**0.5)
		return weights2

	def sigmoid(self, m):
		for r in range(len(m)):
			for c in range(len(m[0])):
				x = m.item((r,c))
				try:
					s = (1 / (1 + exp(-x) ) )
				except OverflowError:
					s = (1 / (1 + exp(-709)))
					#print "Value of %s too large for sigmoid." % x
				m.itemset((r, c), s )
		return m

	def sigmoidPrime(self, m):
		for r in range(len(m)):
			for c in range(len(m[0])):
				x = m.item((r,c))
				try:
					s = (exp(x) / (1+exp(x))**2)
				except OverflowError:
					s = (exp(709) / (1+exp(709))**2)
					#print "Value of %s too large for sigmoid prime." % x
				m.itemset((r, c) , s )
		return m

runProgram = digitcv()