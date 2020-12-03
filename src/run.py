# This script is for actually predicting images!
import pandas as pd
import numpy as np
import scipy.misc as smp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import json
from math import exp

class run:
	# this is the file to predict
	fileName = '../example-imgs/8.png'
	neuralNet = '../models/oldNN.txt'

	def __init__(self):
		#self.load_data()
		#self.draw2()
		self.loadNN(self.neuralNet)
		self.predictImg(self.fileName)
		#self.predictTen(100)

	def draw(self, index):
		# just testing the first image
		image = self.pixels[index]
		square = [[0 for r in range(28)] for c in range(28)]
		for i in range(len(image)):
			square[i/28][i%28] = image[i]

		imgplot = plt.imshow(square, cmap = 'Greys', interpolation = 'nearest')
		plt.show()

	def predictImg(self, img):
		print "Predicting digit classification of image:"
		im = Image.open(img, 'r')
		pix_val = list(im.getdata())
		# squares just for displaying the image
		square = [[0 for r in range(28)] for c in range(28)]
		vec = [0 for r in range(784)]
		for i in range(len(pix_val)):
			square[i/28][i%28] = pix_val[i][3]
			vec[i] = pix_val[i][3] / 255.0
		# uncomment this below line to see images before predicting
		self.show(square)

		digit = np.mat(vec)
		output1 = np.add(np.dot(digit, self.weights1), self.bias1)
		hiddenLayer = self.sigmoid(output1)
		output2 = np.add(np.dot(hiddenLayer, self.weights2), self.bias2)
		prediction = self.sigmoid(output2)

		print "Predicted to be a %s." % self.evalPredict(prediction)

	def evalPredict(self, predictionVector):
		index, maxi = 0, predictionVector.item(0, 0)
		for i in range(10):
			if predictionVector.item(0, i) > maxi:
				index = i
				maxi = predictionVector.item(0, i)
		return index

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

	def predictTen(self, num):
		total = 0
		for i in range(num):
			image = self.pixels[i]
			digit = np.mat(image)
			output1 = np.add(np.dot(digit, self.weights1), self.bias1)
			hiddenLayer = self.sigmoid(output1)
			output2 = np.add(np.dot(hiddenLayer, self.weights2), self.bias2)
			prediction = self.sigmoid(output2)
			guess =  self.evalPredict(prediction)
			if guess == float(self.labels[i]):
				total += 1
		print "Predicted %s correct out of %s." % (total, num)

	def loadNN(self, fileName):
		print 'Loading saved neural network...'
		with open(fileName) as save_file:
			nn = json.load(save_file)
		self.weights1 = nn['w1']
		self.weights2 = nn['w2']
		self.bias1 = nn['b1']
		self.bias2 = nn['b2']

	def load_data(self):
		print 'Loading data...'
		data = pd.read_csv('../data/train1.csv')

		values = data.values
		self.labels = values[:,0]
		self.pixels = values[:,1:]

	def draw2(self):
		im = Image.open('2.png', 'r')
		pix_val = list(im.getdata())
		square = [[0 for r in range(28)] for c in range(28)]
		for i in range(len(pix_val)):
			square[i/28][i%28] = pix_val[i][3]
		self.show(square)

	def rgb2gray(self, rgb):
		return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

	def show(self, image):
		imgplot = plt.imshow(image, cmap = 'Greys', interpolation = 'nearest')
		plt.show()

	def draw(self, index):
		#just testing the first image
		image = self.pixels[index]
		square = [[0 for r in range(28)] for c in range(28)]
		for i in range(len(image)):
			square[i/28][i%28] = image[i]

		imgplot = plt.imshow(square, cmap = 'Greys', interpolation = 'nearest')
		plt.show()

runProgram = run()