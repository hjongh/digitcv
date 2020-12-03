# This script is for transforming the csv data
import pandas as pd
import csv

class csv_data:
	def __init__(self):
		print 'Loading data...'
		data = pd.read_csv('../data/mnist_test.csv')

		values = data.values
		labels = values[:,0]
		pixels = values[:,1:]

		print 'Transforming data...'
		pixels = self.normalize(pixels)

		print 'Writing data...'

		with open('../data/mnist_test1.csv', 'wb') as file:
			writer = csv.writer(file)
			for i in range(len(labels)):
				row = []
				row.append(labels[i])
				for pixel in pixels[i]:
					row.append(pixel)
				writer.writerow(row)

		print 'Finished.'

	def normalize(self, digits):
		normalizedDigits = [[(digits[r][c] / 255.0) for c in range(len(digits[0]))] for r in range(len(digits))]
		return normalizedDigits

runProgram = csv_data()
