# digitcv
Computer vision project for classifying digits.

Feed-forward artificial neural network implemented from scratch that classifies 28x28 MNIST digits. Neural network has a single hidden layer of 115 nodes. Input is a vector of the 784 pixel greyscale values for an image. Uses backpropagation and stochastic gradient descent, with Mean-Squared Error as loss function, to train neural network.

Training with learning rate of ``0.0001`` over ``10`` epochs with ``115`` nodes in the hidden layer resulted in an accuracy of ``0.857``.

Original MNIST data was taken from this [site](https://pjreddie.com/projects/mnist-in-csv/). Normalized dataset with pixel darkness values scaled between 0-1 can be downloaded [here](https://www.dropbox.com/s/8hl7g8e40s0elik/mnist_train.csv?dl=0) - file size is too large for github.

To predict the value of your own unknown digit, save the image file as a 28x28 pixels RGB PNG file in the same directory as ``run.py``. Edit ``fileName`` to be the image file, and then run ``run.py``.