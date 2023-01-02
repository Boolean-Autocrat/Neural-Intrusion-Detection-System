# Neural-Intrusion-Detection-System

This code assumes that you have a CSV file called intrusion_detection_data.csv that contains the data you want to use for training and testing. The code loads the data into a Pandas DataFrame, preprocesses it by encoding the labels and removing missing values, and then splits it into training and test sets.

Next, the code defines a simple feedforward neural network with one hidden layer containing 32 units, and a sigmoid activation function in the output layer. The model is compiled using the binary cross-entropy loss function and the Adam optimization algorithm.

Finally, the model is trained on the training data using the fit function and the resulting model is evaluated on the test set using the evaluate function. The code prints the loss and accuracy of the model on the test set.
