#!usr/local/bin/python3
"""
Module providing functionality for linear regression using 2-variable gradient descend.
"""

import numpy

def gradient_descent(alpha: float, num_iterations: int, training_data: list) -> (float, float):
	"""
	Performs gradient descent with the given hyperparameters and training data, returning
	a tuple in the form of (theta_zero, theta_one) when complete.
	
	@param alpha the learning rate, between 0 and 1
	@param num_iterations the number of iterations to run run the algorithm for
	@param training_data the list of (x, y) tuples of training data
	
	@return the hypothesis learned from the training data in the form (theta_zero, theta_one)
	"""
	
	theta_zero = 0
	theta_one = 0
	m = len(training_data)
	y_prediction = lambda x : theta_zero + theta_one * x
	y_prediction_error = lambda x, y : y_prediction(x) - y
	derivative_theta_zero = lambda : sum([y_prediction_error(x, y) for x, y in training_data]) / m
	derivative_theta_one = lambda : sum([x * y_prediction_error(x, y) for x, y in training_data]) / m
	theta_gradient = lambda derivative : alpha * derivative()
	for i in range(num_iterations):
		theta_zero -= theta_gradient(derivative_theta_zero)
		theta_one -= theta_gradient(derivative_theta_one)
	return (theta_zero, theta_one)

if __name__ == "__main__":
	ALPHA = 0.001
	NUM_ITERATIONS = 1000
	training_x = numpy.array([0.69, 3, 8.3, -5, 2, 1.33, 5, 8, 3.5, 10.5])
	training_y = numpy.array([0.420, 6, 4.8, -0.5, 2, 8.1, -1, 9, 2.5, 5])
	training_data = [(training_x[i], training_y[i]) for i in range(len(training_x))]
	learned_hypothesis = gradient_descent(ALPHA, NUM_ITERATIONS, training_data)
	print("The learned hypothesis is: theta_zero =", round(learned_hypothesis[0], 2), "theta_one =", round(learned_hypothesis[1], 2))