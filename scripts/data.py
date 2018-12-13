from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class IrisData:

	def __init__(self):
		# manually matched values based on visualization of experimental data set
		self.x_axis_min = 0
		self.x_axis_max = 5.5

	@staticmethod
	def get_experimental_data_set():
		full_iris_data_set = datasets.load_iris()
		X = full_iris_data_set["data"][:, (2, 3)]  # petal length, petal width
		y = full_iris_data_set["target"]

		# filtering data for experementation
		setosa_or_versicolor = (y == 0) | (y == 1)
		X = X[setosa_or_versicolor]
		y = y[setosa_or_versicolor]

		# scaling data
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)

		return X, X_scaled, y, scaler

	def show_visualization(self, X, y, scaler, clf_1, clf_2, clf_3):
		self._show_data(X, y)
		self._show_decision_boundary(scaler, clf_1, clf_name='LinearSVC', line_color='b')
		self._show_decision_boundary(scaler, clf_2, clf_name='SVC', line_color='r')
		self._show_decision_boundary(scaler, clf_3, clf_name='SGD', line_color='g')
		plt.legend(loc="lower right", fontsize=11)
		plt.show()

	def _show_data(self, X, y):
		plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris-Versicolor")
		plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris-Setosa")
		plt.axis([self.x_axis_min, self.x_axis_max, 0, 2])
		plt.xlabel("Petal length", fontsize=14)
		plt.ylabel("Petal width", fontsize=14)

	def _show_decision_boundary(self, scaler, clf_model, clf_name, line_color):

		# At the decision boundary wT x + b = 0, for 2 dimension - w0*x0 + w1*x1 + b = 0
		# => x1 = -w0/w1 * x0 - b/w1
		# b - bias
		# w - weights (model parameters)
		b = clf_model.intercept_[0]
		w = clf_model.coef_[0]

		# I don't quite understand why here -10/10 instead of 0/5.5 like on the graphic axis X
		x_min = -10
		x_max = 10
		x0 = np.linspace(x_min, x_max, 200)
		x1 = (-w[0]*x0 - b) / w[1]

		# for visualization data must be scaled back to initial scale
		m = len(x0)
		x0 = x0.reshape((m, 1))
		x1 = x1.reshape((m, 1))
		line = np.concatenate((x0, x1), axis=1)
		line = scaler.inverse_transform(line)

		plt.plot(line[:, 0], line[:, 1], line_color, label=clf_name)

