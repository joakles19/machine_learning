import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class BivariableLinearRegression:
    """
    Simple linear regression with one feature.
    Finds y = m*x + b using gradient descent.

    Attributes:
        m: slope of the line
        b: intercept
        L: learning rate
        iterations: how many times to update m and b
        data: the dataset (two columns: x and y)
    """

    def __init__(self, dataset: pd.DataFrame, learning_rate: float = 0.001, iterations: int = 1000):
        self.data = dataset.dropna()
        self.L = learning_rate
        self.iterations = iterations
        self.m = 0
        self.b = 0

        self.__train()

    def __gradient_descent(self, m_current: float, b_current: float, data_points: pd.DataFrame, L: float) -> tuple[float, float]:
        """
        Do one step of gradient descent for m and b
        """
        n = len(data_points)
        x = data_points.iloc[:, 0].values
        y = data_points.iloc[:, 1].values
        y_pred = m_current * x + b_current

        m_grad = -(2/n) * np.dot(x, (y - y_pred))
        b_grad = -(2/n) * np.sum(y - y_pred)

        return m_current - L * m_grad, b_current - L * b_grad

    def __train(self):
        """Update m and b for the set number of iterations"""
        for _ in range(self.iterations):
            self.m, self.b = self.__gradient_descent(self.m, self.b, self.data, self.L)
        return self.m, self.b

    def predict(self, x_values: np.ndarray | list[float]) -> np.ndarray:
        """Return predicted y for given x values"""
        return self.m * np.array(x_values) + self.b

    def plot(self):
        """Make a scatter plot of the data and the regression line"""
        x = self.data.iloc[:, 0].values
        y = self.data.iloc[:, 1].values

        plt.scatter(x, y, color='blue', label='Data points')

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = self.predict(x_line)
        plt.plot(x_line, y_line, color='red', label='Regression line')

        plt.xlabel(self.data.columns[0])
        plt.ylabel(self.data.columns[1])
        plt.title('Linear Regression')
        plt.legend()
        plt.show()


class MultipleLinearRegression:
    """
    Linear regression with multiple features.
    Uses gradient descent to find weights.
    """

    def __init__(self, dataset: pd.DataFrame, learning_rate: float = 0.001, iterations: int = 1000):
        data = dataset.dropna()
        self.L = learning_rate
        self.iterations = iterations

        # Add bias column
        self.X = data.iloc[:, :-1].values
        self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
        self.y = data.iloc[:, -1].values.reshape(-1, 1)

        self.weights = np.zeros((self.X.shape[1], 1))

        self.__train()

    def __gradient_descent(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, L: float) -> np.ndarray:
        """
        Compute gradient and update weights
        """
        n = len(y)
        y_pred = X @ weights
        error = y - y_pred
        grad = -(2/n) * (X.T @ error)  # <-- fixed x -> X

        return weights - L * grad

    def __train(self):
        """Train the model using gradient descent"""
        for _ in range(self.iterations):
            self.weights = self.__gradient_descent(self.X, self.y, self.weights, self.L)

    def predict(self, x_vals: np.ndarray | list) -> np.ndarray:
        """Predict output for given inputs (x_vals can be 1D or 2D)"""
        x_vals = np.array(x_vals)

        if x_vals.ndim == 1:
            x_vals = x_vals.reshape(1, -1)

        x_vals = np.hstack((np.ones((x_vals.shape[0], 1)), x_vals))
        return x_vals @ self.weights


class LogisticLinearRegression:
    """
    Logistic regression for binary classification.
    Uses gradient descent to find weights.
    """

    def __init__(self, dataset: pd.DataFrame, learning_rate: float = 0.1, iterations: int = 100):
        data = dataset.dropna()
        self.L = learning_rate
        self.iterations = iterations

        self.X = data.iloc[:, :-1].values
        self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
        self.y = data.iloc[:, -1].values.reshape(-1, 1)

        self.weights = np.zeros((self.X.shape[1], 1))

        self.__train()

    def __sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def __calculate_gradient(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient of the loss"""
        n = len(y)
        predictions = self.__sigmoid(X @ weights)
        return X.T @ (predictions - y) / n

    def __gradient_descent(self, X: np.ndarray, y: np.ndarray, L: float, iterations: int) -> np.ndarray:
        """Update weights using gradient descent"""
        weights = self.weights.copy()
        for _ in range(iterations):
            grad = self.__calculate_gradient(weights, X, y)
            weights -= L * grad
        return weights

    def __train(self):
        """Train the logistic regression model"""
        self.weights = self.__gradient_descent(self.X, self.y, self.L, self.iterations)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for given X"""
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.__sigmoid(X @ self.weights)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return class predictions (0 or 1) for given X"""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)