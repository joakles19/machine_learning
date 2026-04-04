import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class LinearRegression:
    def __init__(self, dataset: pd.DataFrame, learning_rate=0.001, iterations=1000):
        self.data = dataset.dropna()
        self.L = learning_rate
        self.iterations = iterations
        self.m = 0
        self.b = 0

        self.__train()

    def __gradient_descent(self, m_current, b_current, data_points: pd.DataFrame, L):
        n = len(data_points)
        x = data_points.iloc[:, 0].values
        y = data_points.iloc[:, 1].values
        y_pred = m_current * x + b_current

        m_grad = -(2/n) * np.dot(x, (y - y_pred))
        b_grad = -(2/n) * np.sum(y - y_pred)

        m_new = m_current - L * m_grad
        b_new = b_current - L * b_grad

        return m_new, b_new

    def __train(self):
        """Train the linear regression model using gradient descent"""
        for _ in range(self.iterations):
            self.m, self.b = self.__gradient_descent(self.m, self.b, self.data, self.L)
        return self.m, self.b

    def predict(self, x_values):
        """Predict y for given x values"""
        return self.m * np.array(x_values) + self.b

    def plot(self):
        """Plot the data and the regression line"""
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

if __name__ == "__main__":
    df = pd.read_csv("data\StudentPerformanceFactors.csv")
    df = df[["Hours_Studied", "Exam_Score"]]

    exam_regression = LinearRegression(df, 0.001, 50000)
    exam_regression.plot()

    print(exam_regression.predict(10))