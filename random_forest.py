import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, depth=0, value=None):
        self.feature = feature
        self.threshold = threshold
        self.depth = depth

        self.left_node = left
        self.right_node = right

        self.value = value

    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, max_depth=10, min_sample_split=2):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.root = None

    def fit_data(self, dataset):
        X, y = dataset[:, :-1], dataset[:, -1]

        self.root = self.__build_tree(X, y, depth=0)

    def __build_tree(self, X, y, depth):
        num_samples, num_features = X.shape

        if len(np.unique(y)) == 1 or depth >= self.max_depth or num_samples < self.min_sample_split:
            leaf_value = self.__majority_class(y)

            return Node(value=leaf_value, depth=depth)
        
        data = np.column_stack((X, y))
        split = self.__best_split(data, num_samples, num_features)

        if not split or split['info gain'] <= 0:
            return Node(value=self.__majority_class(y), depth=depth)
        
        left = self.__build_tree(split['data left'][:, :-1], split['data left'][:, -1], depth + 1)
        right = self.__build_tree(split['data right'][:, :-1], split['data right'][:, -1], depth + 1)

        return Node(split['index'], split['threshold'], left, right, depth)

    def __split_data(self, data, index, threshold):
        left = np.array([row for row in data if row[index] <= threshold])
        right = np.array([row for row in data if row[index] > threshold])

        return left, right
    
    def __entropy(self, y):
        y = y.astype(np.int64)
        counts = np.bincount(y)
        probabilities = counts / len(y)
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return entropy
    
    def __majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    def __best_split(self, data, num_samples, num_features):
        best_split = {}
        max_info_gain = -float("inf")

        for index in range(num_features):
            values = data[:, index]
            thresholds = np.unique(values)

            for threshold in thresholds:
                data_left, data_right = self.__split_data(data, index, threshold)

                if len(data_left) > 0 and len(data_right) > 0:
                    y = data[:, -1]
                    left_y = data_left[:, -1]
                    right_y = data_right[:, -1]

                    current_info_gain = self.__info_gain(y, left_y, right_y)

                    if current_info_gain > max_info_gain:
                        best_split['index'] = index
                        best_split['threshold'] = threshold
                        best_split['data left'] = data_left
                        best_split['data right'] = data_right
                        best_split['info gain'] = current_info_gain

                        max_info_gain = current_info_gain

        return best_split

    def __info_gain(self, parent, left, right):
        left_weight = len(left) / len(parent)
        right_weight = len(right) / len(parent)
        gain = self.__entropy(parent) - (left_weight * self.__entropy(left) + right_weight * self.__entropy(right))

        return gain
    
    def predict(self, X):
        prediction = np.array([self.__predict_recurs(x, self.root) for x in X])

        return prediction
    
    def __predict_recurs(self, x, node:Node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.__predict_recurs(x, node.left_node)
        else:
            return self.__predict_recurs(x, node.right_node)
        
        