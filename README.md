# Machine Learning Algorithms from Scratch

This repository contains my **self-study into machine learning algorithms**, with implementations coded entirely from scratch in Python. It includes:

- Regression (Linear and Logistic)
- Naive Bayes Classification
- Neural Networks
- Decision Trees and Random Forests

---

## My Workflow

For each algorithm, my approach is:

1. **Research**: Study online tutorials, papers and documentation to understand the mathematics and structure behind each algorithm.
2. **Implementation**: Code the algorithm from scratch to gain a deep understanding of its mechanics.
3. **Testing**: Use **Jupyter notebooks** to experiment with datasets, check accuracy, and debug.

This workflow helps me **internalize the concepts** and ensures I can implement ML algorithms without relying on libraries like scikit-learn or TensorFlow.

---

## Repository Structure

- `decision_tree.py` – Decision Tree classifier using entropy and information gain  
- `random_forest.py` – Random Forest ensemble of decision trees  
- `regression.py` – Linear regression using gradient descent and Logistic regression for binary classification  
- `naive_bayes.py` – Text classifier using Naive Bayes and n-grams  
- `neural_network.py` – Feedforward neural network with ReLU/Sigmoid and softmax output  
- `notebooks/` – Jupyter notebooks for experiments, accuracy checks and testing  

---

## Features

- Full implementations **from scratch**, no ML libraries used  
- Gradient descent for linear and logistic regression  
- Entropy-based splitting for decision trees  
- Feedforward neural network with softmax classification  
- Naive Bayes text classifier with n-grams support  
- Modular, clean, and easy-to-read Python code  

---

## Example Usage

### Naive Bayes Text Classifier

```python
from naive_bayes import TextClassifier
import pandas as pd

data = pd.DataFrame({
    'Category': ['spam', 'ham', 'spam', 'ham'],
    'Text': ['buy cheap meds', 'hello friend', 'cheap meds online', 'meeting at noon']
})

clf = TextClassifier(data, alpha=1, ngram=2)
predictions = clf.predict(pd.Series(['buy meds online', 'hello']))
print(predictions)