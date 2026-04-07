from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork

# Load dataset
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build network
nn = NeuralNetwork()
nn.add_layer(128, 64, activation='relu')
nn.add_layer(64, 128, activation='relu')
nn.add_layer(10, 64, activation='sigmoid')

# Train
epochs = 10
lr = 0.05
for epoch in range(epochs):
    print(epoch)
    for x_sample, y_sample in zip(X_train, y_train):
        nn.train(x_sample, y_sample, L=lr)

print("Training completed")

# Test accuracy
correct = 0
for x_sample, y_sample in zip(X_test, y_test):
    pred = nn.predict_class(x_sample)
    true_label = y_sample.argmax()
    if pred == true_label:
        correct += 1

accuracy = correct / len(X_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")