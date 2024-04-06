import numpy as np

X = np.array([[5.1, 3.5, 1.4, 0.2],
              [4.9, 3.0, 1.4, 0.2],
              [4.7, 3.2, 1.3, 0.2],
              [4.6, 3.1, 1.5, 0.2],
              [5.0, 3.6, 1.4, 0.2],
              [7.0, 3.2, 4.7, 1.4],
              [6.4, 3.2, 4.5, 1.5],
              [6.9, 3.1, 4.9, 1.5],
              [5.5, 2.3, 4.0, 1.3],
              [6.5, 2.8, 4.6, 1.5],
              [6.3, 3.3, 6.0, 2.5],
              [5.8, 2.7, 5.1, 1.9],
              [7.1, 3.0, 5.9, 2.1],
              [6.3, 2.9, 5.6, 1.8],
              [6.5, 3.0, 5.8, 2.2]])

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

class KNNClassifier:
    def __init__(self, k=4):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            distances = [np.linalg.norm(sample - x_train) for x_train in self.X_train]
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
            prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(prediction)
        return predictions

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / len(X)) * np.dot(X.T, (y_predicted - y))
            db = (1 / len(X)) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

knn_model = KNNClassifier(k=4)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

def accuracy_score(y_true, y_pred):
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    return correct / len(y_true)

def precision_score(y_true, y_pred, average='weighted'):
    true_positives = 0
    predicted_positives = 0
    for true, pred in zip(y_true, y_pred):
        if pred == 1 and true == pred:
            true_positives += 1
        if pred == 1:
            predicted_positives += 1
    if predicted_positives == 0:
        return 0
    return true_positives / predicted_positives

def recall_score(y_true, y_pred, average='weighted'):
    true_positives = 0
    actual_positives = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred and true == 1:
            true_positives += 1
        if true == 1:
            actual_positives += 1
    if actual_positives == 0:
        return 0
    return true_positives / actual_positives

def f1_score(y_true, y_pred, average='weighted'):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions)
knn_recall = recall_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions)

print("Метод k ближайших соседей:")
print("Правильность:", knn_accuracy)
print("Точность:", knn_precision)
print("Полнота:", knn_recall)
print("F1-мера:", knn_f1)

log_reg_model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
log_reg_model.fit(X_train, y_train)
log_reg_predictions = log_reg_model.predict(X_test)

log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)
log_reg_precision = precision_score(y_test, log_reg_predictions)
log_reg_recall = recall_score(y_test, log_reg_predictions)
log_reg_f1 = f1_score(y_test, log_reg_predictions)

print("\nЛогистическая регрессия:")
print("Правильность:", log_reg_accuracy)
print("Точность:", log_reg_precision)
print("Полнота:", log_reg_recall)
print("F1-мера:", log_reg_f1)

