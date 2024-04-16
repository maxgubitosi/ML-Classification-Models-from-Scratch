import numpy as np
from collections import Counter
import matplotlib.pyplot as plt



# general functions

def get_min_max(X):
    min_max = {}
    for column in X.columns:
        min_max[column] = (X[column].min(), X[column].max())
    return min_max

def min_max_normalize(X, min_max):
    for column in X.columns:
        X[column] = (X[column] - min_max[column][0]) / (min_max[column][1] - min_max[column][0])
    return X


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def k_folds_cross_validation(X, y, model, k=5, seed=42):
    n = len(y)
    fold_size = n // k
    np.random.seed(seed)
    indexes = np.random.permutation(n)

    accuracy_ = 0
    precision_ = 0
    recall_ = 0
    f1_ = 0
    
    for i in range(k):
        test_indexes = indexes[i*fold_size:(i+1)*fold_size]
        train_indexes = np.concatenate([indexes[:i*fold_size], indexes[(i+1)*fold_size:]])
        
        X_train = X.iloc[train_indexes]
        y_train = y[train_indexes]
        X_test = X.iloc[test_indexes]
        y_test = y[test_indexes]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy_ += accuracy(y_test, y_pred)
        precision_ += precision(y_test, y_pred)
        recall_ += recall(y_test, y_pred)
        f1_ += f1(y_test, y_pred)
        
    return accuracy_/k, precision_/k, recall_/k, f1_/k



# performance metrics

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN
    
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def precision(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    return TP / (TP + FP)

def recall(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP / (TP + FN)

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r)

def roc(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR

def roc_curve(y_true, y_pred, plot=False, model_name='ROC curve'):
    thresholds = np.linspace(0, 1, num=100)
    tpr = []
    fpr = []
    for threshold in thresholds:
        tp = np.sum((y_pred >= threshold) & (y_true == 1))
        fn = np.sum((y_pred < threshold) & (y_true == 1))
        tn = np.sum((y_pred < threshold) & (y_true == 0))
        fp = np.sum((y_pred >= threshold) & (y_true == 0))
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
    tpr.append(0)
    fpr.append(0)
    if plot:        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random guesses')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc='lower right')
        plt.show()
    return fpr, tpr



# models

class LDA:

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.linear_discriminants = None
        self.transformed_X = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y) 

        # Within class scatter matrix:
        # SW = sum((X_c - mean_X_c)^2 )

        # Between class scatter:
        # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            SW += (X_c - mean_c).T.dot((X_c - mean_c))

            # (4, 1) * (1, 4) = (4,4) -> reshape
            n_c = X_c.shape[0]
            mean_diff = ((mean_c - mean_overall).values)**2
            mean_diff = mean_diff.reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        # Determine SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvalues high to low
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        self.transformed_X = np.dot(X, self.linear_discriminants.T)
        return self.transformed_X
    

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X.values]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train.values]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
    

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, lmbda=0.01):      # ver que hacer con lr y n_iters
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.lmbda = lmbda  # regularization parameter

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # compute gradients with regularization
            dw = (1 / n_samples) * (np.dot(X.T, (y_predicted - y)) + 2 * self.lmbda * self.weights)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

