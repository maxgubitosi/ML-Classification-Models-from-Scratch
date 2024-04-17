import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

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

def one_hot_encoder(df, columns):
    for column in columns:
        dummies = pd.get_dummies(df[column], prefix=column, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        df.drop(column, axis=1, inplace=True)
    return df

def oversample(dataset):
    max_size = dataset['target'].value_counts().max()
    lst = [dataset]
    for class_index, group in dataset.groupby('target'):
        lst.append(group.sample(max_size-len(group), replace=True))
    return pd.concat(lst)

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    
def cross_validation(X, y, max_depths, forest_sizes, k=10, seed=42):
    """
    Perform k-fold cross validation for random forest
    Inputs:
        - X (np.array): input data
        - y (np.array): target data
        - max_depths (list): list of maximum depths for decision trees
        - forest_sizes (list): list of sizes for the random forest
        - k (int): number of folds
        - seed (int): seed for the random number generator
    Outputs:
        - acc_train (np.array): training accuracy for each fold and parameter combination
        - acc_test (np.array): test accuracy for each fold and parameter combination
    """
    np.random.seed(seed)
    n = X.shape[0]
    acc_train = np.zeros((len(max_depths) * len(forest_sizes), k))
    acc_test = np.zeros((len(max_depths) * len(forest_sizes), k))

    for i, depth in enumerate(max_depths):
        for j, forest_size in enumerate(forest_sizes):
            for fold in range(k):
                idx = np.arange(n)
                np.random.shuffle(idx)
                X_shuffled = X.iloc[idx]  
                y_shuffled = y[idx]

                X_train = np.concatenate([X_shuffled[:fold*(n//k)], X_shuffled[(fold+1)*(n//k):]], axis=0)
                y_train = np.concatenate([y_shuffled[:fold*(n//k)], y_shuffled[(fold+1)*(n//k):]], axis=0)
                X_test = X_shuffled[fold*(n//k):(fold+1)*(n//k)]
                y_test = y_shuffled[fold*(n//k):(fold+1)*(n//k)]
                
                rf_model = RandomForest(n_trees=forest_size, max_depth=depth)
                rf_model.fit(X_train, y_train)
                y_train_pred = rf_model.predict(X_train)
                y_test_pred = rf_model.predict(X_test.values)
                
                acc_train[i*len(forest_sizes) + j, fold] = accuracy(y_train, y_train_pred)
                acc_test[i*len(forest_sizes) + j, fold] = accuracy(y_test, y_test_pred)

    return acc_train, acc_test


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
    


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    

class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, n_feats=None, seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_feats = n_feats
        self.trees = []
        np.random.seed(seed)

    def fit(self, X, y):
        self.trees = []
        for _ in tqdm(range(self.n_trees)):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_feats=self.n_feats,
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [self._most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
    
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    