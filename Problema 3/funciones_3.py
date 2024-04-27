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

class DT_Node:
    def __init__(
        self, feature=None, threshold=None, l=None, r=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.l = l
        self.r = r
        self.value = value

    def is_leaf_DT_node(self):
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
    acc_0 = np.zeros((len(max_depths) * len(forest_sizes), k))
    acc_3 = np.zeros((len(max_depths) * len(forest_sizes), k))

    for i, depth in tqdm(enumerate(max_depths), desc='Max Depths'):
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

                acc_0[i*len(forest_sizes) + j, fold] = prec_class_0(y_test, y_test_pred)
                acc_3[i*len(forest_sizes) + j, fold] = prec_class_3(y_test, y_test_pred)

    return acc_train, acc_test, acc_0, acc_3

def smote(X, y, k=5, seed=None):
    np.random.seed(seed)
    synthetic_samples = []
    
    # find minority and majority classes
    majority_class = np.argmax(np.bincount(y))
    minority_classes = np.unique(y)[np.unique(y) != majority_class]
    majority_samples = np.sum(y == majority_class)      # samples in majority class
    
    for minority_class in minority_classes:
        X_minority = X[y == minority_class]
        n_samples = majority_samples - len(X_minority)   # number of samples to create for the given class
        indices = np.random.choice(len(X_minority), size=n_samples, replace=True)
        for index in indices:
            distances = np.linalg.norm(X_minority - X_minority[index], axis=1)      # calculate distances between the current sample and all other samples
            nearest_neighbors_indices = np.argsort(distances)
            nearest_neighbors_indices = nearest_neighbors_indices[1:k+1]
            chosen_neighbor_index = np.random.choice(nearest_neighbors_indices)
            synthetic_sample = X_minority[index] + np.random.rand() * (X_minority[chosen_neighbor_index] - X_minority[index])
            synthetic_samples.append(synthetic_sample)
    
    X_resampled = np.vstack((X, np.array(synthetic_samples)))
    y_resampled = np.concatenate((y, np.tile(minority_classes, majority_samples // len(minority_classes)))).astype(int)
    
    return X_resampled, y_resampled



# performance metrics

def confusion_matrix(y_true, y_pred, num_classes=4, model_name=None, plot=True):
    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)
    for true_label in range(num_classes):
        for pred_label in range(num_classes):
            confusion_mat[true_label, pred_label] = np.sum((y_true == true_label) & (y_pred == pred_label))

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(confusion_mat, cmap='Blues')
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, f'\n{confusion_mat[i, j]}', ha='center', va='center', color='black')        
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        if model_name: plt.title(f'Confusion Matrix for {model_name}')
        else: plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()    
    return confusion_mat

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def prec_class_0(y_true, y_pred):
    return np.sum(y_true[y_true == 0] == y_pred[y_true == 0]) / len(y_true[y_true == 0])

def prec_class_3(y_true, y_pred):
    return np.sum(y_true[y_true == 3] == y_pred[y_true == 3]) / len(y_true[y_true == 3])



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
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train.values] # distances between x and all samples in training set
        k_idx = np.argsort(distances)[: self.k]
        k_neinfo_gainhbor_labels = [self.y_train[i] for i in k_idx]
        most_common = Counter(k_neinfo_gainhbor_labels).most_common(1)  # most common class in the k nearest neighbors
        return most_common[0][0]
    


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_DT(X, y)

    def predict(self, X):
        return np.array([self._traverse_DT(x, self.root) for x in X])

    # Aux functions
    def _grow_DT(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stop (prune to avoid overfitting)
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._choose_label(y)
            return DT_Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        l_idxs, r_idxs = self._split(X[:, best_feat], best_thresh)
        l = self._grow_DT(X[l_idxs, :], y[l_idxs], depth + 1)
        r = self._grow_DT(X[r_idxs, :], y[r_idxs], depth + 1)
        return DT_Node(best_feat, best_thresh, l, r)

    def _best_criteria(self, X, y, feat_idxs):
        max_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > max_gain:
                    max_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        l_idxs, r_idxs = self._split(X_column, split_thresh)

        if len(l_idxs) == 0 or len(r_idxs) == 0:
            return 0

        # children loss
        n = len(y)
        n_l, n_r = len(l_idxs), len(r_idxs)
        e_l, e_r = entropy(y[l_idxs]), entropy(y[r_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        info_gain = parent_entropy - child_entropy
        return info_gain

    def _traverse_DT(self, x, DT_node):
        if DT_node.is_leaf_DT_node():
            return DT_node.value

        if x[DT_node.feature] <= DT_node.threshold:
            return self._traverse_DT(x, DT_node.l)
        return self._traverse_DT(x, DT_node.r)
    
    def _split(self, X_column, split_thresh):
        l_idxs = np.argwhere(X_column <= split_thresh).flatten()
        r_idxs = np.argwhere(X_column > split_thresh).flatten()
        return l_idxs, r_idxs

    def _choose_label(self, y):
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
        for _ in range(self.n_trees):
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
        y_pred = np.mean(tree_preds, axis=0)  # average prediction from all trees
        y_pred = np.round(y_pred).astype(int)
        return y_pred
    
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]