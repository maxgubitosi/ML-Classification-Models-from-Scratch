import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
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



# Esta clase corresponde a la implementación realizada para el TP2 de la materia
class MLP(object):

    def __init__(self, input_size, layers=[6, 30, 1], activations='default', seed=42, verbose=False):
        self.verbose = verbose
        self.seed = seed
        self.input_size = input_size
        self.layers = layers  
        self.num_layers = len(self.layers)
        if activations == 'default':
            self.activations = ['relu'] * (self.num_layers -1) + ['linear']
        else: self.activations = activations
        self.check_compatability()
        self.set_weights_and_biases()

    def check_compatability(self):
        assert len(self.activations) == self.num_layers, 'Debe haber una función de activación por capa'


    def set_weights_and_biases(self):
        """
        Initialize weights and biases randomly
        """
        np.random.seed(self.seed)
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
        if self.verbose:
            print(f"b.shape: {self.biases[0].shape}")
            print(f"W.shape: {self.weights[0].shape}")


    def activation_function(self, activation_str):
        """
        Returns the activation function given its name
        
        Inputs:
            - activation_str (str): name of the activation function
        Outputs:
            - lambda function: activation function
        """
        if activation_str == 'relu':
            return lambda z : np.maximum(z, 0)
        elif activation_str == 'linear':
            return lambda z : z
        elif activation_str == 'sigmoid':
            return lambda z : 1 / (1 + np.exp(-z))
        else:
            print("Invalid activation function")
        

    def deriv_activation_function(self, activation_str):
        """
        Returns the derivative of the activation function given its name

        Inputs:
            - activation_str (str): name of the activation function
        Outputs:
            - lambda function: derivative of the activation function
        """
        if activation_str == 'relu':
            return lambda z : (z > 0).astype(int)
        elif activation_str == 'linear':
            return lambda z : np.ones(z.shape)
        elif activation_str == 'sigmoid':
            return lambda z : z * (1 - z)
        else:
            print("Invalid activation function")


    def compute_loss(self, a_out, y):
        return np.mean((a_out - y) ** 2)


    def forward_pass(self, X):
        """
        Forward pass through the network
        
        Inputs:
            - X (np.array): input data
        Outputs:
            - a (list): list of activations
            - z (list): list of weighted inputs
        """
        z = [np.array(X).reshape(-1, 1)]
        a = []
        for l in range(1, self.num_layers):
            if self.verbose: print(f"Shapes: Weight {l-1}: {self.weights[l-1].shape}, Activation {l-1}: {z[l-1].shape}")
            a_l = np.dot(self.weights[l-1], z[l-1]) + self.biases[l-1]
            a.append(np.copy(a_l))

            # Check if the current layer has an associated activation function
            if l < len(self.activations):
                h = self.activation_function(self.activations[l-1])
                z_l = h(a_l)
            else:
                # If no activation function is specified, use linear activation
                z_l = a_l
                # # If no activation function is specified, use relu activation
                # z_l = np.maximum(a_l, 0)

            z.append(np.copy(z_l))

        if self.verbose:
            print(f"z.shape: {z[0].shape}", end=" ")
            print(f"a.shape: {a[0].shape}", end=" ")

        return a, z


    def backward_pass(self, a, z, y):
        """
        Backward pass through the network

        Inputs:
            - a (list): list of activations
            - z (list): list of weighted inputs
            - y (np.array): target data
        Outputs:
            - loss (float): loss of the network
            - nabla_w (list): list of gradients of the weights
            - nabla_b (list): list of gradients of the biases
        """ 
        d = [np.zeros(w.shape) for w in self.weights]
        h_deriv = self.deriv_activation_function(self.activations[-1])
        d[-1] = (a[-1] - y) * h_deriv(a[-1])

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_b[-1] = d[-1]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_w[-1] = np.dot(d[-1], z[-2].T)

        for l in reversed(range(1, len(d))):
            h_deriv = self.deriv_activation_function(self.activations[l-1])
            d[l-1] = np.dot(self.weights[l].T, d[l]) * h_deriv(a[l-1])
            nabla_b[l-1] = d[l-1]
            nabla_w[l-1] = np.dot(d[l-1], z[l-1].T)
        
        loss = self.compute_loss(a[-1], y)
        return loss, nabla_w, nabla_b


    def update_mini_batch(self, mini_batch, alpha):
        """
        Update weights and biases using mini-batch gradient descent
        
        Inputs:
            - mini_batch (list): list of mini-batches
            - alpha (float): learning rate
            Outputs:
            - total_loss (float): total loss of the mini-batch
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        total_loss = 0

        for x, y in mini_batch:
            a, z = self.forward_pass(x)
            loss, d_nabla_w, d_nabla_b = self.backward_pass(a, z, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, d_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, d_nabla_w)]
            total_loss += loss

        self.weights = [w - (alpha / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (alpha / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        return total_loss


    def update_single_example(self, x, y, alpha):
        """
        Update weights and biases using single example gradient descent
        
        Inputs:
            - x (np.array): input data
            - y (np.array): target data
            - alpha (float): learning rate
            Outputs:
            - loss (float): loss of the single example
        """
        a, z = self.forward_pass(x)
        loss, nabla_w, nabla_b = self.backward_pass(a, z, y)
        self.weights = [w - alpha * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - alpha * nb for b, nb in zip(self.biases, nabla_b)]
        return loss


    def evaluate(self, test_data):
        """
        Evaluate the network on test data

        Inputs:
            - test_data (list): list of test data
        Outputs:
            - sum_sq_error (float): mean squared error of the test data
        """
        sum_sq_error = 0
        for x, y in test_data:
            pred = self.forward_pass(x)[-1][-1].flatten()
            sum_sq_error += self.compute_loss(pred, y)
        return sum_sq_error / len(test_data)


    def fit(self, training_data, test_data, mini_batch_size, alpha=0.01, max_epochs=100, update_rule='mini_batch'):
        """
        Fit the model to the training data

        Inputs:
            - training_data (list): list of training data
            - test_data (list): list of test data
            - mini_batch_size (int): size of the mini-batches
            - alpha (float): learning rate
            - max_epochs (int): maximum number of epochs
            - update_rule (str): update rule for the network
        Outputs:
            - train_losses (list): list of training losses
            - test_losses (list): list of test losses
        """
        if update_rule == 'mini_batch':
            train_losses, test_losses = [], []
            n_train = len(training_data)

            for epoch in tqdm(range(max_epochs)):
                # random.shuffle(training_data)
                mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n_train, mini_batch_size)]

                for mini_batch in mini_batches:
                    train_loss = self.update_mini_batch(mini_batch, alpha)
                
                train_losses.append(train_loss)
    
                test_loss = self.evaluate(test_data)
                test_losses.append(test_loss)

                if self.verbose:
                    print(f"Epoch {epoch}: Train loss: {train_loss}, Test loss: {test_loss}")

            return train_losses, test_losses
        
        elif update_rule == 'single_example':
            train_losses, test_losses = [], []

            for epoch in tqdm(range(max_epochs)):
                # random.shuffle(training_data)

                for x, y in training_data:
                    train_loss = self.update_single_example(x, y, alpha)
                
                train_losses.append(train_loss)
                
                test_loss = self.evaluate(test_data)
                test_losses.append(test_loss)

                if self.verbose:
                    print(f"Epoch {epoch}: Train loss: {train_loss}, Test loss: {test_loss}")
                
            return train_losses, test_losses
    

    def predict(self, X):
        X = X.values
        predictions = []
        for x in X:
            a, z = self.forward_pass(x.reshape(-1, 1))
            pred = z[-1][-1].flatten()
            predictions.append(pred)

        return np.array(predictions)
    


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
        for feat_idx in tqdm(feat_idxs):
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