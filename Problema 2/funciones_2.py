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

class Node():
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.feature_importance = self.data.shape[0] * self.information_gain
        self.left = None
        self.right = None



# class re-balance

def undersample(X, y):
    """
    Eliminates samples from the majority class randomly until the classes are balanced
    """
    # Convert DataFrame to NumPy array
    X_array = X.values

    # find the class with the least samples and the class with the most samples
    classes, counts = np.unique(y, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    min_class_size = np.min(counts)
    # find the indexes of the samples in the majority class
    majority_indexes = np.where(y == majority_class)[0]
    # randomly select samples from the majority class to eliminate
    indexes_to_keep = np.random.choice(majority_indexes, size=min_class_size, replace=False)
    # concatenate the samples from the minority class and the samples from the majority class that were kept
    X_balanced = np.concatenate([X_array[y == minority_class], X_array[indexes_to_keep]])
    y_balanced = np.concatenate([y[y == minority_class], y[indexes_to_keep]])
    return X_balanced, y_balanced
    
def oversample(X, y):
    """
    Creates new samples in the minority class randomly until the classes are balanced
    """
    X_array = X.values
    classes, counts = np.unique(y, return_counts=True)   
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    max_class_size = np.max(counts)
    minority_indexes = np.where(y == minority_class)[0]   # indexes of the samples in the minority class
    # randomly duplicate samples from the minority class until the classes are balanced
    indexes_to_duplicate = np.random.choice(minority_indexes, size=max_class_size, replace=True)
    X_balanced = np.concatenate([X_array[y == majority_class], X_array[indexes_to_duplicate]])
    y_balanced = np.concatenate([y[y == majority_class], y[indexes_to_duplicate]])
    return X_balanced, y_balanced

def smote(X, y, k=5, seed=None, oversampling_ratio=1.0):
    np.random.seed(seed)
    X_minority = X[y == 1]    
    n_samples = int(len(X_minority) * oversampling_ratio)   # number of samples to create    
    indices = np.random.choice(len(X_minority), size=n_samples, replace=True)
    synthetic_samples = []
    
    for index in indices:
        distances = np.linalg.norm(X_minority - X_minority[index], axis=1)   # calculate the distances between the current sample and all the other samples
        nearest_neighbors_indices = np.argsort(distances)
        nearest_neighbors_indices = nearest_neighbors_indices[1:k+1]
        chosen_neighbor_index = np.random.choice(nearest_neighbors_indices)
        synthetic_sample = X_minority[index] + np.random.rand() * (X_minority[chosen_neighbor_index] - X_minority[index])
        synthetic_samples.append(synthetic_sample)
    
    X_resampled = np.vstack((X, np.array(synthetic_samples)))
    y_resampled = np.concatenate((y, np.ones(len(synthetic_samples)))).astype(int)
    
    return X_resampled, y_resampled


# performance metrics

def confusion_matrix(y_true, y_pred, model_name=None, plot=True):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    if plot:
        values = np.array([[TP, FP], [FN, TN]])
        fig, ax = plt.subplots()
        ax.imshow(values, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{"TP" if i == 0 and j == 0 else "FP" if i == 0 and j == 1 else "FN" if i == 1 and j == 0 else "TN"}\n{values[i, j]}', ha='center', va='center', color='black')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('True Values')

        if model_name: plt.title(f'Confusion Matrix for {model_name}')
        else: plt.title('Confusion Matrix')
        plt.show()
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

def roc_curve(y_true, model, X_validation, plot=False, model_name=None, thresh_iters=100, thresh_range=(0, 1)):
    thresholds = np.linspace(thresh_range[0], thresh_range[1], num=thresh_iters)
    tpr = []
    fpr = []
    for threshold in thresholds:
        model.threshold = threshold
        y_pred = model.predict(X_validation)
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
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc='lower right')
        plt.show()
    return fpr, tpr

def auc_roc(fpr, tpr):
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    sorted_indices = np.argsort(fpr)
    sorted_fpr = fpr[sorted_indices]
    sorted_tpr = tpr[sorted_indices]
    auc = np.trapz(sorted_tpr, sorted_fpr)   # calculate area under the curve (numerical integration)
    return auc

def pr_curve(y_true, model, X_validation, plot=False, model_name=None, thresh_iters=100, thresh_range=(0, 1)):
    thresholds = np.linspace(thresh_range[0], thresh_range[1], num=thresh_iters)
    
    precision = []
    recall = []
    for threshold in thresholds:
        model.threshold = threshold
        y_pred = model.predict(X_validation)
        tp = np.sum((y_pred >= threshold) & (y_true == 1))
        fn = np.sum((y_pred < threshold) & (y_true == 1))
        tn = np.sum((y_pred < threshold) & (y_true == 0))
        fp = np.sum((y_pred >= threshold) & (y_true == 0))
        if tp + fp == 0:
            precision.append(1)
        else:
            precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))
    precision.append(1)
    recall.append(0)
    if plot:        
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='blue', lw=2, label='PR curve')
        plt.plot([0, 1], [1, 0], 'red', label='Random Guesses')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {model_name}')
        plt.legend(loc='lower left')
        plt.show()
    return recall, precision

def auc_pr(recall, precision):
    recall = np.array(recall)
    precision = np.array(precision)
    sorted_indices = np.argsort(recall)
    sorted_recall = recall[sorted_indices]
    sorted_precision = precision[sorted_indices]
    auc = np.trapz(sorted_precision, sorted_recall)   # calculate area under curve (numerical integration)
    return auc



# Nota: Esta clase corresponde a la implementación realizada para el TP2 de la materia
class MLP(object):

    def __init__(self, input_size, layers=[6, 30, 1], activations='default', seed=42, verbose=False, threshold=0.5):
        self.verbose = verbose
        self.seed = seed
        self.input_size = input_size
        self.layers = layers  
        self.num_layers = len(self.layers)
        self.threshold = threshold
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
        return -np.mean(y * np.log(a_out) + (1 - y) * np.log(1 - a_out))   # binary cross-entropy loss

    def forward_pass(self, X):
        """
        Forward pass through the network
        
        Inputs:
            - X (np.array): input data
        Outputs:
            - a (list): list of activations
            - z (list): list of weighted inputs
        """
        # print(f"X shape: {X.shape}")         
        # z = [np.array(X).reshape(-1, 1)]      # en esta parte hay un error: para fit hay que usar esta línea
        z = [X.T]                               # para predict hay que usar esta línea en lugar de la anterior
        # print(f"z shape: {z.shape}")
        a = []

        for l in range(1, self.num_layers):
            if self.verbose: print(f"Shapes: Weight {l-1}: {self.weights[l-1].shape}, Activation {l-1}: {z[l-1].shape}")

            # print(f"Weights shape: {self.weights[l-1].shape}")
            # print(f"Data shape: {z[l-1].shape}")
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
        a, _ = self.forward_pass(X)
        return (a[-1] > self.threshold).astype(int)



class DecisionTree():
    def __init__(self, 
                 max_depth=4, 
                 min_samples_leaf=1, 
                 min_information_gain=0.0,
                 max_features=None) -> None:
        """
        Constructor function for DecisionTree instance
        Inputs:
            max_depth (int): max depth of the tree
            min_samples_leaf (int): min number of samples required to be in a leaf 
                                    to make the splitting possible
            min_information_gain (float): min information gain required to make the 
                                          splitting possible                              
            max_features (str): number of features (n_features) to consider when looking for the best split
                                if sqrt, then max_features=sqrt(n_features),
                                if log, then max_features=log2(n_features),
                                if None, then max_features=n_features
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.max_features = max_features

    def entropy(self, class_probabilities: list) -> float:
        return sum([-p * np.log2(p) for p in class_probabilities if p>0])
    
    def class_probabilities(self, labels: list) -> list:
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def data_entropy(self, labels: list) -> float:
        return self.entropy(self.class_probabilities(labels))
    
    def partition_entropy(self, subsets: list) -> float:
        """
        Calculates the entropy of a partitioned dataset. 
        Inputs:
            - subsets (list): list of label lists 
            (Example: [[1,0,0], [1,1,1] represents two subsets 
            with labels [1,0,0] and [1,1,1] respectively.)

        Returns:
            - Entropy of the labels
        """
        # Total count of all labels across all subsets.
        total_count = sum([len(subset) for subset in subsets]) 
        # Calculates entropy of each subset and weights it by its proportion in the total dataset 
        return sum([self.data_entropy(subset) * (len(subset) / total_count) for subset in subsets])
    
    def split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        """
        Partitions the dataset into two groups based on a specified feature 
        and its corresponding threshold value.
        Inputs:
        - data (np.array): training dataset
        - feature_idx (int): feature used to split
        - feature_val (float): threshold value 
        """
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]

        return group1, group2
    
    def _select_features_to_use(self, data: np.array) -> list:
        """
        Randomly selects the subset of features to use while
        splitting with respect to hyperparameter max_features.
        Inputs:
            - data (np.array): numpy array with training data
        Returns:
            - features_idx_to_use(np.array): numpy array with feature indexes to be used during splitting
        """
        feature_idx = list(range(data.shape[1]-1))

        if self.max_features == 'sqrt':
            features_idx_to_use = np.random.choice(feature_idx, size=int(np.sqrt(len(feature_idx))))
        elif self.max_features == 'log':
            features_idx_to_use = np.random.choice(feature_idx, size=int(np.log2(len(feature_idx))))
        else:
            features_idx_to_use = np.random.choice(feature_idx, size=len(feature_idx))
        
        return features_idx_to_use
    
    def _find_best_split(self, data: np.array) -> tuple:
        """
        Finds the optimal feature and value to split the dataset on 
        at each node of the tree (with the lowest entropy).
        Inputs:
            - data (np.array): numpy array with training data
        Returns:
            - 2 splitted groups (g1_min, g2_min) and split information 
            (min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy)
        """
        min_part_entropy = 1e9
        # feature_idx =  list(range(data.shape[1]-1))
        feature_idx_to_use =  self._select_features_to_use(data)

        for idx in feature_idx_to_use: # For each feature
            feature_vals = np.percentile(data[:, idx], q=np.arange(25, 100, 25)) # Calc 25th, 50th, and 75th percentiles
            for feature_val in feature_vals: # For each percentile value we partition in 2 groups
                g1, g2, = self.split(data, idx, feature_val)
                part_entropy = self.partition_entropy([g1[:, -1], g2[:, -1]]) # Calculate entropy of that partition
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = idx
                    min_entropy_feature_val = feature_val
                    g1_min, g2_min = g1, g2

        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def _find_label_probs(self, data: np.array) -> np.array:
        """
        Computes the distribution of labels in the dataset.
        It returns the array label_probabilities, which contains 
        the probabilities of each label occurring in the dataset.

        Inputs:
            - data (np.array): numpy array with training data
        Returns:
            - label_probabilities (np.array): numpy array with the
            probabilities of each label in the dataset.
        """
        # Transform labels to ints (assume label in last column of data)
        labels_as_integers = data[:,-1].astype(int)
        # Calculate the total number of labels
        total_labels = len(labels_as_integers)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)
        # Populate the label_probabilities array based on the specific labels
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def _create_tree(self, data: np.array, current_depth: int) -> Node:
        """
        Recursive, depth first tree creation algorithm.
        Inputs:
            - data (np.array): numpy array with training data
            - current_depth (int): current depth of the recursive tree
        Returns:
            - node (Node): current node, which contains references to its left and right child nodes.
        """
        # Check if the max depth has been reached (stopping criteria)
        if current_depth > self.max_depth:
            return None
        # Find best split
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self._find_best_split(data)
        # Find label probs for the node
        label_probabilities = self._find_label_probs(data)
        # Calculate information gain
        node_entropy = self.entropy(label_probabilities)
        information_gain = node_entropy - split_entropy
        # Create node
        node = Node(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)
        # Check if the min_samples_leaf has been satisfied (stopping criteria)
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        # Check if the min_information_gain has been satisfied (stopping criteria)
        elif information_gain < self.min_information_gain:
            return node
        
        current_depth += 1
        node.left = self._create_tree(split_1_data, current_depth)
        node.right = self._create_tree(split_2_data, current_depth)
        
        return node
    
    def _predict_one_sample(self, X: np.array) -> np.array:
        """
        Returns prediction for 1 dim array.
        """
        node = self.tree
        # Finds the leaf which X belongs to
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def fit(self, X_train: np.array, Y_train: np.array) -> None:
        """
        Trains the model with given X and Y datasets.
        Inputs:
            - X_train (np.array): training features
            - Y_train (np.array): training labels
        """
        # Concat features and labels
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)
        # Create tree
        self.tree = self._create_tree(data=train_data, current_depth=0)

    def _predict_proba(self, X_set: np.array) -> np.array:
        """
        Returns the predicted probs for a given data set
        """
        pred_probs = np.apply_along_axis(self._predict_one_sample, 1, X_set)
        
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """
        Returns the predicted labels for a given data set
        """
        pred_probs = self._predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        
        return preds    
    


class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples_leaf=2, min_information_gain=0, max_features=None, threshold=0.5, C=None, seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.max_features = max_features
        self.threshold = threshold
        self.trees = []
        self.C = C
        np.random.seed(seed)

    def fit(self, X, y):
        self.trees = []
        for _ in tqdm(range(self.n_trees)):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_information_gain=self.min_information_gain,
                max_features=self.max_features,
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        y_pred = np.mean(tree_preds, axis=0)  # Calculate average prediction from all trees
        for i in range(len(y_pred)):
            if y_pred[i] >= self.threshold:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        # y_pred = np.round(y_pred).astype(int)
        return y_pred
    
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        if self.C == None: 
            idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        else:
            weights = np.where(y == 1, self.C, 1) 
            idxs = np.random.choice(n_samples, size=n_samples, replace=True, p=weights / np.sum(weights))
        return X[idxs], y[idxs]