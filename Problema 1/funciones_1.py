import numpy as np

# class LDA:
#     def __init__(self, n_components=None):
#         self.n_components = n_components
#         self.scalings = None
        
#     def fit(self, X, y):
#         # Separar las características por clases
#         X0 = X[y == 0]
#         X1 = X[y == 1]
        
#         # Calcular las medias de cada clase
#         mean0 = np.mean(X0, axis=0)
#         mean1 = np.mean(X1, axis=0)
        
#         # Calcular la matriz de dispersión dentro de las clases
#         Sw = np.cov(X0.T) + np.cov(X1.T)
        
#         # Calcular la diferencia entre las medias de las clases
#         mean_diff = (mean1 - mean0).values
        
#         # Calcular los autovectores y autovalores de (Sw^-1)(mean_diff)
#         eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw) @ mean_diff.reshape(-1, 1))
        
#         # Ordenar los autovalores de mayor a menor
#         sorted_indices = np.argsort(eigenvalues.real)[::-1]
        
#         # Tomar los autovectores correspondientes a los n_components autovalores más grandes
#         if self.n_components is not None:
#             self.scalings = eigenvectors[:, sorted_indices[:self.n_components]]
#         else:
#             self.scalings = eigenvectors[:, sorted_indices]

#     def transform(self, X):
#         return X @ self.scalings

class LDA:

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.linear_discriminants = None
        self.transformed_X = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y) # en nuestro caso son solo 2

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
            mean_diff = (mean_c - mean_overall).values
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
    

    # performance

    def confusion_matrix(self, y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        return np.array([[TP, FP], [FN, TN]])
    
