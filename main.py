class PCAModel:
    def __init__(self, n_components):
        self.n_components = n_components
        # Initialize all these as None
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        """
        Fit PCA on the dataset X.
        """
        # 0. Convert to array
        X = np.array(X, dtype=float)

        # 1. Center the data
        # compute mean
        self.mean = np.mean(X, axis=0)
        # center around mean
        X_centered = X - self.mean

        # 2. Covariance matrix
        # compute variance - each feature in columns - rowvar is False
        # rowvar False means features are columns, True means features are rows
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Eigen decomposition
        # compute the eigen values and vectors of covariance martix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvectors by descending eigenvalues
        # sort in descending order of eigen values
        sorted_idx = np.argsort(eigenvalues)[::-1]

        # take top n components
        self.explained_variance = eigenvalues[sorted_idx][:self.n_components]

        # get the components
        self.components = eigenvectors[:, sorted_idx][:, :self.n_components]

    def predict(self, X):
        """
        Project the data X onto the principal components.
        """
        if self.mean is None or self.components is None:
            # eigen values or vectors do not exist if model is not fitted yet
            raise ValueError("The PCA model has not been fitted yet.")

        # center the data around zero
        X_centered = X - self.mean

        # return the dot product with eigen vectors, to retrieve components
        return np.dot(X_centered, self.components)

    def reconstruct(self, X):
        """
        Reconstruct the original data from the reduced representation.
        """
        # predict projections
        Z = self.predict(X)  # Projected data

        # reconstruct X based on projects onto components, and add mean value
        return np.dot(Z, self.components.T) + self.mean

    def detect_anomalies(self, X, threshold=None, return_errors=False):
        """
        Detect anomalies based on reconstruction error.

        Parameters:
        - X: Input data
        - threshold: Optional. If not provided, uses 95th percentile of reconstruction errors
        - return_errors: If True, also returns reconstruction errors

        Returns:
        - is_anomaly: Boolean mask of anomalies
        - errors (optional): Reconstruction errors for each point
        """
        # reconstrut X
        X_reconstructed = self.reconstruct(X)

        # compute reconstruction error
        errors = np.mean((X - X_reconstructed) ** 2, axis=1)

        # if no threshold, take 95% value
        if threshold is None:
            threshold = np.percentile(errors, 95)

        # flag observations, whose reconstruction error is higher than this threshold
        flag = errors > threshold

        is_anomaly = flag * 1

        return is_anomaly, errors

import numpy as np
from sklearn.metrics import f1_score

from algorithms import (
    RFFTransformer, SoftmaxRegression,HistGBDT, RandomForest, 
)

def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):
    
    df_train = pd.read_csv(trainfile)
    df_valid = pd.read_csv(validationfile)

    return (df_train,df_valid)

X_train = df_train.drop(columns=["label"]).values.astype(np.float32)
y_train = df_train["label"].values.astype(np.int64)

X_valid = df_valid.drop(columns=["label"]).values.astype(np.float32)
y_valid = df_valid["label"].values.astype(np.int64)

# NORMALIZE
X_train = X_train / 255.0
X_valid = X_valid / 255.0


# STANDARDIZATION 
def standardize(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sigma, mu, sigma

def apply_standardize(X, mu, sigma):
    return (X - mu) / sigma

X_train_std, mu, sigma = standardize(X_train)
X_valid_std = apply_standardize(X_valid, mu, sigma)

# PCA WITH 64 COMPONENTS

pca_dim = 64
pca = PCAModel(n_components=pca_dim)
pca.fit(X_train_std)
X_train_pca = pca.predict(X_train_std)
X_valid_pca = pca.predict(X_valid_std)

# TRAIN MODELS 

#  RFF + Softmax Regression
rff = RFFTransformer(D=2048, gamma=0.0078125, seed=42)
X_train_rff = rff.fit_transform(X_train_pca)
X_valid_rff = rff.transform(X_valid_pca)

clf_rff = SoftmaxRegression(lr=0.5, epochs=40, batch_size=256, l2=1e-4, seed=42)
clf_rff.fit(X_train_rff, y_train)

proba_rff = clf_rff.softmax(X_valid_rff @ clf_rff.W + clf_rff.b)

# Histogram GBDT 
gbdt = HistGBDT(
    n_classes=len(np.unique(y_train)),
    n_estimators=120,
    lr=0.1,
    max_depth=5,
    num_bins=16,
    subsample=0.6,
    colsample=0.5,
    min_child_weight=2.0,
    early_stopping_rounds=5,
    verbose=True,
    seed=42
)

gbdt.fit(X_train_pca, y_train, X_valid_pca, y_valid)
proba_gbdt = gbdt.predict_proba(X_valid_pca)

# RandomForest
rf = RandomForest(
    n_trees=50,
    max_depth=12,
    min_samples=20,
    mtry=None,
    num_bins=16,
    seed=42
)

rf.fit(X_train_pca, y_train)
pred_rf = rf.predict(X_valid_pca)

num_classes = len(np.unique(y_train))
proba_rf = np.zeros((len(pred_rf), num_classes), dtype=np.float32)
proba_rf[np.arange(len(pred_rf)), pred_rf] = 1.0

#  WEIGHTED ENSEMBLE
#    weights: GBDT = 3, RFF = 1, RF = 1


w_gbdt = 3
w_rff  = 1
w_rf   = 1

ensemble_proba = (
      w_gbdt * proba_gbdt
    + w_rff  * proba_rff
    + w_rf   * proba_rf
)

# normalize 
ensemble_proba /= (w_gbdt + w_rff + w_rf)

pred_ens = ensemble_proba.argmax(axis=1)

# FINAL WEIGHTED F1 SCORE

f1 = f1_score(y_valid, pred_ens, average="weighted")
print("\nFinal Ensemble Weighted F1 Score =", f1)
