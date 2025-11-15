
import numpy as np
from sklearn.metrics import f1_score

from algorithms import *

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

# PCA
pca_dim = 128
pca = PCAModel(n_components=pca_dim)
pca.fit(X_train_std)
X_train_pca = pca.predict(X_train_std)
X_valid_pca = pca.predict(X_valid_std)

# ---------------------------------------------------------
# MODEL 1: RFF + Softmax Regression
# ---------------------------------------------------------
rff = RFFTransformer(D=4096, gamma=0.015625, seed=42)
X_train_rff = rff.fit_transform(X_train_pca)
X_valid_rff = rff.transform(X_valid_pca)

clf_rff = SoftmaxRegression(
    lr=0.2, epochs=50, batch_size=256,
    l2=1e-4, seed=42
)
clf_rff.fit(X_train_rff, y_train)

proba_rff = clf_rff.softmax(X_valid_rff @ clf_rff.W + clf_rff.b)

# ---------------------------------------------------------
# MODEL 2: Histogram GBDT
# ---------------------------------------------------------
gbdt = HistGBDT(
    n_classes=len(np.unique(y_train)),
    n_estimators=150,
    lr=0.1,
    max_depth=5,
    num_bins=16,
    subsample=0.6,
    colsample=0.5,
    min_child_weight=2.0,
    early_stopping_rounds=5,
    verbose=True,
    seed=42)

gbdt.fit(X_train_pca, y_train, X_valid_pca, y_valid)
proba_gbdt = gbdt.predict_proba(X_valid_pca)

# ---------------------------------------------------------
# MODEL 3: KNN
# ---------------------------------------------------------
knn = KNearestNeighbors(k=5, weights='distance')
knn.fit(X_train_pca, y_train)

proba_knn = knn.predict_proba(X_valid_pca)

# ---------------------------------------------------------
# FIXED WEIGHTED ENSEMBLE
# ---------------------------------------------------------
w_knn  = 0.50
w_gbdt = 0.40
w_rff  = 0.10

ensemble_proba = (
      w_knn  * proba_knn
    + w_gbdt * proba_gbdt
    + w_rff  * proba_rff
)

# Final predicted class
pred_ens = ensemble_proba.argmax(axis=1)

# Metrics
f1 = f1_score(y_valid, pred_ens, average="weighted")
print("\nFinal Weighted F1 Score:", f1)

