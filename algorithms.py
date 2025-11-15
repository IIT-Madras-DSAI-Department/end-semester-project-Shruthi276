import numpy as np

# ------------------- HIST BASED GBDT -------------------

def one_hot(y, K):
    """
    Convert class labels y into one-hot encoded matrix of shape (N, K).
    """
    Y = np.zeros((y.shape[0], K), dtype=np.float32)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y

def softmax(F):
    """
    Row-wise softmax with numerical stabilization.
    """
    F = F - F.max(axis=1, keepdims=True)
    e = np.exp(F)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def multiclass_logloss(y_true, probs):
    """
    Standard multiclass log-loss.
    """
    n = y_true.shape[0]
    p = np.clip(probs[np.arange(n), y_true], 1e-15, 1.0)
    return -float(np.mean(np.log(p)))


# ------------------- Binning -------------------
def fit_bins(X, num_bins=32):
    """
    Learn quantile-based bin boundaries for each feature.
    Each feature gets (num_bins - 1) cut points.
    """
    n, d = X.shape
    cuts = []
    for j in range(d):
        col = X[:, j]
        qs = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]  # internal quantiles
        pts = np.unique(np.quantile(col, qs))           # unique boundaries

        # If feature is constant, fall back to single mean cut
        if pts.size == 0:
            pts = np.array([np.mean(col)])
        cuts.append(pts.astype(np.float32))
    return cuts

def apply_bins(X, cuts):
    """
    Convert continuous features into integer bin indices using learned cut points.
    """
    n, d = X.shape
    binned = np.empty((n, d), dtype=np.int32)
    for j in range(d):
        pts = cuts[j]
        # searchsorted returns bin index (0...num_bins-1)
        binned[:, j] = np.searchsorted(pts, X[:, j], side='right')
    return binned


# ------------------- Tree Node -------------------
class HistTreeNode:
    """
    A single node in the decision tree.
    Stores split info OR leaf prediction.
    """
    __slots__ = ("is_leaf","pred","feature","bin_index","left","right","depth")

    def __init__(self):
        self.is_leaf = True       # becomes False if a split is made
        self.pred = 0.0           # leaf score (weight)
        self.feature = -1         # feature used in split
        self.bin_index = -1       # threshold bin index
        self.left = None
        self.right = None
        self.depth = 0            # tree depth at this node


# ------------------- Histogram-based GBDT -------------------
class HistGBDT:
    """
    A custom implementation of Histogram-based Gradient Boosted Decision Trees
    for multiclass classification using softmax + additive trees.
    """

    def __init__(self,
                 n_classes=10,
                 n_estimators=150, lr=0.1, max_depth=6,
                 min_child_weight=1.0, lamb=1.0, gamma=0.0,
                 subsample=0.8, colsample=0.8,
                 num_bins=32, seed=0,
                 early_stopping_rounds=10,
                 verbose=True):

        # Model hyperparameters
        self.K = n_classes
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.lamb = lamb              # L2 regularization
        self.gamma = gamma            # split penalty
        self.subsample = subsample    # row sampling
        self.colsample = colsample    # feature sampling
        self.num_bins = num_bins
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        # Will be filled during fit()
        self.trees = []               # list of K trees per boosting round
        self.cuts = None              # per-feature cut points
        self.feature_bins = None      # number of bins per feature

    # ============================================================
    # Histogram Construction
    # ============================================================
    def _build_feature_histograms(self, binned_X, grad, hess, rows, feat_idxs):
        """
        Build histogram for G & H for selected features.
        Gbins[b] = sum of gradients for bin b
        Hbins[b] = sum of Hessians for bin b
        """
        hist = {}
        for f in feat_idxs:
            bins_count = self.feature_bins[f]

            G = np.zeros(bins_count + 1, dtype=np.float64)
            H = np.zeros(bins_count + 1, dtype=np.float64)

            col_bins = binned_X[rows, f]

            # Add gradients/hessians into bins
            np.add.at(G, col_bins, grad[rows])
            np.add.at(H, col_bins, hess[rows])

            hist[f] = (G, H)
        return hist

    def _find_best_split_feature(self, hist, totalG, totalH):
        """
        Given histograms, try all possible splits for all features.
        Returns feature, bin, gain, and branch statistics.
        """
        best_gain = -1e18
        best_feat = None
        best_bin = None
        best_parts = None

        for f, (Gbins, Hbins) in hist.items():
            # Prefix sums → aggregates left side stats
            prefixG = np.cumsum(Gbins)
            prefixH = np.cumsum(Hbins)
            nb = len(Gbins)

            for b in range(nb - 1):
                GL = prefixG[b]; HL = prefixH[b]       # Left child stats
                GR = totalG - GL; HR = totalH - HL     # Right child stats

                # Enforce minimum child weight
                if HL < self.min_child_weight or HR < self.min_child_weight:
                    continue

                # Standard GBDT gain formula for 2nd-order boosting
                gain = 0.5 * (
                    (GL*GL)/(HL + self.lamb) +
                    (GR*GR)/(HR + self.lamb) -
                    (totalG*totalG)/(totalH + self.lamb)
                ) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_bin = b
                    best_parts = (GL, HL, GR, HR)

        return best_feat, best_bin, best_gain, best_parts

    def _compute_leaf_weight(self, G, H):
        """
        Leaf weight formula for second-order boosting:
            w = -G / (H + lambda)
        """
        return -G / (H + self.lamb)

    # ============================================================
    # Tree Construction
    # ============================================================
    def _build_tree(self, binned_X, grad, hess, rows, feat_pool):
        """
        Build a single CART tree using histograms and gradient statistics.
        """
        root = HistTreeNode()
        nodes = [(root, rows)]  # (node, row_indices)

        for depth in range(self.max_depth):
            new_nodes = []

            for node, node_rows in nodes:
                totalG = grad[node_rows].sum()
                totalH = hess[node_rows].sum()

                # If empty or tiny node → make leaf
                if node_rows.size == 0 or totalH < self.min_child_weight:
                    node.is_leaf = True
                    node.pred = float(self._compute_leaf_weight(totalG, totalH))
                    continue

                # Build histograms only for feature subset
                hist = self._build_feature_histograms(binned_X, grad, hess,
                                                      node_rows, feat_pool)

                # Find best split from histograms
                feat, bin_idx, gain, parts = self._find_best_split_feature(
                    hist, totalG, totalH
                )

                # If no good split, create leaf
                if feat is None or gain <= 0:
                    node.is_leaf = True
                    node.pred = float(self._compute_leaf_weight(totalG, totalH))
                    continue

                # Perform split
                node.is_leaf = False
                node.feature = feat
                node.bin_index = bin_idx

                # Partition rows into children
                left_mask = binned_X[node_rows, feat] <= bin_idx
                left_rows = node_rows[left_mask]
                right_rows = node_rows[~left_mask]

                # Create child nodes
                node.left = HistTreeNode(); node.left.depth = node.depth + 1
                node.right = HistTreeNode(); node.right.depth = node.depth + 1

                new_nodes.append((node.left, left_rows))
                new_nodes.append((node.right, right_rows))

            # No new nodes → stop
            if not new_nodes:
                break

            nodes = new_nodes

        return root

    def _predict_tree_array(self, tree, binned_X):
        """
        Run every sample through the tree to obtain leaf prediction.
        """
        n = binned_X.shape[0]
        out = np.zeros((n,), dtype=np.float32)

        for i in range(n):
            node = tree
            # Traverse tree until a leaf
            while not node.is_leaf:
                if binned_X[i, node.feature] <= node.bin_index:
                    node = node.left
                else:
                    node = node.right
            out[i] = node.pred
        return out

    # ============================================================
    # FIT
    # ============================================================
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train GBDT using softmax gradients + histogram trees.
        """
        n, d = X.shape
        rng = np.random.default_rng(self.seed)

        # Bin training & validation features
        self.cuts = fit_bins(X, num_bins=self.num_bins)
        self.feature_bins = [len(c) + 1 for c in self.cuts]
        binned_X = apply_bins(X, self.cuts)
        binned_X_val = apply_bins(X_val, self.cuts) if X_val is not None else None

        # Raw model outputs (logits), updated after each boosting round
        F = np.zeros((n, self.K), dtype=np.float32)
        if X_val is not None:
            F_val = np.zeros((X_val.shape[0], self.K), dtype=np.float32)

        Y = one_hot(y, self.K)

        best_val = 1e9
        rounds_no_improve = 0

        # Number of features sampled each round (column subsampling)
        mf = max(1, int(self.colsample * d))

        # --------------------------------------------------------
        # Boosting loop
        # --------------------------------------------------------
        for it in range(self.n_estimators):

            # Row sampling
            rows = rng.choice(n, size=int(self.subsample * n), replace=False) \
                    if self.subsample < 1 else np.arange(n)

            # Softmax predictions from current model
            P = softmax(F)

            # 1st and 2nd derivatives of softmax cross entropy
            G_all = (P - Y).astype(np.float64)
            H_all = (P * (1 - P)).astype(np.float64)

            trees_round = []

            # Column subsampling
            feat_pool = rng.choice(d, size=mf, replace=False)

            # --- Train one tree per class ---
            for k in range(self.K):
                grad = G_all[:, k]
                hess = H_all[:, k]

                # Build tree for this class
                tree = self._build_tree(binned_X, grad, hess, rows, feat_pool)

                # Update model logits
                preds_train = self._predict_tree_array(tree, binned_X)
                F[:, k] += self.lr * preds_train

                # Validation logits
                if X_val is not None:
                    preds_val = self._predict_tree_array(tree, binned_X_val)
                    F_val[:, k] += self.lr * preds_val

                trees_round.append(tree)

            self.trees.append(trees_round)

            # Early stopping check
            if X_val is not None:
                Pval = softmax(F_val)
                val_loss = multiclass_logloss(y_val, Pval)

                if self.verbose:
                    print(f"Iter {it+1}: val_logloss={val_loss:.6f}")

                if val_loss + 1e-9 < best_val:
                    best_val = val_loss
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1

                if rounds_no_improve >= self.early_stopping_rounds:
                    print(f"Early stopping at iter {it+1}. Best={best_val:.6f}")
                    break

        return self

    # ============================================================
    # PREDICT
    # ============================================================
    def predict_proba(self, X):
        """
        Predict class probabilities by summing logits from all trained trees.
        """
        bX = apply_bins(X, self.cuts)
        n = bX.shape[0]

        F = np.zeros((n, self.K), dtype=np.float32)

        # Sum contributions of all boosting rounds
        for trees_round in self.trees:
            for k, tree in enumerate(trees_round):
                preds = self._predict_tree_array(tree, bX)
                F[:, k] += self.lr * preds

        return softmax(F)

    def predict(self, X):
        """
        Predict class labels by choosing argmax probability.
        """
        return self.predict_proba(X).argmax(axis=1)


# ----- RFFT + SOFTMAX 

class RFFTransformer:
    """
    Random Fourier Features (RFF) transformer.
    Approximates an RBF kernel by mapping data into a higher-dimensional
    cosine feature space, enabling linear models to mimic kernel methods.
    """

    def __init__(self, D=1024, gamma=None, seed=42):
        self.D = int(D)          # Number of random features
        self.gamma = gamma       # RBF kernel width parameter
        self.seed = seed
        self.W = None            # Random projection matrix
        self.b = None            # Random phase offsets
        self.norm = None         # Normalization factor

    def fit(self, X):
        """
        Learns the random projection matrix W and bias vector b.
        These remain fixed after fitting.
        """
        n, d = X.shape

        # If gamma not provided, use 1/d (standard heuristic for RBF kernels)
        if self.gamma is None:
            self.gamma = 1.0 / max(1.0, float(d))

        rng = np.random.default_rng(self.seed)

        # W is sampled from N(0, sqrt(2*gamma)) as per RFF theory (Rahimi & Recht)
        scale = np.sqrt(2.0 * self.gamma)
        self.W = rng.normal(0.0, scale, size=(d, self.D)).astype(np.float32)

        # Random phase offsets sampled uniformly from [0, 2π]
        self.b = rng.uniform(0.0, 2*np.pi, size=(self.D,)).astype(np.float32)

        # Normalization constant for cosine features
        self.norm = np.sqrt(2.0 / self.D)

        return self

    def transform(self, X):
        """
        Applies the random Fourier feature transformation:
            z = sqrt(2/D) * cos(XW + b)
        """
        # Linear projection
        Z = X.astype(np.float32) @ self.W

        # Add random phases
        Z += self.b[None, :]

        # Apply cosine nonlinearity
        np.cos(Z, out=Z)

        # Normalize
        Z *= self.norm

        return Z.astype(np.float32)

    def fit_transform(self, X):
        """
        Convenience method: fit() + transform() in one step.
        """
        self.fit(X)
        return self.transform(X)



class SoftmaxRegression:
    """
    Multiclass Softmax Regression (multinomial logistic regression)
    optimized using mini-batch gradient descent.

    Works very well when inputs come from RFFTransformer.
    """

    def __init__(self, lr=0.5, epochs=25, batch_size=256, l2=1e-4, seed=42):
        self.lr = lr
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.l2 = l2                # L2 regularization strength
        self.seed = seed
        self.W = None               # Weight matrix (d x K)
        self.b = None               # Bias vector (K,)

    def fit(self, X, y, num_classes=None):
        """
        Trains softmax regression using gradient descent.
        X: (n, d)
        y: (n,)
        """

        X = X.astype(np.float32)
        n, d = X.shape

        # Determine number of classes automatically
        if num_classes is None:
            num_classes = int(y.max()) + 1

        # Initialize weights (small random) and biases (0)
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 0.01, size=(d, num_classes)).astype(np.float32)
        self.b = np.zeros((num_classes,), dtype=np.float32)

        # Training loop
        for ep in range(self.epochs):
            idx = np.arange(n)
            rng.shuffle(idx)  # Shuffle data each epoch

            # Mini-batch training
            for i in range(0, n, self.batch_size):
                batch = idx[i:i+self.batch_size]
                Xb = X[batch]
                yb = y[batch]

                # Compute raw scores (logits)
                logits = Xb @ self.W + self.b

                # Numerical stability: subtract max
                logits = logits - logits.max(axis=1, keepdims=True)

                # Softmax
                exp = np.exp(logits)
                probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)

                # Create one-hot targets
                onehot = np.zeros_like(probs)
                onehot[np.arange(len(yb)), yb] = 1.0

                # Gradient of cross-entropy loss
                gradW = (Xb.T @ (probs - onehot)) / Xb.shape[0]
                gradb = (probs - onehot).mean(axis=0)

                # Add L2 regularization gradient
                gradW += self.l2 * self.W

                # Gradient descent update
                self.W -= self.lr * gradW
                self.b -= self.lr * gradb

    def softmax(self, logits):
        """
        Computes stable softmax for inference.
        """
        l = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(l)
        return e / (e.sum(axis=1, keepdims=True) + 1e-12)
    def predict_proba(self, X):
        X = X.astype(np.float32)
        logits = X @ self.W + self.b
        return self.softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

#------- KNN --------------------

import numpy as np
from collections import Counter, defaultdict

class KNearestNeighbors:

    def __init__(self, k=5, weights='uniform', batch_size=1024, eps=1e-8, seed=42):
        assert weights in ('uniform', 'distance')
        self.k = int(k)
        self.weights = weights
        self.batch_size = int(batch_size)
        self.eps = float(eps)
        self.seed = int(seed)
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        return self

    def _pairwise_distances(self, X_batch):
        a2 = np.sum(X_batch**2, axis=1).reshape(-1, 1)
        b2 = np.sum(self.X_train**2, axis=1).reshape(1, -1)
        ab = X_batch @ self.X_train.T
        return np.sqrt(np.maximum(a2 + b2 - 2.0 * ab, 0.0))

    def predict_proba(self, X, return_labels=None):
        if self.X_train is None:
            raise ValueError("Model not fitted")

        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        classes = self.classes_ if return_labels is None else np.asarray(return_labels)
        C = classes.shape[0]
        probs = np.zeros((n, C), dtype=np.float32)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        for start in range(0, n, self.batch_size):
            end = min(n, start + self.batch_size)
            Xb = X[start:end]
            dists = self._pairwise_distances(Xb)

            idx_part = np.argpartition(dists, kth=self.k-1, axis=1)[:, :self.k]
            rows = np.arange(idx_part.shape[0])[:, None]
            k_dists = dists[rows, idx_part]
            order = np.argsort(k_dists, axis=1)
            idx_sorted = idx_part[rows, order]

            for i in range(idx_sorted.shape[0]):
                neigh_idx = idx_sorted[i]
                neigh_labels = self.y_train[neigh_idx]

                if self.weights == 'uniform':
                    cnt = Counter(neigh_labels.tolist())
                    for label, ccount in cnt.items():
                        probs[start + i, class_to_idx[label]] = ccount / float(self.k)
                else:
                    neigh_d = dists[i, neigh_idx]
                    w = 1.0 / (neigh_d + self.eps)
                    accum = defaultdict(float)
                    for lab, wi in zip(neigh_labels.tolist(), w.tolist()):
                        accum[lab] += wi
                    total = sum(accum.values())
                    if total == 0.0:
                        for lab in accum:
                            probs[start + i, class_to_idx[lab]] = 1.0 / len(accum)
                    else:
                        for lab, val in accum.items():
                            probs[start + i, class_to_idx[lab]] = val / total

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y_true):
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)


#------PCA---------
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