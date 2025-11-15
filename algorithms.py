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

#-----RANDOM FOREST ---

def gini_impurity(y):
    """
    Computes Gini impurity:
        Gini = 1 - Σ (p_k^2)
    Measures how 'mixed' a label set is.
    Pure nodes (all same class) have Gini = 0.
    """
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    return 1.0 - np.sum(p**2)



class TreeNode:
    """
    Node for a decision tree.
    Uses __slots__ to reduce memory footprint.
    """
    __slots__ = ("is_leaf", "pred", "feat", "thr", "left", "right")

    def __init__(self):
        self.is_leaf = True      # Whether node is terminal
        self.pred = 0            # Class prediction if leaf
        self.feat = None         # Feature index used for split
        self.thr = None          # Threshold value
        self.left = None         # Left child
        self.right = None        # Right child



class DecisionTree:
    """
    Basic CART-like decision tree classifier using:
      - Gini impurity
      - Random feature subset (mtry)
      - Quantile-based candidate thresholds
    Supports use inside a RandomForest.
    """

    def __init__(self, max_depth=10, min_samples=20, mtry=None, num_bins=16, seed=42):
        self.max_depth = max_depth
        self.min_samples = min_samples      # Minimum samples to attempt a split
        self.mtry = mtry                    # Number of features to consider at each split
        self.num_bins = num_bins            # Number of quantile thresholds
        self.seed = seed
        self.root = None

    def _build(self, X, y, depth):
        """
        Recursively builds the tree from data.
        """
        node = TreeNode()

        # --- Stopping conditions ------------------------------------------------
        # Stop if:
        #   - max depth reached
        #   - too few samples
        #   - all samples are same class
        if depth >= self.max_depth or y.size <= self.min_samples or np.unique(y).size == 1:
            node.is_leaf = True
            lab, cnts = np.unique(y, return_counts=True)
            node.pred = lab[np.argmax(cnts)]   # Majority class
            return node

        # ------------------------------------------------------------------------
        n, d = X.shape
        rng = np.random.default_rng(self.seed + depth)

        # Feature subsampling (mtry)
        if self.mtry is None:
            features = np.arange(d)
        else:
            features = rng.choice(d, size=self.mtry, replace=False)

        base = gini_impurity(y)
        best_feat, best_thr, best_gain = None, None, -1.0

        # --- Try each candidate feature ----------------------------------------
        for f in features:
            col = X[:, f]

            # Candidate thresholds = quantiles between 5% and 95%
            qs = np.unique(np.quantile(col, np.linspace(0.05, 0.95, self.num_bins)))

            # Try each possible threshold
            for thr in qs:
                left_mask = col <= thr
                right_mask = ~left_mask

                # Skip invalid splits
                if not left_mask.any() or not right_mask.any():
                    continue

                yl = y[left_mask]
                yr = y[right_mask]

                # Weighted Gini after split
                g = base - (yl.size/n)*gini_impurity(yl) - (yr.size/n)*gini_impurity(yr)

                # Keep best split
                if g > best_gain:
                    best_gain = g
                    best_feat, best_thr = f, thr

        # If no valid split found → make leaf
        if best_feat is None:
            node.is_leaf = True
            lab, cnts = np.unique(y, return_counts=True)
            node.pred = lab[np.argmax(cnts)]
            return node

        # --- Commit the split ---------------------------------------------------
        node.is_leaf = False
        node.feat = best_feat
        node.thr = best_thr

        left_mask = X[:, best_feat] <= best_thr

        # Recursively build children
        node.left = self._build(X[left_mask], y[left_mask], depth+1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth+1)

        return node

    def fit(self, X, y):
        """
        Builds a full decision tree starting from the root node.
        """
        if self.mtry is None:
            self.mtry = int(np.sqrt(X.shape[1]))  # Standard RF heuristic
        self.root = self._build(X, y, 0)

    def _predict_row(self, x, node):
        """
        Drops a single input row down the tree until reaching a leaf.
        """
        while not node.is_leaf:
            if x[node.feat] <= node.thr:
                node = node.left
            else:
                node = node.right
        return node.pred

    def predict(self, X):
        """
        Predicts labels for all samples using tree traversal.
        """
        n = X.shape[0]
        out = np.zeros(n, dtype=np.int64)
        for i in range(n):
            out[i] = self._predict_row(X[i], self.root)
        return out



class RandomForest:
    """
    Random Forest classifier:
      - Trains many decision trees on bootstrap samples
      - Each tree receives a random subset of features (mtry)
      - Predictions are obtained by majority vote
    """

    def __init__(self, n_trees=25, max_depth=10, min_samples=20, mtry=None, num_bins=16, seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.mtry = mtry
        self.num_bins = num_bins
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        """
        Trains n_trees trees, each on a bootstrap sample of (X, y).
        """
        n = X.shape[0]
        rng = np.random.default_rng(self.seed)

        self.trees = []
        for t in range(self.n_trees):
            # Bootstrap sampling (sample with replacement)
            idx = rng.integers(0, n, size=n)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                mtry=self.mtry,
                num_bins=self.num_bins,
                seed=self.seed + t
            )

            # Train on bootstrap sample
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict using majority vote across all trees.
        """
        n = X.shape[0]

        # Assuming 10 classes—can be generalized if needed
        votes = np.zeros((n, 10), dtype=np.int32)

        # Each tree votes for one class
        for tree in self.trees:
            p = tree.predict(X)
            votes[np.arange(n), p] += 1

        # Pick class with most votes
        return votes.argmax(axis=1)
