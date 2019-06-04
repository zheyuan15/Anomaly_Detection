# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


class DecisionNode:
    def __init__(self, X, attribute, split):
        self.split = split
        self.attribute = attribute
        self.size = len(X)
        self.left = None
        self.right = None


class LeafNode:
    def __init__(self, X):
        self.size = len(X)


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.height_limit = np.ceil(np.log2(self.sample_size))
        self.trees = []

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        improved = improved
        if isinstance(X, pd.DataFrame):
            X = X.values

        for i in range(self.n_trees):
            sample_index = np.random.choice(len(X), self.sample_size, replace=False)
            subsample = X[sample_index, :]
            tree = IsolationTree(self.height_limit)
            tree.fit(subsample, 0, improved)
            self.trees.append(tree)
        return self

    def c_value(self, size):
        if size > 2:
            return 2 * (np.log(size - 1) + 0.5772156649) - 2 * (size - 1) / size
        elif size == 2:
            return 1
        else:
            return 0

    def find_node(self, x, tree, e):
        root = tree.root
        #print(isinstance(root, LeafNode))
        while isinstance(root, LeafNode) == 0:
            a = root.attribute
            if x[a] < root.split:
                root = root.left
            else:
                root = root.right
            e += 1

        return root.size, e

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """

        avg_length = np.zeros(len(X))
        for i in range(len(X)):
            x_i_path = []
            for tree in self.trees:
                leaf_node, e = self.find_node(X[i], tree, 0)
                path_len = self.c_value(leaf_node) + e
                x_i_path.append(path_len)
            avg_length[i] = np.mean(x_i_path)
        return avg_length

        """
        # vectorize
        vec_func = np.vectorize(self.find_node, cache=True, excluded=[0])
        leaf = []
        e = []
        for x_i in X:
            result = vec_func(x_i, self.trees, 0)
            leaf.append(result[0])
            e.append(result[1])
        vec_c = np.vectorize(self.c_value)
        x_path = vec_c(np.array(leaf)) + np.array(e)
        print(x_path.shape)
        avg_length = np.mean(x_path, axis=1)
        return avg_length
        """

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        avg_length = self.path_length(X)
        c = self.c_value(self.sample_size)
        score = 2 ** (- avg_length / c)
        return score

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return np.where(scores >= threshold, 1, 0)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        scores = self.anomaly_score(X)
        return self.predict_from_anomaly_scores(scores, threshold)


class IsolationTree:
    def __init__(self, height_limit):
        # self.X = X
        self.height_limit = height_limit
        self.n_nodes = 1
        self.root = None

    def fit(self, X: np.ndarray, e, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        improved = improved
        if len(X) <= 1 or e == self.height_limit or np.all(X == X[0]):  # e: current height
            return LeafNode(X)
        # print(X.shape)
        attribute = np.random.randint(X.shape[1])
        attr_list = X[:, attribute]
        # print(improved)
        if improved == False:
            split = np.random.uniform(min(attr_list), max(attr_list))
        else:
            attr_min = np.min(attr_list)
            attr_max = np.max(attr_list)

            #attr_mean = np.mean(attr_list)
            #attr_median = np.median(attr_list)
            #if attr_mean < attr_median:
            #    split_list = np.random.uniform(attr_min, attr_mean, 3)
            #else:
            #    split_list = np.random.uniform(attr_mean, attr_max, 3)

            split_list = np.random.uniform(attr_min, attr_max, 3)
            opt_split_size = len(attr_list)
            for i in split_list:
                left_size = len(attr_list[np.where(attr_list < i)])
                right_size = len(attr_list) - left_size
                split_size = np.min([left_size, right_size])
                if split_size < opt_split_size:
                    opt_split_size = split_size
                    split = i

        root = DecisionNode(X, attribute, split)
        e += 1
        # print(e)
        self.n_nodes += 2
        root.left = self.fit(X[X[:, attribute] < split], e, improved)
        root.right = self.fit(X[X[:, attribute] >= split], e, improved)
        self.root = root
        return self.root



def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1.0
    start = True
    while start:
        output = np.where(scores >= threshold, 1, 0)
        confusion = confusion_matrix(y.values, output)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        start = TPR < desired_TPR
        threshold -= 0.01
    threshold += 0.01
    return threshold, FPR

