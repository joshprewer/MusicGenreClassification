import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn import metrics, svm, model_selection, utils

class ReliefF(object):

    """Feature selection using data-mined expert knowledge.
    
    Based on the ReliefF algorithm as introduced in:
    
    Kononenko, Igor et al. Overcoming the myopia of inductive learning algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55
    
    """
    
    def __init__(self, n_neighbors=100, n_features_to_keep=10):
        """Sets up ReliefF to perform feature selection.
        Parameters
        ----------
        n_neighbors: int (default: 100)
            The number of neighbors to consider when assigning feature importance scores.
            More neighbors results in more accurate scores, but takes longer.
        Returns
        -------
        None
        """
        
        self.feature_scores = None
        self.top_features = None
        self.tree = None
        self.n_neighbors = n_neighbors
        self.n_features_to_keep = n_features_to_keep
    
    def fit(self, X, y):
        """Computes the feature importance scores from the training data.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        Returns
        -------
        None
        """
        self.feature_scores = np.zeros(X.shape[1])
        self.tree = KDTree(X)

        for source_index in range(X.shape[0]):
            distances, indices = self.tree.query(X[source_index].reshape(1, -1), k=self.n_neighbors + 1)

            # First match is self, so ignore it
            for neighbor_index in indices[0][1:]:
                similar_features = X[source_index] == X[neighbor_index]
                label_match = y[source_index] == y[neighbor_index]

                # If the labels match, then increment features that match and decrement features that do not match
                # Do the opposite if the labels do not match
                if label_match:
                    self.feature_scores[similar_features] += 1.
                    self.feature_scores[~similar_features] -= 1.
                else:
                    self.feature_scores[~similar_features] += 1.
                    self.feature_scores[similar_features] -= 1.
        
        self.top_features = np.argsort(self.feature_scores)[::-1]
        
    def transform(self, X):
        """Reduces the feature set down to the top `n_features_to_keep` features.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on
        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix
        """
        return X[:, self.top_features[:len(X)]]

class ReliefFSFS(object):

    def __init__(self, n_neighbors=100, n_features_to_keep=10):
        """Sets up ReliefF to perform feature selection.
        Parameters
        ----------
        n_neighbors: int (default: 100)
            The number of neighbors to consider when assigning feature importance scores.
            More neighbors results in more accurate scores, but takes longer.
        Returns
        -------
        None
        """
        self.feature_scores = None
        self.top_features = None
        self.tree = None
        self.n_neighbors = n_neighbors
        self.n_features_to_keep = n_features_to_keep


    def transform(self, X, y, fuunction):
        """Reduces the feature set down to the top `n_features_to_keep` features.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on
        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix
        """
        reliefF = ReliefF()
        reliefF.fit(X, y)
        weighted_features = reliefF.transform(X)

        optimal_features = np.atleast_2d()
        optimal_features = np.resize(optimal_features, 1000)
        previous_results = 0
        
        for feature in range(weighted_features.shape[1]):
            feature_to_test = weighted_features[:, feature]
            optimal_features = np.vstack((optimal_features, feature_to_test))
            classifier = function
            cv_results = model_selection.cross_val_score(classifier, pd.DataFrame(data=np.transpose(optimal_features)), y, cv=10)

            if (cv_results.mean() > previous_results):
                previous_results = cv_results.mean()
            else:
                optimal_features = np.delete(optimal_features, optimal_features.shape[0] - 1, axis=0)


        return np.transpose(optimal_features)