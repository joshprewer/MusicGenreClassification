import array
import numpy as np
import warnings
import scipy.sparse as sp
import itertools

from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.base import MetaEstimatorMixin, is_regressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import (_check_partial_fit_first_call,
                               check_classification_targets,
                               _ovr_decision_function)
from sklearn.utils.metaestimators import _safe_split, if_delegate_has_method

from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed
from sklearn.externals.six.moves import zip as izip
from feature_selection import binary_optimization, SAHS, ReliefFSFS

__all__ = [
    "OneVsOneFeatureSelection",
    "OneVsAllFeatureSelection",
]

def _fit_binary(estimator, X, y, classes=None):
    """Fit a single binary estimator."""

    n_iterations = 300
    hc = SAHS.SAHS(SAHS.SAHSObjectiveFunction(X, y, harmony_menmory_size=20, hmcr_proba=0.99, iteration=n_iterations, clf=estimator))
    hms, hms_scores, max_score_index = hc.run()
    hs_features = np.nonzero(hms[max_score_index])[0]
    
    obj = binary_optimization.Evaluate(X, y, clf=estimator)
    cs_score, g, l = binary_optimization.BCS(obj, m_i=n_iterations)
    cs_features = np.nonzero(g)[0]

    obj = binary_optimization.Evaluate(X, y, clf=estimator)
    dfa_score, g, l = binary_optimization.BDFA(obj, m_i=n_iterations)
    dfa_features = np.nonzero(g)[0]

    return [hs_features, cs_features, dfa_features]

def _fit_ovo_binary(estimator, X, y, i, j):
    """Fit a single binary estimator (one-vs-one)."""
    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    y_binary = np.empty(y.shape, np.int)
    y_binary[y == i] = 0
    y_binary[y == j] = 1
    indcond = np.arange(X.shape[0])[cond]
    return _fit_binary(estimator,
                       _safe_split(estimator, X, None, indices=indcond)[0],
                       y_binary, classes=[i, j]), [i, j]

class OneVsRestClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """One-vs-the-rest (OvR) multiclass/multilabel strategy
    Also known as one-vs-all, this strategy consists in fitting one classifier
    per class. For each classifier, the class is fitted against all the other
    classes. In addition to its computational efficiency (only `n_classes`
    classifiers are needed), one advantage of this approach is its
    interpretability. Since each class is represented by one and one classifier
    only, it is possible to gain knowledge about the class by inspecting its
    corresponding classifier. This is the most commonly used strategy for
    multiclass classification and is a fair default choice.
    This strategy can also be used for multilabel learning, where a classifier
    is used to predict multiple labels for instance, by fitting on a 2-d matrix
    in which cell [i, j] is 1 if sample i has label j and 0 otherwise.
    In the multilabel learning literature, OvR is also known as the binary
    relevance method.
    Read more in the :ref:`User Guide <ovr_classification>`.
    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and one of `decision_function`
        or `predict_proba`.
    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Attributes
    ----------
    estimators_ : list of `n_classes` estimators
        Estimators used for predictions.
    classes_ : array, shape = [`n_classes`]
        Class labels.
    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and
        vice-versa.
    multilabel_ : boolean
        Whether a OneVsRestClassifier is a multilabel classifier.
    """
    def __init__(self, estimator, n_jobs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-class targets. An indicator matrix turns on multilabel
            classification.
        Returns
        -------
        self
        """
        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_binary)(
            self.estimator, X, column, classes=[
                "not %s" % self.label_binarizer_.classes_[i],
                self.label_binarizer_.classes_[i]])
            for i, column in enumerate(columns))

        return self.estimators_

    @if_delegate_has_method(['_first_estimator', 'estimator'])
    def decision_function(self, X):
        """Returns the distance of each sample from the decision boundary for
        each class. This can only be used with estimators which implement the
        decision_function method.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
        """
        check_is_fitted(self, 'estimators_')
        if len(self.estimators_) == 1:
            return self.estimators_[0].decision_function(X)
        return np.array([est.decision_function(X).ravel()
                         for est in self.estimators_]).T

    @property
    def multilabel_(self):
        """Whether this is a multilabel classifier"""
        return self.label_binarizer_.y_type_.startswith('multilabel')

    @property
    def n_classes_(self):
        return len(self.classes_)

    @property
    def coef_(self):
        check_is_fitted(self, 'estimators_')
        if not hasattr(self.estimators_[0], "coef_"):
            raise AttributeError(
                "Base estimator doesn't have a coef_ attribute.")
        coefs = [e.coef_ for e in self.estimators_]
        if sp.issparse(coefs[0]):
            return sp.vstack(coefs)
        return np.vstack(coefs)

    @property
    def intercept_(self):
        check_is_fitted(self, 'estimators_')
        if not hasattr(self.estimators_[0], "intercept_"):
            raise AttributeError(
                "Base estimator doesn't have an intercept_ attribute.")
        return np.array([e.intercept_.ravel() for e in self.estimators_])

    @property
    def _pairwise(self):
        """Indicate if wrapped estimator is using a precomputed Gram matrix"""
        return getattr(self.estimator, "_pairwise", False)

    @property
    def _first_estimator(self):
        return self.estimators_[0]


def _fit_ovo_binary(estimator, X, y, i, j):
    """Fit a single binary estimator (one-vs-one)."""
    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    y_binary = np.empty(y.shape, np.int)
    y_binary[y == i] = 0
    y_binary[y == j] = 1
    indcond = np.arange(X.shape[0])[cond]
    return _fit_binary(estimator,
                       _safe_split(estimator, X, None, indices=indcond)[0],
                       y_binary, classes=[i, j]), indcond


class OneVsOneFeatureSelection(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    def __init__(self, estimator, n_jobs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        y : array-like, shape = [n_samples]
            Multi-class targets.
        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("OneVsOneClassifier can not be fit when only one"
                             " class is present.")
        n_classes = self.classes_.shape[0]
        estimators_indices = list(zip(*(Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_ovo_binary)
            (self.estimator, X, y, self.classes_[i], self.classes_[j])
            for i in range(n_classes) for j in range(i + 1, n_classes)))))

        self.estimators_ = estimators_indices[0]
        # try:
        #     self.pairwise_indices_ = (
        #         estimators_indices[1] if self._pairwise else None)
        # except AttributeError:
        #     self.pairwise_indices_ = None

        return estimators_indices

    @property
    def n_classes_(self):
        return len(self.classes_)

    @property
    def _pairwise(self):
        """Indicate if wrapped estimator is using a precomputed Gram matrix"""
        return getattr(self.estimator, "_pairwise", False)
