import random
import numpy as np

from scipy import linalg
from scipy.linalg import lapack
import logging

from gpytorch.lazy import RootLazyTensor
from gpytorch.utils.memoize import add_to_cache


def set_random_seed(seed):
    """Set seeds across modules for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)


def minmaxnorm(x, min_value=0.0, max_value=1.0):
    x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
    return x * (max_value - min_value) + min_value


def modify_GPy(GPy_package):
    """
    This function gets the original GPy package and modifies two functions in GPy/util/linalg.py:
    1. To avoid issues with the jitchol method (SheffieldML/GPy#660), we used np.linalg.cholesky
       instead of scipy.linalg.cholesky and raised the number of regularization attempts.
    2. We also removed a warning in force_F_ordered about the arrays not being F order.
    """
    def jitchol(A, maxtries=100):
        A = np.ascontiguousarray(A)
        L, info = lapack.dpotrf(A, lower=1)
        if info == 0:
            return L
        else:
            diagA = np.diag(A)
            if np.any(diagA <= 0.):
                raise linalg.LinAlgError("not pd: non-positive diagonal elements")
            jitter = diagA.mean() * 1e-6
            num_tries = 1
            while num_tries <= maxtries and np.isfinite(jitter):
                try:
                    L = np.linalg.cholesky(A + np.eye(A.shape[0]) * jitter)
                    return L
                except:
                    jitter *= 10
                finally:
                    num_tries += 1
            raise linalg.LinAlgError("not positive definite, even with jitter.")
        import traceback
        try:
            raise
        except:
            logging.warning('\n'.join(['Added jitter of {:.10e}'.format(jitter), '  in ' + traceback.format_list(traceback.extract_stack(limit=3)[-2:-1])[0][2:]]))
        return L

    def force_F_ordered(A):
        if A.flags['F_CONTIGUOUS']:
            return A
        return np.asfortranarray(A)

    GPy_package.util.linalg.jitchol = jitchol
    GPy_package.util.linalg.force_F_ordered = force_F_ordered

    return GPy_package


def modify_gpytorch(gpytorch_package):
    """
    This function gets the gpytorch package and modifies the init in
    gpytorch/models/exact_prediction_strategies.py
    """
    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood, root=None, inv_root=None):
        # Get training shape
        self._train_shape = train_prior_dist.event_shape

        # Flatten the training labels
        # train_labels = train_labels.reshape(*train_labels.shape[: -len(self.train_shape)], self._train_shape.numel())
        train_labels = train_labels.reshape(self._train_shape.numel())

        self.train_inputs = train_inputs
        self.train_prior_dist = train_prior_dist
        self.train_labels = train_labels
        self.likelihood = likelihood
        self._last_test_train_covar = None
        mvn = self.likelihood(train_prior_dist, train_inputs)
        self.lik_train_train_covar = mvn.lazy_covariance_matrix

        if root is not None:
            add_to_cache(self.lik_train_train_covar, "root_decomposition", RootLazyTensor(root))

        if inv_root is not None:
            add_to_cache(self.lik_train_train_covar, "root_inv_decomposition", RootLazyTensor(inv_root))

    gpytorch_package.models.exact_prediction_strategies.DefaultPredictionStrategy.__init__ = __init__

    return gpytorch_package
