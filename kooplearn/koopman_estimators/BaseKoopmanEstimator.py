from sklearn.base import BaseEstimator


class BaseKoopmanEstimator(BaseEstimator):
    def __init__(self):
        pass

    def _check_backend_solver_compatibility(self):
        if self.backend not in ['numpy']:
            raise ValueError('Invalid backend. Allowed values is \'numpy\'.')
        if self.svd_solver not in ['full', 'arnoldi', 'randomized']:
            raise ValueError('Invalid svd_solver. Allowed values are \'full\', \'arnoldi\' and \'randomized\'.')
        if self.svd_solver == 'randomized' and self.iterated_power < 0:
            raise ValueError('Invalid iterated_power. Must be non-negative.')
        if self.svd_solver == 'randomized' and self.n_oversamples < 0:
            raise ValueError('Invalid n_oversamples. Must be non-negative.')
        return

    def fit(self):
        pass

    def forecast(self):
        pass
