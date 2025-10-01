from sklearn.base import BaseEstimator

class Kernel(BaseEstimator):
    def __init__(
        self,
        n_components=None,
        *,
        kernel="linear",
        kernel_params=None,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        random_state=None,
        copy_X=True,
        n_jobs=None,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.copy_X = copy_X

        pass

    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # tags.target_tags.single_output = False
        # tags.non_deterministic = True
        return tags