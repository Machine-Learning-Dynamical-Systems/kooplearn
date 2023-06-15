from .Trainer import Trainer


class DNNTrainer(Trainer):
    def fit_feature_map(self):
        self.feature_map.fit(self.dataset, self.koopman_estimator)
        