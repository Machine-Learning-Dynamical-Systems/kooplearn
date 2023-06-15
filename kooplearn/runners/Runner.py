class Runner:
    def __init__(self, feature_map, koopman_estimator, decoder, dataset):
        self.feature_map = feature_map
        self.koopman_estimator = koopman_estimator
        self.decoder = decoder
        self.dataset = dataset

    def fit_feature_map(self):
        pass

    def fit_koopman(self):
        pass

    def test(self):
        pass

    def initialize(self):
        pass

    def run(self):
        self.initialize()
        self.fit_feature_map()
        self.fit_koopman()
        self.test()
