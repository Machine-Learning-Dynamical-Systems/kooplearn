from pathlib import Path
import numpy as np
import optuna 
import ml_confs as mlcfg
from socket import gethostname
import kooplearn

class PolyFeatures(kooplearn.abc.FeatureMap):
    def __init__(self, order: int = 3) -> None:
        super().__init__()
        self.order = order
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.concatenate([X**i for i in range(1, self.order + 1)], axis=1)

def main():
    example_path = Path(__file__).parent.parent
    example_name = Path(__file__).name.split('.')[0]
    host_name = gethostname().split('.')[0]
    STUDY_NAME = example_name + '_' + host_name
    configs = mlcfg.from_file(example_path / 'configs.yaml')

    ds = np.load(example_path / f'dataset_{configs.N}.npz')
    train, test_in, test_out  = ds['train'], ds['test_in'], ds['test_out']
    model = kooplearn.ExtendedDMD(feature_map = PolyFeatures(order = 5))
    model.fit(train[:-1], train[1:])

    eigs = model.eig()

    mae = np.abs(model.predict(test_in) - test_out).mean()
    hausdorff = kooplearn.metrics.one_sided_hausdorff_distance(eigs, ds['eig'])
    projection_score = kooplearn.metrics.projection_score(model.cov_X, model.cov_Y, model.cov_XY)
    

    results = {
        'hausdorff[eig]': hausdorff,
        'mae[predict]': mae,
        'proj[invariance]': projection_score
    }
    return results


if __name__ == '__main__':
    main()