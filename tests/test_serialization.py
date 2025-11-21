from pathlib import Path
from shutil import rmtree

import lightning
import numpy as np
import pytest
import torch

from kooplearn.data import traj_to_contexts
from kooplearn.datasets import Mock
from kooplearn.models import ConsistentAE, DynamicAE
from kooplearn.models.feature_maps import NNFeatureMap
from kooplearn.nn import DPLoss, VAMPLoss
from kooplearn.nn.data import collate_context_dataset

NUM_SAMPLES = 100
DIM = 7

TRAJ = Mock(num_features=DIM, rng_seed=0).sample(None, NUM_SAMPLES).astype(np.float32)
DATA = traj_to_contexts(TRAJ)
NN_DATALOADER = torch.utils.data.DataLoader(
    DATA, batch_size=100, shuffle=True, collate_fn=collate_context_dataset
)


def _make_tmp_path(model_name: str):
    tmp_path = Path(__file__).parent / f"tmp/{model_name}.bin"
    return tmp_path


def _cleanup():
    tmp_path = Path(__file__).parent / "tmp"
    rmtree(tmp_path)


def _allclose(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.allclose(a[m], b[m], rtol=1e-3, atol=1e-5)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(DIM, 32)

    def forward(self, x):
        return self.encoder(x)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(32, DIM)

    def forward(self, x):
        return self.decoder(x)


def test_Kernel_serialization():
    from kooplearn.models import Kernel

    model = Kernel().fit(DATA)
    tmp_path = _make_tmp_path(model.__class__.__name__)
    model.save(tmp_path)
    restored_model = Kernel.load(tmp_path)

    # Check that predict, eig and modes return the same values
    assert _allclose(
        model.predict(DATA),
        restored_model.predict(DATA),
    )

    assert _allclose(
        model.modes(DATA)[0],
        restored_model.modes(DATA)[0],
    )

    e1, l1, r1 = model.eig(
        eval_left_on=DATA,
        eval_right_on=DATA,
    )
    e2, l2, r2 = restored_model.eig(
        eval_left_on=DATA,
        eval_right_on=DATA,
    )

    assert _allclose(e1, e2)
    assert _allclose(l1, l2)
    assert _allclose(r1, r2)

    _cleanup()


def test_Nonlinear_serialization():
    from kooplearn.models import Nonlinear

    model = Nonlinear().fit(DATA)
    tmp_path = _make_tmp_path(model.__class__.__name__)
    model.save(tmp_path)
    restored_model = Nonlinear.load(tmp_path)

    # Check that predict, eig and modes return the same values
    assert _allclose(
        model.predict(DATA),
        restored_model.predict(DATA),
    )

    assert _allclose(
        model.modes(DATA)[0],
        restored_model.modes(DATA)[0],
    )

    e1, l1, r1 = model.eig(
        eval_left_on=DATA,
        eval_right_on=DATA,
    )
    e2, l2, r2 = restored_model.eig(
        eval_left_on=DATA,
        eval_right_on=DATA,
    )

    assert _allclose(e1, e2)
    assert _allclose(l1, l2)
    assert _allclose(r1, r2)

    _cleanup()


@pytest.mark.parametrize("feature_map_loss", [DPLoss, VAMPLoss])
def test_NN_Nonlinear_serialization(feature_map_loss):
    from kooplearn.models import Nonlinear

    trainer = lightning.Trainer(
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator="cpu",
        max_epochs=1,
    )
    feature_map = NNFeatureMap(
        Encoder,
        feature_map_loss,
        torch.optim.SGD,
        trainer,
        optimizer_kwargs={"lr": 1e-6},
        seed=0,
    )
    feature_map.fit(NN_DATALOADER)

    tmp_path = _make_tmp_path(feature_map.__class__.__name__)
    feature_map.save(tmp_path)
    restored_feature_map = NNFeatureMap.load(tmp_path)

    assert _allclose(feature_map(TRAJ), restored_feature_map(TRAJ))

    model = Nonlinear(feature_map=feature_map).fit(DATA)
    tmp_path = _make_tmp_path(model.__class__.__name__)
    model.save(tmp_path)
    restored_model = Nonlinear.load(tmp_path)

    # Check that predict, eig and modes return the same values
    assert _allclose(
        model.predict(DATA),
        restored_model.predict(DATA),
    )

    assert _allclose(
        model.modes(DATA)[0],
        restored_model.modes(DATA)[0],
    )

    e1, l1, r1 = model.eig(
        eval_left_on=DATA,
        eval_right_on=DATA,
    )
    e2, l2, r2 = restored_model.eig(
        eval_left_on=DATA,
        eval_right_on=DATA,
    )

    assert _allclose(e1, e2)
    assert _allclose(l1, l2)
    assert _allclose(r1, r2)

    _cleanup()


@pytest.mark.parametrize("model_cls", [DynamicAE, ConsistentAE])
def test_AE_serialization(model_cls):
    trainer = lightning.Trainer(
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator="cpu",
        max_epochs=1,
    )
    model = model_cls(
        Encoder,
        Decoder,
        32,
        torch.optim.SGD,
        trainer,
        optimizer_kwargs={"lr": 1e-6},
        seed=0,
    )
    model.fit(NN_DATALOADER)

    tmp_path = _make_tmp_path(model.__class__.__name__)
    model.save(tmp_path)
    restored_model = model_cls.load(tmp_path)

    # Check that predict, eig and modes return the same values
    assert _allclose(
        model.predict(DATA),
        restored_model.predict(DATA),
    )

    e1, l1, r1 = model.eig(
        eval_left_on=DATA,
        eval_right_on=DATA,
    )
    e2, l2, r2 = restored_model.eig(
        eval_left_on=DATA,
        eval_right_on=DATA,
    )

    assert _allclose(e1, e2)
    assert _allclose(l1, l2)
    assert _allclose(r1, r2)

    _cleanup()
