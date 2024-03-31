import logging
import os
import time
from abc import abstractmethod
from typing import Any, Optional

import lightning
import torch.optim

import kooplearn
from kooplearn._src.serialization import pickle_load, pickle_save
from kooplearn.data import TensorContextDataset

logger = logging.getLogger("kooplearn")


class LatentBaseModel(kooplearn.abc.BaseModel):
    r"""Base class for latent models of Markov processes.

    This class defines the interface for discrete-time latent models of Markov processes :math:`(\mathbf{x}_t)_{
    t\\in\\mathbb{
        T}}`,
    where :math:`\mathbf{x}_t` is the system state vector-valued observables at time :math:`t` and :math:`\\mathbb{
    T}` is the
    time index set. These models [...]

    [Suggestion: Since in practice/code we need to work with final-dimensional vector spaces, we will try to
    always highlight the relationship between infinite-dimensional objects (function space, operator) and its
    finite-dimensional representation/approximation (\\mathbb{R}^l`, matrix).] This will make the code more readable
                                                                                                            and
                                                                                                            easily
                                                                                                            modifiable.]

    The class is free to enable the practicioner to define at wish the encoding-decoding process definition, but
    assumes the the evolution of the latent state :math:`\mathbf{z} \\in \\mathcal{Z} \approx \\mathbb{R}^l` is
    modeled by a
    linear evolution operator :math:`T: \\mathcal{Z} \to \\mathcal{Z}` (i.e., approximated by a matrix of shape
    :math:`(l, l)`). Such that :math:`\mathbf{z}_{t+1} = T \\, \mathbf{z}_t`. The spectral decomposition of the
    evolution operator
    :math:`T = V \\Lambda V^*` is assumed to approximate the spectral decomposition of the process's true evolution
    operator. Therefore, the eigenvectors :math:`V` and eigenvalues :math:`\\Lambda` are the approximations of the
    eigenfunctions and eigenvalues of the true evolution operator.
    [TODO:
    Define the functional-analytical spectral decomposition notation and symbols used in the class.
        ... define code/notation conventions for the names and symbols/variable-names used in the class ()
        We will denote the:
        - name: latent state / latent observables  - symbol :math:`\mathbf{z}`    - var_name: z
        - name: encoder / observable_function     - symbol :math:`\\phi` - var_name: encoder
        - name: evolution operator                - symbol :math:`T`    - var_name: evolution_operator
        - name: decoder / observable_function     - symbol :math:`\\psi^-1` - var_name: decoder

    Define the abstract functions input types when useful/needed.
    ]
    """

    @abstractmethod
    def encode_contexts(self, contexts_batch: TensorContextDataset) -> dict[Any]:
        raise NotImplementedError()

    @abstractmethod
    def decode_contexts(self, encoded_contexts_batch: TensorContextDataset) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def evolve_contexts(self, contexts_batch: TensorContextDataset) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def compute_loss_and_metrics(self,
                                 state_context: Optional[TensorContextDataset] = None,
                                 pred_state_context: Optional[TensorContextDataset] = None,
                                 latent_obs_context: Optional[TensorContextDataset] = None,
                                 pred_latent_obs_context: Optional[TensorContextDataset] = None,
                                 **kwargs
                                 ) -> dict[str, torch.Tensor]:
        r"""Compute the loss and metrics of the model.

        Args:
        ----
            state_context: trajectory of states :math:`(x_t)_{t\\in\\mathbb{T}}` in the context window
            :math:`\\mathbb{T}`.
            pred_state_context: predicted trajectory of states :math:`(\\hat{x}_t)_{t\\in\\mathbb{T}}`
            latent_obs_context: trajectory of latent observables :math:`(z_t)_{t\\in\\mathbb{T}}` in the context window
            :math:`\\mathbb{T}`.
            pred_latent_obs_context: predicted trajectory of latent observables :math:`(\\hat{z}_t)_{t\\in\\mathbb{T}}`
            **kwargs:

        Returns:
        -------
            Dictionary containing the key "loss" and other metrics to log.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def evolution_operator(self):
        raise NotImplementedError()

    def save(self, filename: os.PathLike):
        """Serialize the model to a file.

        Args:
        ----
            filename (path-like or file-like): Save the model to file.
        """
        # self.lightning_module._kooplearn_model_weakref = None  ... Why not simply use self reference?
        pickle_save(self, filename)

    @classmethod
    def load(cls, path: os.PathLike) -> 'LatentBaseModel':
        """Load a serialized model from a file.

        Args:
        ----
            filename (path-like or file-like): Load the model from file.

        Returns:
        -------
            Saved instance of `LatentBaseModel`.
        """
        restored_obj = pickle_load(cls, path)
        # Restore the weakref # TODO Why?
        # restored_obj.lightning_module._kooplearn_model_weakref = weakref.ref(
        #     restored_obj
        #     )
        return restored_obj


class LightningLatentModel(lightning.LightningModule):
    """Base `LightningModule` class to define the common codes for training instances of `LatentBaseModels`.

    For most Latent Models, this class should suffice to train the model. User should inherit this class in case he/she
    wants to modify some of the lighting hooks/callbacks or the basic generic pipeline defined in this class.

    DAE, and DPNets models should be trained by this same class instance. So the class should cover the common pipeline
    between Autoencoder based models and representation-learning-then-operator-regression based models.
    """

    def __init__(self,
                 latent_model: LatentBaseModel,
                 optimizer_fn: type[torch.optim.Optimizer] = torch.optim.Adam,
                 optimizer_kwargs: Optional[dict] = None,
                 ):
        self.latent_model: LatentBaseModel = latent_model
        self._optimizer_fn = optimizer_fn
        self._optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        # TODO: Deal with with latent_model hparams if needed.

    def forward(self, state_contexts: TensorContextDataset) -> Any:
        # encoding/observation-function-evaluation =====================================================================
        # Compute z_t = phi(x_t) for all t in the train_batch context_length
        encoder_out = self.latent_model.encode_contexts(state_contexts)

        # Evolution of latent observables ==============================================================================
        # Compute the approximate evolution of the latent state z̄_t for t in look-forward/prediction-horizon
        lin_evolved_out = self.latent_model.evolve_contexts(**encoder_out)

        # decoder/observation-function-inversion =======================================================================
        # Compute the approximate evolution of the state x̄_t for t in look-forward/prediction-horizon
        decoder_out = self.latent_model.decode_contexts(**lin_evolved_out)
        if decoder_out is None:  # Forward passes that do not output the state predictions. E.g. DPNet models
            decoder_out = {}

        return dict(**encoder_out, **lin_evolved_out, **decoder_out)

    def training_step(self, train_contexts: TensorContextDataset, batch_idx):
        model_out = self(train_contexts)
        out = self.latent_model.compute_loss_and_metrics(**model_out)
        assert "loss" in out, f"Loss not found within {out.keys()}"
        loss = out["loss"]
        self.log_metrics(out, suffix="train", batch_size=self._batch_size)
        return loss

    def validation_step(self, val_contexts: TensorContextDataset, batch_idx):
        model_out = self(val_contexts)
        out = self.latent_model.compute_loss_and_metrics(**model_out)
        assert "loss" in out, f"Loss not found within {out.keys()}"
        loss = out["loss"]
        self.log_metrics(out, suffix="val", batch_size=self._batch_size)
        return loss

    def test_step(self, test_contexts: TensorContextDataset, batch_idx):
        model_out = self(test_contexts)
        out = self.latent_model.compute_loss_and_metrics(**model_out)
        assert "loss" in out, f"Loss not found within {out.keys()}"
        loss = out["loss"]
        self.log_metrics(out, suffix="test", batch_size=self._batch_size)
        return loss

    def predict_step(self, batch, batch_idx, **kwargs):
        return self(batch)

    def log_metrics(self, metrics: dict, suffix='', batch_size=None):
        flat_metrics = flatten_dict(metrics)
        for k, v in flat_metrics.items():
            name = f"{k}/{suffix}"
            self.log(name, v, prog_bar=False, batch_size=batch_size)

    def on_train_epoch_start(self) -> None:
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self) -> None:
        self.log('time_per_epoch', time.time() - self._epoch_start_time, prog_bar=False, on_epoch=True)

    def configure_optimizers(self) -> Any:
        if "lr" in self._optimizer_kwargs:  # For Lightning's LearningRateFinder
            self.lr = self._optimizer_kwargs["lr"]
        else:
            self.lr = 1e-3
            self._optimizer_kwargs["lr"] = self.lr
            _class_name = self.__class__.__name__
            logger.warning(
                f"Using default learning rate value lr=1e-3 for {self.__class__.__name__}. "
                f"You can specify the learning rate by passing it to the optimizer_kwargs initialization argument.")
        return self._optimizer_fn(self.parameters(), **self._optimizer_kwargs)


# TODO: Should move to appropriate file
def flatten_dict(d: dict, prefix=''):
    """Flatten a nested dictionary."""
    a = {}
    for k, v in d.items():
        if isinstance(v, dict):
            a.update(flatten_dict(v, prefix=f"{k}/"))
        else:
            a[f"{prefix}{k}"] = v
    return a
