import logging
import os
import time
from abc import abstractmethod
from typing import Any, Optional, Union

import lightning
import numpy as np
import scipy
import torch.optim

import kooplearn
from kooplearn._src.serialization import pickle_load, pickle_save
from kooplearn._src.utils import check_is_fitted
from kooplearn.data import TensorContextDataset
from kooplearn.models.ae.utils import flatten_context_data, multi_matrix_power, unflatten_context_data

logger = logging.getLogger("kooplearn")


class LatentBaseModel(kooplearn.abc.BaseModel, torch.nn.Module):
    r"""Base class for latent models of Markov processes.

    This class defines the interface for discrete-time latent models of Markov processes :math:`(\mathbf{x}_t)_{
    t\\in\\mathbb{T}}`, where :math:`\mathbf{x}_t` is the system state vector-valued observables at time :math:`t`
    and :math:`\\mathbb{T}` is the time index set. These models [...]

    [Suggestion: Since in practice/code we need to work with final-dimensional vector spaces, we will try to
    always highlight the relationship between infinite-dimensional objects (function space, operator) and its
    finite-dimensional representation/approximation (\\mathbb{R}^l`, matrix).] This will make the code more readable
    and easily modifiable.]

    The class is free to enable the practitioner to define at wish the encoding-decoding process definition, but
    assumes the the evolution of the latent state :math:`\mathbf{z} \\in \\mathcal{Z} \approx \\mathbb{R}^l` is
    modeled by a linear evolution operator :math:`T: \\mathcal{Z} \to \\mathcal{Z}` (i.e., approximated by a matrix
    of shape
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

    def encode_contexts(self, state: TensorContextDataset, **kwargs) -> Union[dict, TensorContextDataset]:
        r"""Encodes the given state into the latent space using the model's encoder.

        Args:
            state (TensorContextDataset): The state to be encoded. This should be a trajectory of states
            :math:`(x_t)_{t\\in\\mathbb{T}}` in the context window :math:`\\mathbb{T}`.

        Returns:
            Either of the following:
            - TensorContextDataset: trajectory of encoded latent observables :math:`z_t`
            - dict: A dictionary containing the key "latent_obs" mapping to a TensorContextDataset
        """
        encoder = self.encoder
        # From (batch, context_length, *features_shape) to (batch * context_length, *features_shape)
        flat_encoded_contexts = encoder(flatten_context_data(state))
        # From (batch * context_length, latent_dim) to (batch, context_length, latent_dim)
        latent_obs_contexts = unflatten_context_data(flat_encoded_contexts,
                                                     batch_size=len(state),
                                                     features_shape=(self.latent_dim,))
        return latent_obs_contexts

    def decode_contexts(self, latent_obs: TensorContextDataset, **kwargs) -> Union[dict, TensorContextDataset]:
        r"""Decodes the given latent observables back into the original state space using the model's decoder.

        Args:
            latent_obs (TensorContextDataset): The latent observables to be decoded. This should be a trajectory of
                latent observables :math:`(z_t)_{t\\in\\mathbb{T}}` in the context window :math:`\\mathbb{T}`.

        Returns:
            Either of the following:
            - TensorContextDataset: trajectory of decoded states :math:`x_t`
            - dict: A dictionary containing the key "decoded_contexts" mapping to a TensorContextDataset
        """
        if self.decoder is not None:
            decoder = self.decoder
            # From (batch, context_length, latent_dim) to (batch * context_length, latent_dim)
            flat_decoded_contexts = decoder(flatten_context_data(latent_obs))
            # From (batch * context_length, *features_shape) to (batch, context_length, *features_shape)
            decoded_contexts = unflatten_context_data(flat_decoded_contexts,
                                                      batch_size=len(latent_obs),
                                                      features_shape=self.state_features_shape)
            return decoded_contexts
        else:
            return None

    def evolve_contexts(self, latent_obs: TensorContextDataset, **kwargs) -> Union[dict, TensorContextDataset]:
        r"""Evolves the given latent observables forward in time using the model's evolution operator.

        If the model has been fitted to forecast we avoid powering the evolution operator and instead use the
        eigendecomposition of the operator to compute the evolution of the latent observables.

        Args:
            latent_obs (TensorContextDataset): The latent observables to be evolved forward. This should be a
            trajectory of latent observables
            :math:`(z_t)_{t\\in\\mathbb{T}}` in the context window :math:`\\mathbb{T}`.

        Returns:
            Either of the following:
            - TensorContextDataset: trajectory of predicted latent observables :math:`\\hat{z}_t`
            - dict: A dictionary containing the key "pred_latent_obs" mapping to a TensorContextDataset
        """
        assert latent_obs.data.ndim == 3, \
            f"Expected tensor of shape (batch, context_len, latent_dim), got {latent_obs.data.shape}"
        assert latent_obs.data.shape[2] == self.latent_dim, \
            f"Expected latent dimension {self.latent_dim}, got {latent_obs.data.shape[2]}"

        context_length = latent_obs.context_length

        # Initial condition to evolve in time.
        z_0 = latent_obs.lookback(self.lookback_len).squeeze()

        if self.is_fitted:
            if not hasattr(self, "_eigvals"): self.eig()  # Ensure eigendecomposition is in cache
            # T = V Λ V^-1 : V = eigvecs_r, Λ = eigvals, V^-1 = eigvecs_r_inv
            eigvals, eigvecs_r, eigvecs_r_inv = self._eigvals, self._eigvecs_r, self._eigvecs_r_inv
            z_0_eigbasis = torch.einsum("oi,...i->...o", eigvecs_r_inv.data, z_0.to(dtype=eigvecs_r.dtype))
            # Compute the powers of the eigenvalues used to evolve the latent state z_t | t in [0, context_length]
            # powered_eigvals: (context_length, latent_dim) -> [1, λ, λ^2, ..., λ^context_length]
            exponents = torch.arange(context_length, device=z_0.device).unsqueeze(1)
            powered_eigvals = eigvals.pow(exponents)
            # Compute z_t_eigbasis[batch, t] = Λ^t V^-1 z_0 | t in [0, context_length]
            z_t_eigbasis = torch.einsum("to,...o->...to", powered_eigvals, z_0_eigbasis)
            # Convert back to the original basis z_t[batch,t] = V Λ^t V^-1 z_0 | t in [0, context_length]
            z_t = torch.einsum("oi,...ti->...to", eigvecs_r.data, z_t_eigbasis).real.to(dtype=z_0.dtype)
        else:
            # T : (latent_dim, latent_dim)
            evolution_operator = self.evolution_operator()
            # z_0: (..., latent_dim)
            # Compute the powers of the evolution operator used T_t such that z_t = T_t @ z_0 | t in [0, context_length]
            # powered_evolution_ops: (context_length, latent_dim, latent_dim) -> [I, T, T^2, ..., T^context_length]
            powered_evolution_ops = multi_matrix_power(evolution_operator, context_length)
            # Compute evolved latent observable states z_t | t in [0, context_length] (single parallel operation)
            z_t = torch.einsum("toi,...i->...to", powered_evolution_ops, z_0)

        z_pred_t = TensorContextDataset(z_t)
        return z_pred_t

    def evolve_forward(self, state: TensorContextDataset) -> dict:
        r"""Evolves the given state forward in time using the model's encoding, evolution, and decoding processes.

        This method first encodes the given state into the latent space using the model's encoder.
        It then evolves the encoded state forward in time using the model's evolution operator.
        If a decoder is defined, it decodes the evolved latent state back into the original state space.
        The method also performs a reconstruction of the original state from the encoded latent state.

        Args:
            state (TensorContextDataset): The state to be evolved forward. This should be a trajectory of states
            :math:`(x_t)_{t\\in\\mathbb{T}}` in the context window :math:`\\mathbb{T}`.

        Returns:
            dict: A dictionary containing the following keys:
                - `latent_obs`: The latent observables, :math:`z_t`. They represent the encoded state in the latent
                space.
                - `pred_latent_obs`: The predicted latent observables, :math:`\\hat{z}_t`. They represent the model's
                prediction of the latent observables.
                - `pred_state`: (Optional) The predicted state, :math:`\\hat{x}_t`. This can be None if the decoder
                is not defined.
                - `rec_state`: (Optional) The reconstructed state, :math:`\\tilde{x}_t`. It represents the state that
                the model reconstructs from the latent observables. This can be None if the decoder is not defined.
                The dictionary may also contain additional outputs from the `encode_contexts`, `decode_contexts`,
                and `evolve_contexts`.
        """
        # encoding/observation-function-evaluation =====================================================================
        # Compute z_t = phi(x_t) for all t in the train_batch context_length
        encoder_out = self.encode_contexts(state)
        z_t = encoder_out if isinstance(encoder_out, TensorContextDataset) else encoder_out.pop("latent_obs")
        encoder_out = {} if isinstance(encoder_out, TensorContextDataset) else encoder_out

        # Evolution of latent observables ==============================================================================
        # Compute the approximate evolution of the latent state z̄_t for t in look-forward/prediction-horizon
        evolved_out = self.evolve_contexts(latent_obs=z_t, **encoder_out)
        pred_z_t = evolved_out if isinstance(evolved_out, TensorContextDataset) else evolved_out.pop("pred_latent_obs")
        evolved_out = {} if isinstance(evolved_out, TensorContextDataset) else evolved_out

        # (Optional) decoder/observation-function-inversion ============================================================
        # Compute the approximate evolution of the state x̄_t for t in look-forward/prediction-horizon
        decoder_out = self.decode_contexts(latent_obs=pred_z_t, **evolved_out)
        pred_x_t = None
        if decoder_out is not None:
            pred_x_t = decoder_out if isinstance(decoder_out, TensorContextDataset) else decoder_out.pop(
                "decoded_contexts")
            decoder_out = {} if isinstance(decoder_out, TensorContextDataset) else decoder_out

        return dict(latent_obs=z_t,
                    pred_latent_obs=pred_z_t,
                    pred_state=pred_x_t,
                    # Attached any additional outputs from encoder/decoder/evolution
                    **encoder_out,
                    **evolved_out,
                    **decoder_out)

    @torch.no_grad
    def eig(self,
            eval_left_on: Optional[TensorContextDataset] = None,
            eval_right_on: Optional[TensorContextDataset] = None,
            ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Returns the eigenvalues of the Koopman/Transfer operator and optionally evaluates left and right
        eigenfunctions.

        TODO: Should make everything default to backend torch. As the entire model is on torch, independent on device
        TODO: Should improve the documentation of the method.

        Args:
            eval_left_on (TensorContextDataset or None): Dataset of context windows on which the left eigenfunctions
            are evaluated.
            eval_right_on (TensorContextDataset or None): Dataset of context windows on which the right
            eigenfunctions are evaluated.

        Returns:
            Eigenvalues of the Koopman/Transfer operator, shape ``(rank,)``. If ``eval_left_on`` or ``eval_right_on``
             are not ``None``, returns the left/right eigenfunctions evaluated at ``eval_left_on``/``eval_right_on``:
             shape ``(n_samples, rank)``.
        """
        if not hasattr(self, "_eigvals"):
            K = self.evolution_operator()
            K_np = K.detach().cpu().numpy()
            # K is a square real-valued matrix.
            eigvals, eigvecs_l, eigvecs_r = scipy.linalg.eig(K_np, left=True, right=True)
            # Left and right eigenvectors are stored in columns: eigvecs_l/r[:, i] is the i-th left/right eigenvector
            # This we have that:
            # K @ eigvecs_r[:, i] = eigvals[i] @ eigvecs_r[:, i] <==>  K = eigvecs_r @ eigvals[i] @ eigvecs_r^-1
            # assert np.allclose(eigvecs_r @ np.diag(eigvals) @ eigvecs_r.conj().T, K_np, rtol=1e-5, atol=1e-5)
            # The left eigenvectors are the ones associated to K^T.
            # K^T @ eigvecs_l[:, i] = eigvals[i]^H @ eigvecs_l[:, i] <==>  K^T = eigvecs_l @ eigvals[i]^H @ eigvecs_l^-1
            # assert np.allclose(eigvecs_l @ np.diag(eigvals.conj()) @ np.linalg.inv(eigvecs_l), K_np.T, rtol=1e-5)

            # Store as torch parameters for lighting to manage device automatically. And forecast avoiding matrix power
            # _dtype = torch.complex32 if K.dtype == torch.float32 else torch.complex64
            _dtype = None
            self._eigvals = torch.nn.Parameter(torch.tensor(eigvals, device=K.device, dtype=_dtype),
                                               requires_grad=False)
            self._eigvecs_l = torch.nn.Parameter(torch.tensor(eigvecs_l, device=K.device, dtype=_dtype),
                                                 requires_grad=False)
            self._eigvecs_r = torch.nn.Parameter(torch.tensor(eigvecs_r, device=K.device, dtype=_dtype),
                                                 requires_grad=False)
            eigvecs_r_inv = np.linalg.inv(eigvecs_r)
            assert np.allclose((eigvecs_r @ np.diag(eigvals) @ eigvecs_r_inv).real, K_np, rtol=1e-6, atol=1e-6)
            self._eigvecs_r_inv = torch.nn.Parameter(torch.tensor(eigvecs_r_inv, device=K.device, dtype=_dtype),
                                                     requires_grad=False)

        eigvals, eigvecs_l, eigvecs_r = self._eigvals, self._eigvecs_l, self._eigvecs_r

        left_eigfn, right_eigfn = None, None
        if eval_right_on is not None:
            # Ensure data is on the same device as the model
            eval_right_on.to(device=self.evolution_operator().device, dtype=self.evolution_operator().dtype)
            # Compute the latent observables for the data (batch, context_len, latent_dim)
            z_t = self.encode_contexts(state=eval_right_on)
            # Evaluation of eigenfunctions in (batch/n_samples, context_len, latent_dim)
            right_eigfn = torch.einsum("...s,so->...o", z_t, eigvecs_l)  # TODO: Check this is correct.

        if eval_left_on is not None:
            # Ensure data is on the same device as the model
            eval_left_on.to(device=self.evolution_operator().device, dtype=self.evolution_operator().dtype)
            # Compute the latent observables for the data (batch, context_len, latent_dim)
            z_t = self.encode_contexts(state=eval_left_on)
            # Evaluation of eigenfunctions in (batch/n_samples, context_len, latent_dim)
            # left_eigfn[...,t, i] = <v_i, z_t>_C : i=1,...,l
            left_eigfn = torch.einsum("...il,...l->...i", torch.linalg.inv(eigvecs_r), z_t)

        # TODO: Does it make sense for DeepLearning based models to return as numpy? ...
        eigvals = eigvals.detach().cpu().numpy()
        left_eigfn = left_eigfn.detach().cpu().numpy() if left_eigfn is not None else None
        right_eigfn = right_eigfn.detach().cpu().numpy() if right_eigfn is not None else None

        if eval_left_on is None and eval_right_on is None:
            return eigvals
        elif eval_left_on is None and eval_right_on is not None:
            return eigvals, right_eigfn
        elif eval_left_on is not None and eval_right_on is None:
            return eigvals, left_eigfn
        else:
            return eigvals, left_eigfn, right_eigfn

    def predict(
            self,
            data: TensorContextDataset,
            t: int = 1,
            predict_observables: bool = True,
            reencode_every: int = 0,
            ):
        r"""Predicts the state or, if the system is stochastic, its expected value :math:`\mathbb{E}[X_t | X_0 = X]`
        after ``t`` instants given the initial conditions ``data.lookback(self.lookback_len)`` being the lookback
        slice of ``data``.
        If ``data.observables`` is not ``None``, returns the analogue quantity for the observable instead.

        Args:
            data (TensorContextDataset): Dataset of context windows. The lookback window of ``data`` will be used as
            the initial condition, see the note above.
            t (int): Number of steps in the future to predict (returns the last one).
            predict_observables (bool): Return the prediction for the observables in ``data.observables``,
            if present. Defaults to ``True``.
            reencode_every (int): When ``t > 1``, periodically reencode the predictions as described in
            :footcite:t:`Fathi2023`. Only available when ``predict_observables = False``.

        Returns:
           The predicted (expected) state/observable at time :math:`t`. The result is composed of arrays with shape
           matching ``data.lookforward(self.lookback_len)`` or the contents of ``data.observables``. If
           ``predict_observables = True`` and ``data.observables != None``, the returned ``dict``will contain the
           special key ``__state__`` containing the prediction for the state as well.
        """
        # TODO: Requires update
        raise NotImplementedError("This method is not updated yet.")
        check_is_fitted(self, ["_state_trail_dims"])
        assert tuple(data.shape[2:]) == self._state_trail_dims

        data = self._to_torch(data)
        if predict_observables and hasattr(data, "observables"):
            observables = data.observables
            observables["__state__"] = None
        else:
            observables = {"__state__": None}

        results = {}
        for obs_name, obs in observables.items():
            if (reencode_every > 0) and (t > reencode_every):
                if (predict_observables is True) and (observables is not None):
                    raise ValueError(
                        "rencode_every only works when forecasting states, not observables. Consider setting "
                        "predict_observables to False."
                        )
                else:
                    num_reencodings = floor(t / reencode_every)
                    for k in range(num_reencodings):
                        raise NotImplementedError
            else:
                with torch.no_grad():
                    evolved_data = evolve_forward(
                        data,
                        self.lookback_len,
                        t,
                        self.lightning_module.encoder,
                        self.lightning_module.decoder,
                        self.lightning_module.evolution_operator,
                        )
                    evolved_data = evolved_data.data.detach().cpu().numpy()
                    if obs is None:
                        results[obs_name] = evolved_data
                    elif callable(obs):
                        results[obs_name] = obs(evolved_data)
                    else:
                        raise ValueError(
                            "Observables must be either None, or callable."
                            )

        if len(results) == 1:
            return results["__state__"]
        else:
            return results

    @abstractmethod
    def compute_loss_and_metrics(self,
                                 state: Optional[TensorContextDataset] = None,
                                 pred_state: Optional[TensorContextDataset] = None,
                                 latent_obs: Optional[TensorContextDataset] = None,
                                 pred_latent_obs: Optional[TensorContextDataset] = None,
                                 **kwargs
                                 ) -> dict[str, torch.Tensor]:
        r"""Compute the loss and metrics of the model.

        Args:
            state_context: trajectory of states :math:`(x_t)_{t\\in\\mathbb{T}}` in the context window
            :math:`\\mathbb{T}`.
            pred_state_context: predicted trajectory of states :math:`(\\hat{x}_t)_{t\\in\\mathbb{T}}`
            latent_obs_context: trajectory of latent observables :math:`(z_t)_{t\\in\\mathbb{T}}` in the context window
            :math:`\\mathbb{T}`.
            pred_latent_obs_context: predicted trajectory of latent observables :math:`(\\hat{z}_t)_{t\\in\\mathbb{T}}`
            **kwargs:

        Returns:
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
            filename (path-like or file-like): Save the model to file.
        """
        # self.lightning_module._kooplearn_model_weakref = None  ... Why not simply use self reference?
        pickle_save(self, filename)

    @classmethod
    def load(cls, path: os.PathLike) -> 'LatentBaseModel':
        """Load a serialized model from a file.

        Args:
            filename (path-like or file-like): Load the model from file.

        Returns:
            Saved instance of `LatentBaseModel`.
        """
        restored_obj = pickle_load(cls, path)
        # Restore the weakref # TODO Why?
        # restored_obj.lightning_module._kooplearn_model_weakref = weakref.ref(
        #     restored_obj
        #     )
        return restored_obj

    def _dry_run(self, state: TensorContextDataset):
        class_name = self.__class__.__name__

        assert self.state_features_shape is not None, f"state_features_shape not identified for {class_name}"
        x_t = state

        model_out = self.evolve_forward(x_t)
        try:
            z_t = model_out.pop("latent_obs")
            pred_z_t = model_out.pop("pred_latent_obs")
            pred_x_t = model_out.pop("pred_state")
        except KeyError as e:
            raise KeyError(f"Missing output of {class_name}.evolve_forward") from e

        # Check latent observable contexts shapes
        assert z_t.shape == pred_z_t.shape, \
            f"Encoded latent context shape {z_t.shape} different from input shape {pred_z_t.shape}"
        if pred_x_t is not None:
            assert pred_x_t.shape == x_t.shape, \
                f"Evolved latent context shape {pred_x_t.shape} different from input shape {x_t.shape}"


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
        super(LightningLatentModel, self).__init__()
        self.latent_model: LatentBaseModel = latent_model
        self._optimizer_fn = optimizer_fn
        self._optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        # TODO: Deal with with latent_model hparams if needed.

    def forward(self, state_contexts: TensorContextDataset) -> Any:
        out = self.latent_model.evolve_forward(state_contexts)
        return out

    def training_step(self, train_contexts: TensorContextDataset, batch_idx):
        model_out = self(train_contexts)
        out = self.latent_model.compute_loss_and_metrics(state=train_contexts, **model_out)
        assert "loss" in out, f"Loss not found within {out.keys()}"
        loss = out["loss"]
        self.log_metrics(out, suffix="train", batch_size=len(train_contexts))
        return loss

    def validation_step(self, val_contexts: TensorContextDataset, batch_idx):
        model_out = self(val_contexts)
        out = self.latent_model.compute_loss_and_metrics(state=val_contexts, **model_out)
        assert "loss" in out, f"Loss not found within {out.keys()}"
        loss = out["loss"]
        self.log_metrics(out, suffix="val", batch_size=len(val_contexts))
        return loss

    def test_step(self, test_contexts: TensorContextDataset, batch_idx):
        model_out = self(test_contexts)
        out = self.latent_model.compute_loss_and_metrics(state=test_contexts, **model_out)
        assert "loss" in out, f"Loss not found within {out.keys()}"
        loss = out["loss"]
        self.log_metrics(out, suffix="test", batch_size=len(test_contexts))
        return loss

    def predict_step(self, batch, batch_idx, **kwargs):
        with torch.no_grad():
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

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch.data = batch.data.to(device)
        return batch


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
