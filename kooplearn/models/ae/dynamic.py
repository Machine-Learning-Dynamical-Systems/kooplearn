import logging
import weakref
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
from scipy.linalg import eig

from kooplearn._src.check_deps import check_torch_deps
from kooplearn._src.serialization import pickle_load, pickle_save
from kooplearn._src.utils import ShapeError, check_contexts_shape, check_is_fitted
from kooplearn.abc import BaseModel
from kooplearn.models.ae.utils import (
    decode_contexts,
    encode_contexts,
    evolve_batch,
    evolve_contexts,
)

logger = logging.getLogger("kooplearn")
check_torch_deps()
import lightning  # noqa: E402
import torch  # noqa: E402


class DynamicAE(BaseModel):
    """Dynamic AutoEncoder introduced by :footcite:t:`Lusch2018`. This class also implement the variant introduced by :footcite:t:`Morton2018` in which the linear evolution of the embedded state is given by a least square model.

    Args:
        encoder (torch.nn.Module): Encoder network. Will be initialized as ``encoder(**encoder_kwargs)``.
        decoder (torch.nn.Module): Decoder network. Will be initialized as ``decoder(**decoder_kwargs)``.
        latent_dim (int): Latent dimension. In must match the dimension of the outputs of ``encoder``, as well the dimensions of the inputs of ``decoder``.
        optimizer_fn (torch.optim.Optimizer): Any optimizer from :class:`torch.optim.Optimizer`.
        trainer (lightning.Trainer): An initialized `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_ object used to train the Consistent AutoEncoder.
        loss_weights (dict, optional): Weights of the different loss terms. Should be a dictionary containing either some of all the keys ``rec`` (reconstruction loss), ``pred`` (prediction loss) and ``lin`` (linear evolution loss). Defaults to ``{ "rec": 1.0, "pred": 1.0, "lin": 1.0}`` .
        encoder_kwargs (dict, optional): Dictionary of keyword arguments passed to the encoder network upon initialization. Defaults to ``{}``.
        decoder_kwargs (dict, optional): Dictionary of keyword arguments passed to the decoder network upon initialization. Defaults to ``{}``.
        optimizer_kwargs (dict, optional): Dictionary of keyword arguments passed to the optimizer at initialization. Defaults to ``{}``.
        use_lstsq_for_evolution (bool, optional): Number of steps of backward dynamics to perform. If ``backward_steps == 0``, this model reduces to :class:`kooplearn.models.DynamicAE`. Defaults to False.
        seed (Optional[int], optional): Seed of the internal random number generator. Defaults to None.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_dim: int,
        optimizer_fn: torch.optim.Optimizer,
        trainer: lightning.Trainer,
        loss_weights: dict = {"rec": 1.0, "pred": 1.0, "lin": 1.0},
        encoder_kwargs: dict = {},
        decoder_kwargs: dict = {},
        optimizer_kwargs: dict = {},
        use_lstsq_for_evolution: bool = False,  # If true, implements "Deep Dynamical Modeling and Control of Unsteady Fluid Flows" by Morton et al. (2018)
        seed: Optional[int] = None,
    ):
        lightning.seed_everything(seed)
        self.lightning_trainer = trainer
        self.lightning_module = DynamicAEModule(
            encoder,
            decoder,
            latent_dim,
            optimizer_fn,
            optimizer_kwargs,
            loss_weights=loss_weights,
            encoder_kwargs=encoder_kwargs,
            decoder_kwargs=decoder_kwargs,
            use_lstsq_for_evolution=use_lstsq_for_evolution,
            kooplearn_model_weakref=weakref.ref(self),
        )
        self.seed = seed
        self._is_fitted = False
        # Todo: Add warning on lookback_len for this model

    def fit(
        self,
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule: Optional[lightning.LightningDataModule] = None,
        ckpt_path: Optional[str] = None,
    ):
        """Fits the Koopman AutoEncoder model. Accepts the same arguments as :meth:`lightning.Trainer.fit`, except for the ``model`` keyword, which is automatically set internally.

        Args:
            train_dataloaders: An iterable or collection of iterables specifying training samples.
                Alternatively, a :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.train_dataloader` hook.

            val_dataloaders: An iterable or collection of iterables specifying validation samples.

            datamodule: A :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.train_dataloader` hook.

            ckpt_path: Path/URL of the checkpoint from which training is resumed. Could also be one of two special
                keywords ``"last"`` and ``"hpc"``. If there is no checkpoint file at the path, an exception is raised.
        """
        if isinstance(train_dataloaders, lightning.LightningDataModule):
            datamodule = train_dataloaders
            train_dataloaders = None

        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if (
            train_dataloaders is not None or val_dataloaders is not None
        ) and datamodule is not None:
            raise ValueError(
                "You cannot pass `train_dataloader` or `val_dataloaders` to `VAMPNet.fit(datamodule=...)`"
            )
        # Get the shape of the first batch to determine the lookback_len
        if train_dataloaders is None:
            assert isinstance(datamodule, lightning.LightningDataModule)
            for batch in datamodule.train_dataloader():
                with torch.no_grad():
                    self.lightning_module.dry_run(batch)
                    self._state_trail_dims = tuple(batch.shape[2:])
                break
        else:
            assert isinstance(train_dataloaders, torch.utils.data.DataLoader)
            for batch in train_dataloaders:
                with torch.no_grad():
                    self.lightning_module.dry_run(batch)
                    self._state_trail_dims = tuple(batch.shape[2:])
                break

        self.lightning_trainer.fit(
            model=self.lightning_module,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
        self._is_fitted = True

    def _np_to_torch(self, data: np.ndarray):
        check_contexts_shape(data, self.lookback_len, is_inference_data=True)
        model_device = self.lightning_module.device
        return torch.from_numpy(data.copy()).float().to(model_device)

    def predict(
        self,
        data: np.ndarray,
        t: int = 1,
        observables: Optional[Callable] = None,
    ):
        """
        Predicts the state or, if the system is stochastic, its expected value :math:`\mathbb{E}[X_t | X_0 = X]` after ``t`` instants given the initial conditions ``X = data[:, self.lookback_len:, ...]`` being the lookback slice of ``data``.

        .. attention::

            ``data.shape[1]`` must match the lookback length ``self.lookback_len``. Otherwise, an error is raised.

        If ``observables`` are not ``None``, returns the analogue quantity for the observable instead.

        Args:
            data (numpy.ndarray): Initial conditions to predict. Array of context windows with shape ``(n_init_conditions, self.lookback_len, *self.data_fit.shape[2:])`` (see the note above).
            t (int): Number of steps in the future to predict (returns the last one).
            observables (callable or None): Callable or ``None``. If callable should map batches of states of shape ``(batch, *self.data_fit.shape[2:])`` to batches of observables ``(batch, *obs_features_shape)``.

        Returns:
           The predicted (expected) state/observable at time :math:`t`, shape ``(n_init_conditions, *obs_features_shape)``.
        """
        torch_data = self._np_to_torch(
            data
        )  # [n_samples, context_len == 1, *trail_dims]
        check_is_fitted(self, ["_state_trail_dims"])
        assert tuple(torch_data.shape[2:]) == self._state_trail_dims

        with torch.no_grad():
            init_data = torch_data[:, self.lookback_len - 1, ...]
            evolved_data = evolve_batch(
                init_data,
                t,
                self.lightning_module.encoder,
                self.lightning_module.decoder,
                self.lightning_module.evolution_operator,
            )
            evolved_data = evolved_data.detach().cpu().numpy()
        if observables is None:
            return evolved_data
        elif callable(observables):
            return observables(evolved_data)
        else:
            raise ValueError("Observables must be either None, or callable.")

    def modes(
        self,
        data: np.ndarray,
        observables: Optional[Callable] = None,
    ):
        raise NotImplementedError()

    def eig(
        self,
        eval_left_on: Optional[np.ndarray] = None,
        eval_right_on: Optional[np.ndarray] = None,
    ):
        if hasattr(self, "_eig_cache"):
            w, vl, vr = self._eig_cache
        else:
            if self.lightning_module.hparams.use_lstsq_for_evolution:
                raise NotImplementedError(
                    f"Eigenvalues and eigenvectors are not implemented when {self.lightning_module.hparams.use_lstsq_for_evolution} == True."
                )
            else:
                K = self.lightning_module.evolution_operator
                K_np = K.detach().cpu().numpy()
                w, vl, vr = eig(K_np, left=True, right=True)
                self._eig_cache = w, vl, vr

        if eval_left_on is None and eval_right_on is None:
            # (eigenvalues,)
            return w
        elif eval_left_on is None and eval_right_on is not None:
            # (eigenvalues, right eigenfunctions)
            eval_right_on = self._np_to_torch(
                eval_right_on
            )  # [n_samples, context_len == 1, *trail_dims]
            eval_right_on = eval_right_on[
                :, self.lookback_len - 1, ...
            ]  # [n_samples, *trail_dims]
            with torch.no_grad():
                phi_Xin = self.lightning_module.encoder(eval_right_on)
                r_fns = (
                    phi_Xin @ vl
                )  # Not a typo: I need the left eigenvectors of K to get the right eigenfunctions of the Koopman operator
            return w, r_fns.detach().cpu().numpy()
        elif eval_left_on is not None and eval_right_on is None:
            # (eigenvalues, left eigenfunctions)
            eval_left_on = self._np_to_torch(
                eval_left_on
            )  # [n_samples, context_len == 1, *trail_dims]
            eval_left_on = eval_left_on[
                :, self.lookback_len - 1, ...
            ]  # [n_samples, *trail_dims]
            with torch.no_grad():
                phi_Xin = self.lightning_module.encoder(eval_left_on)
                l_fns = (
                    phi_Xin @ vr
                )  # Not a typo: I need the right eigenvectors of K to get the left eigenfunctions of the Koopman operator
            return w, l_fns.detach().cpu().numpy()
        elif eval_left_on is not None and eval_right_on is not None:
            # (eigenvalues, left eigenfunctions, right eigenfunctions)

            eval_right_on = self._np_to_torch(
                eval_right_on
            )  # [n_samples, context_len == 1, *trail_dims]
            eval_right_on = eval_right_on[
                :, self.lookback_len - 1, ...
            ]  # [n_samples, *trail_dims]

            eval_left_on = self._np_to_torch(
                eval_left_on
            )  # [n_samples, context_len == 1, *trail_dims]
            eval_left_on = eval_left_on[
                :, self.lookback_len - 1, ...
            ]  # [n_samples, *trail_dims]

            with torch.no_grad():
                phi_Xin_r = self.lightning_module.encoder(eval_right_on)
                r_fns = (
                    phi_Xin_r @ vl
                )  # Not a typo: I need the left eigenvectors of K to get the right eigenfunctions of the Koopman operator

                phi_Xin_l = self.lightning_module.encoder(eval_left_on)
                l_fns = (
                    phi_Xin_l @ vr
                )  # Not a typo: I need the right eigenvectors of K to get the left eigenfunctions of the Koopman operator

            return w, l_fns.detach().cpu().numpy(), r_fns.detach().cpu().numpy()

    def save(self, filename):
        """Serialize the model to a file.

        Args:
            filename (path-like or file-like): Save the model to file.
        """
        self.lightning_module._kooplearn_model_weakref = None
        pickle_save(self, filename)

    @classmethod
    def load(cls, filename):
        """Load a serialized model from a file.

        Args:
            filename (path-like or file-like): Load the model from file.

        Returns:
            DynamicAE: The loaded model.
        """
        restored_obj = pickle_load(cls, filename)
        # Restore the weakref
        restored_obj.lightning_module._kooplearn_model_weakref = weakref.ref(
            restored_obj
        )
        return restored_obj

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def lookback_len(self) -> int:
        return 1


class DynamicAEModule(lightning.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_dim: int,
        optimizer_fn: torch.optim.Optimizer,
        optimizer_kwargs: dict,
        loss_weights: dict = {"rec": 1.0, "pred": 1.0, "lin": 1.0},
        encoder_kwargs: dict = {},
        decoder_kwargs: dict = {},
        use_lstsq_for_evolution: bool = False,
        kooplearn_model_weakref: weakref.ReferenceType = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["kooplearn_model_weakref", "optimizer_fn"])
        self.encoder = encoder(**encoder_kwargs)
        self.decoder = decoder(**decoder_kwargs)
        if not self.hparams.use_lstsq_for_evolution:
            self._lin = torch.nn.Linear(latent_dim, latent_dim, bias=False)
            self.evolution_operator = self._lin.weight
        self._optimizer = optimizer_fn
        _tmp_opt_kwargs = deepcopy(optimizer_kwargs)
        if "lr" in _tmp_opt_kwargs:  # For Lightning's LearningRateFinder
            self.lr = _tmp_opt_kwargs.pop("lr")
            self.opt_kwargs = _tmp_opt_kwargs
        else:
            self.lr = 1e-3
            logger.warning(
                "No learning rate specified. Using default value of 1e-3. You can specify the learning rate by passing it to the optimizer_kwargs argument."
            )
        self._kooplearn_model_weakref = kooplearn_model_weakref

    def _lstsq_evolution(self, batch: torch.Tensor):
        X = batch[:, 0, ...]
        Y = batch[:, 1, ...]
        return (torch.linalg.lstsq(X, Y).solution).T

    def configure_optimizers(self):
        kw = self.opt_kwargs | {"lr": self.lr}
        return self._optimizer(self.parameters(), **kw)

    def training_step(self, train_batch, batch_idx):
        lookback_len = self._kooplearn_model_weakref().lookback_len
        encoded_batch = encode_contexts(train_batch, self.encoder)
        if self.hparams.use_lstsq_for_evolution:
            K = self._lstsq_evolution(encoded_batch)
        else:
            K = self.evolution_operator
        evolved_batch = evolve_contexts(encoded_batch, lookback_len, K)
        decoded_batch = decode_contexts(evolved_batch, self.decoder)

        MSE = torch.nn.MSELoss()
        # Reconstruction + prediction loss
        rec_loss = MSE(
            train_batch[:, lookback_len - 1, ...],
            decoded_batch[:, lookback_len - 1, ...],
        )
        pred_loss = MSE(
            train_batch[:, lookback_len:, ...], decoded_batch[:, lookback_len:, ...]
        )

        alpha_rec = self.hparams.loss_weights.get("rec", 1.0)
        alpha_pred = self.hparams.loss_weights.get("pred", 1.0)

        loss = alpha_rec * rec_loss + alpha_pred * pred_loss
        metrics = {
            "train/reconstruction_loss": rec_loss.item(),
            "train/prediction_loss": pred_loss.item(),
        }
        if not self.hparams.use_lstsq_for_evolution:
            # Linear loss
            lin_loss = MSE(encoded_batch, evolved_batch)
            metrics["train/linear_loss"] = lin_loss.item()
            alpha_lin = self.hparams.loss_weights.get("lin", 1.0)
            loss += alpha_lin * lin_loss

        metrics["train/full_loss"] = loss.item()
        self.log_dict(metrics, on_step=True, prog_bar=True, logger=True)
        return loss

    def dry_run(self, batch: torch.Tensor):
        lookback_len = self._kooplearn_model_weakref().lookback_len
        check_contexts_shape(batch, lookback_len)
        # Caution: this method is designed only for internal calling.
        Z = encode_contexts(batch, self.encoder)
        if self.hparams.use_lstsq_for_evolution:
            X = Z[:, 0, ...]
            Y = Z[:, 1, ...]
            evolution_operator = (torch.linalg.lstsq(X, Y).solution).T
        else:
            evolution_operator = self.evolution_operator
        Z_evolved = evolve_contexts(Z, lookback_len, evolution_operator)
        X_evol = decode_contexts(
            Z_evolved, self.decoder
        )  # Should fail if the shape is wrong
        assert Z.shape == Z_evolved.shape

        if batch.shape != X_evol.shape:
            raise ShapeError(
                f"The shape of the evolved states {X_evol.shape[2:]} does not match the shape of the initial states {batch.shape[2:]}. Please check that the decoder networks output tensors of the same shape as the input tensors."
            )
