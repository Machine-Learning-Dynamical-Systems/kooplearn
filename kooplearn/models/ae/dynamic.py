import logging
import math
from math import floor
from typing import Optional, Union

import numpy as np
import scipy

from kooplearn._src.check_deps import check_torch_deps
from kooplearn._src.utils import check_is_fitted
from kooplearn._src.linalg import full_rank_lstsq
from kooplearn.data import TensorContextDataset  # noqa: E402
from kooplearn.models.base_model import LatentBaseModel, LightningLatentModel
from kooplearn.models.ae.utils import flatten_context_data, unflatten_context_data

check_torch_deps()
import lightning  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader


logger = logging.getLogger("kooplearn")


class DynamicAE(LatentBaseModel):
    """Dynamic AutoEncoder introduced by :footcite:t:`Lusch2018`. This class also implement the variant introduced by
    :footcite:t:`Morton2018` in which the linear evolution of the embedded state is given by a least square model.

    Args:
    ----
        encoder (torch.nn.Module): Encoder network. Will be initialized as ``encoder(**encoder_kwargs)``.
        decoder (torch.nn.Module): Decoder network. Will be initialized as ``decoder(**decoder_kwargs)``.
        latent_dim (int): Latent dimension. In must match the dimension of the outputs of ``encoder``, as well the
        dimensions of the inputs of ``decoder``.
        optimizer_fn (torch.optim.Optimizer): Any optimizer from :class:`torch.optim.Optimizer`.
        trainer (lightning.Trainer): An initialized `Lightning Trainer
        <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_ object used to train the Consistent
        AutoEncoder.
        loss_weights (dict, optional): Weights of the different loss terms. Should be a dictionary containing either
        some of all the keys ``rec`` (reconstruction loss), ``pred`` (prediction loss) and ``lin`` (linear evolution
        loss). Defaults to ``{ "rec": 1.0, "pred": 1.0, "lin": 1.0}`` .
        encoder_kwargs (dict, optional): Dictionary of keyword arguments passed to the encoder network upon
        initialization. Defaults to ``{}``.
        decoder_kwargs (dict, optional): Dictionary of keyword arguments passed to the decoder network upon
        initialization. Defaults to ``{}``.
        optimizer_kwargs (dict, optional): Dictionary of keyword arguments passed to the optimizer at initialization.
        Defaults to ``{}``.
        use_lstsq_for_evolution (bool, optional): Number of steps of backward dynamics to perform. If
        ``backward_steps == 0``, this model reduces to :class:`kooplearn.models.DynamicAE`. Defaults to False.
        seed (Optional[int], optional): Seed of the internal random number generator. Defaults to None.
    """

    def __init__(self,
                 encoder: type[torch.nn.Module],
                 decoder: type[torch.nn.Module],
                 encoder_kwargs: dict,
                 decoder_kwargs: dict,
                 latent_dim: int,
                 loss_weights: Optional[dict] = None,
                 use_lstsq_for_evolution: bool = False,
                 evolution_op_bias: bool = False,
                 evolution_op_init_mode: str = "stable"
                 ):
        super().__init__()

        self.encoder = encoder(**encoder_kwargs)
        self.decoder = decoder(**decoder_kwargs)
        self.lin_decoder = None  # Linear decoder for mode decomp. Fitted after learning the latent representation space
        self.latent_dim = latent_dim
        self.evolution_op_bias = evolution_op_bias
        self.loss_weights = loss_weights
        self.use_lstsq_for_evolution = use_lstsq_for_evolution

        if use_lstsq_for_evolution:
            raise NotImplementedError("use_lstsq_for_evolution = True is not implemented/tested yet.")
        else:
            # TODO: Should we add bias to hardcode the constant function modeling the evolution operator?
            # Adding a bias term is analog to enforcing the evolution operator to model the constant function.
            self.linear_dynamics = torch.nn.Linear(latent_dim, latent_dim, bias=evolution_op_bias)
            self.initialize_evolution_operator(init_mode=evolution_op_init_mode)

        self.state_features_shape = None  # Shape of the state/input features
        # Lightning variables populated during fit
        self.trainer = None
        self.lightning_model = None
        # self._is_fitted = False  # Automatically determined by looking at self.trainer.status.

    def fit(
            self,
            train_dataloaders: Optional[Union[DataLoader, list[DataLoader]]] = None,
            val_dataloaders: Optional[Union[DataLoader, list[DataLoader]]] = None,
            datamodule: Optional[lightning.LightningDataModule] = None,
            trainer: Optional[lightning.Trainer] = None,
            optimizer_fn: Optional[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[dict] = None,
            ckpt_path: Optional[str] = None,
            ):
        """Fits the Koopman AutoEncoder model. Accepts the same arguments as :meth:`lightning.Trainer.fit`,
        except for the ``model`` keyword, which is automatically set internally.

        Args:
        ----
            train_dataloaders: An iterable or collection of iterables specifying training samples.
                Alternatively, a :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.train_dataloader` hook.

            val_dataloaders: An iterable or collection of iterables specifying validation samples.

            datamodule: A :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.train_dataloader` hook.

            ckpt_path: Path/URL of the checkpoint from which training is resumed. Could also be one of two special
                keywords ``"last"`` and ``"hpc"``. If there is no checkpoint file at the path, an exception is raised.
        """
        self.trainer = _default_lighting_trainer() if trainer is None else trainer
        assert isinstance(self.trainer, lightning.Trainer)

        lightning_module = LightningLatentModel(
            latent_model=self,
            optimizer_fn=optimizer_fn,
            optimizer_kwargs=optimizer_kwargs,
            )

        self._check_dataloaders_and_shapes(datamodule, train_dataloaders, val_dataloaders)

        if self.loss_weights is None:  # Automatically set the loss weights if not provided
            # Weight evenly a delta error in state space as a delta error in latent observable space
            # A λ error in each dimension of the state/observable space leads to a mean square error:
            # err_state = λ * sqrt(input_dim)
            # err_latent = λ * sqrt(latent_dim)
            # To balance the errors, we need to weight the latent error by the ratio sqrt(input_dim / latent_dim)
            input_dim = np.prod(self.state_features_shape)
            obs_dim_state_dim_ratio = math.sqrt(input_dim / self.latent_dim)
            self.loss_weights = dict(rec=1.0, pred=1.0, lin=1.0 * obs_dim_state_dim_ratio)

        # Fit the encoder, decoder and (optionally) the evolution operator =============================================
        self.trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            )

        train_dataset = train_dataloaders.dataset if train_dataloaders is not None else datamodule.train_dataset
        train_dataloader = train_dataloaders if train_dataloaders is not None else datamodule.train_dataloader()
        # Compute the spectral decomposition of the learned evolution operator =========================================
        sample_batch = next(iter(train_dataloader))
        eigvals = self.eig(eval_left_on=sample_batch)

        # Fit linear decoder to perform dynamic mode decomposition =====================================================
        # Denote by φ(x) an eigenfunction of the evolution operator T = VΛV⁻¹, where V is the matrix of eigenvectors
        # and Λ is the diagonal matrix of eigenvalues. The vector of eigenfunctions is computed as
        # `φ(x_t)=V⁻¹z_t = V⁻¹Ψ(x_t)`. Where Ψ:X→Z is the encoder/obs_function. If we want to do mode decomposition
        # we have that z_t+1 = Σ_i V_[i,:] @ (λ_i · φ_i(x_t)). If we apply a non-linear decoder, we loose the
        # decomposition in modes (Σ_i) as Ψ⁻¹(z_1,t + z_2,t) != Ψ⁻¹(z_1,t) + Ψ⁻¹(z_2,t). Therefore, after learning
        # the representation space Z, we fit a linear decoder `Ψ⁻¹_lin : Z → X` to perform mode decomposition by
        # x_t+1 = Ψ⁻¹_lin(Σ_i V_[i,:] @ (λ_i · φ_i(x_t))).
        # Use the best parameters to compute latent observables to train linear decoder
        # Lightning sends outputs to CPU by default and will use automatically the best model from the training
        predict_out = trainer.predict(dataloaders=train_dataloader,
                                      ckpt_path='best')
        Z = [out_batch['latent_obs'] for out_batch in predict_out]
        Z = TensorContextDataset(torch.cat([z.data for z in Z], dim=0))         # (n_samples, context_len, latent_dim)
        Z_flat = flatten_context_data(Z)                                        # (n_samples * context_len, latent_dim)
        X_flat = flatten_context_data(train_dataset).to(device=Z_flat.device)   # (n_samples * context_len, state_dim)

        lin_decoder, _ = full_rank_lstsq(X=Z_flat.T, Y=X_flat.T, bias=False)
        _expected_shape = (np.prod(self.state_features_shape), self.latent_dim)
        assert lin_decoder.shape == _expected_shape, \
            f"Expected linear decoder shape {_expected_shape}, got {lin_decoder.shape}"
        # Define linear_decoder=`Ψ⁻¹_lin` as a non-trainable model Parameter.
        self.lin_decoder = torch.nn.Parameter(lin_decoder, requires_grad=False)

        # Use the learned encoder to compute the final fitted evolution operator
        if self.use_lstsq_for_evolution:
            raise NotImplementedError("use_lstsq_for_evolution = True is not implemented/tested yet.")
            # train_dataset = self._to_torch(train_dataloaders.dataset)
            # encoded_train = encode_contexts(train_dataset, self.lightning_module.encoder)
            # self.lightning_module.evolution_operator = self.lightning_module._lstsq_evolution(encoded_train)

        return lightning_module

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
        ----
            data (TensorContextDataset): Dataset of context windows. The lookback window of ``data`` will be used as
            the initial condition, see the note above.
            t (int): Number of steps in the future to predict (returns the last one).
            predict_observables (bool): Return the prediction for the observables in ``data.observables``,
            if present. Defaults to ``True``.
            reencode_every (int): When ``t > 1``, periodically reencode the predictions as described in
            :footcite:t:`Fathi2023`. Only available when ``predict_observables = False``.

        Returns:
        -------
           The predicted (expected) state/observable at time :math:`t`. The result is composed of arrays with shape
           matching ``data.lookforward(self.lookback_len)`` or the contents of ``data.observables``. If
           ``predict_observables = True`` and ``data.observables != None``, the returned ``dict``will contain the
           special key ``__state__`` containing the prediction for the state as well.
        """
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

    def modes(self,
              state: TensorContextDataset,
              predict_observables: bool = False,
              ):
        r"""Compute the modes of the state and/or observables"""

        if predict_observables:
            raise NotImplementedError("Need to implement the approximation of the outer product / inner products "
                                      "between each dimension/function of the latent space and observables provided")

        # Check the provided data has
        eigvals = self.eig()



        raise NotImplementedError()

    def compute_loss_and_metrics(self,
                                 state: TensorContextDataset,
                                 pred_state: TensorContextDataset,
                                 latent_obs: TensorContextDataset,
                                 pred_latent_obs: TensorContextDataset,
                                 ) -> dict[str, torch.Tensor]:
        r"""Compute the Dynamics Autoencoder loss and metrics.

        TODO: Add docstring definition of the loss terms.

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
        lookback_len = self.lookback_len
        alpha_rec = self.loss_weights["rec"]
        alpha_pred = self.loss_weights["pred"]
        alpha_lin = self.loss_weights["lin"]

        MSE = torch.nn.MSELoss()
        # Reconstruction + prediction loss
        rec_loss = MSE(state.lookback(lookback_len), pred_state.lookback(lookback_len))
        pred_loss = MSE(state.lookforward(lookback_len), pred_state.lookforward(lookback_len))

        metrics = dict(reconstruction_loss=rec_loss.item(), prediction_loss=pred_loss.item())

        loss = alpha_rec * rec_loss + alpha_pred * pred_loss

        if self.use_lstsq_for_evolution:
            pass
        else:
            lin_loss = MSE(latent_obs.data, pred_latent_obs.data)
            loss += alpha_lin * lin_loss

            metrics["linear_loss"] = lin_loss.item()

        metrics["loss"] = loss
        return metrics

    def evolution_operator(self):
        if self.use_lstsq_for_evolution:
            raise NotImplementedError("use_lstsq_for_evolution = True is not implemented/tested yet.")
        else:
            return self.linear_dynamics.weight

    @property
    def lookback_len(self) -> int:
        return 1

    @property
    def is_fitted(self):
        if self.trainer is None:
            return False
        else:
            return self.trainer.state.status == "finished"

    def initialize_evolution_operator(self, init_mode: str):
        """Initializes the evolution operator.

        Orthogonal or random initialization of the evolution operator's parameters results in brittle performance of AE
        architectures with trainable evolution operators. This method implements different initialization strategies
        for the evolution operator.

        Args:
            init_mode (str): Either ("stable", ...). If "stable", the evolution operator is initialized as the identity
        Returns:
            None
        """
        if self.use_lstsq_for_evolution:
            raise NotImplementedError("use_lstsq_for_evolution = True is not implemented/tested yet.")
        else:
            if init_mode == "stable":
                self.linear_dynamics.weight.data = torch.eye(self.latent_dim)
                if self.evolution_op_bias:
                    self.transfer_op.bias.data = torch.zeros(self.latent_dim)
            else:
                logger.warning(f"Eival init mode {init_mode} not implemented")
                return
            logger.info(f"Trainable evolution operator initialized with mode {init_mode}")

    def _check_dataloaders_and_shapes(self, datamodule, train_dataloaders, val_dataloaders):
        """Check dataloaders and the shape of the first batch to determine state features shape."""
        # TODO: Should we add a features_shape attribute to the Context class, and avoid all this?

        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        # TODO: Lightning has checks this before each train starts. Is this necessary?
        # if (train_dataloaders is not None or val_dataloaders is not None) and datamodule is not None:
        #     raise ValueError(
        #         f"You cannot pass `train_dataloader` or `val_dataloaders` to `{self.__class__.__name__}.fit(
        #         datamodule=...)`")

        # Get the shape of the first batch to determine the lookback_len
        # TODO: lookback_len is model-fixed and all this process is unnecessary if we know the `state_obs_shape`
        if train_dataloaders is None:
            assert isinstance(datamodule, lightning.LightningDataModule)
            for batch in datamodule.train_dataloader():
                assert isinstance(batch, TensorContextDataset)
                with torch.no_grad():
                    self.state_features_shape = tuple(batch.shape[2:])
                    self._dry_run(batch)
                break
        else:
            assert isinstance(train_dataloaders, torch.utils.data.DataLoader)
            for batch in train_dataloaders:
                assert isinstance(batch, TensorContextDataset)
                with torch.no_grad():
                    self.state_features_shape = tuple(batch.shape[2:])
                    self._dry_run(batch)
                break
        # Get the shape of the first batch to determine the lookback_len and state features shape
        if train_dataloaders is None:
            assert isinstance(datamodule, lightning.LightningDataModule)
            for batch in datamodule.train_dataloader():
                assert isinstance(batch, TensorContextDataset)
                with torch.no_grad():
                    self._state_observables_shape = tuple(batch.shape[2:])
                    self._dry_run(batch)
                break
        else:
            assert isinstance(train_dataloaders, torch.utils.data.DataLoader)
            for batch in train_dataloaders:
                assert isinstance(batch, TensorContextDataset)
                with torch.no_grad():
                    self._state_observables_shape = tuple(batch.shape[2:])
                    self._dry_run(batch)
                break

    def _to_torch(self, data: TensorContextDataset):
        # check_contexts_shape(data, self.lookback_len, is_inference_data=True)
        model_device = self.lightning_module.device
        return TensorContextDataset(data.data, backend="torch", device=model_device)

    def _preprocess_for_eigfun(self, data: TensorContextDataset):
        data_ondevice = self._to_torch(data)
        contexts_for_inference = data_ondevice.slice(
            slice(self.lookback_len - 1, self.lookback_len)
            )
        return contexts_for_inference.view(len(data_ondevice), *data_ondevice.shape[2:])



def _default_lighting_trainer() -> lightning.Trainer:
    return lightning.Trainer(accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                             devices='auto',
                             logger=None,
                             log_every_n_steps=1,
                             max_epochs=100,
                             enable_progress_bar=True)


def _default_lighting_module() -> type[LightningLatentModel]:
    return LightningLatentModel

# class DynamicAEModule(lightning.LightningModule):
#     def __init__(
#             self,
#             encoder: torch.nn.Module,
#             decoder: torch.nn.Module,
#             latent_dim: int,
#             optimizer_fn: torch.optim.Optimizer,
#             optimizer_kwargs: dict,
#             loss_weights: dict = {"rec": 1.0, "pred": 1.0, "lin": 1.0},
#             encoder_kwargs: dict = {},
#             decoder_kwargs: dict = {},
#             use_lstsq_for_evolution: bool = False,
#             kooplearn_model_weakref: weakref.ReferenceType = None,
#             ):
#         super().__init__()
#         self.save_hyperparameters(ignore=["kooplearn_model_weakref", "optimizer_fn"])
#         self.encoder = encoder(**encoder_kwargs)
#         self.decoder = decoder(**decoder_kwargs)
#         if not self.hparams.use_lstsq_for_evolution:
#             self._lin = torch.nn.Linear(latent_dim, latent_dim, bias=False)
#             self.evolution_operator = self._lin.weight
#         self._optimizer = optimizer_fn
#         _tmp_opt_kwargs = deepcopy(optimizer_kwargs)
#         if "lr" in _tmp_opt_kwargs:  # For Lightning's LearningRateFinder
#             self.lr = _tmp_opt_kwargs.pop("lr")
#             self.opt_kwargs = _tmp_opt_kwargs
#         else:
#             self.lr = 1e-3
#             logger.warning(
#                 "No learning rate specified. Using default value of 1e-3. You can specify the learning rate by "
#                 "passing it to the optimizer_kwargs argument."
#                 )
#         self._kooplearn_model_weakref = kooplearn_model_weakref
#
#     def _lstsq_evolution(self, batch: TensorContextDataset):
#         X = batch.data[:, 0, ...]
#         Y = batch.data[:, 1, ...]
#         return (torch.linalg.lstsq(X, Y).solution).T
#
#     def configure_optimizers(self):
#         kw = self.opt_kwargs | {"lr": self.lr}
#         return self._optimizer(self.parameters(), **kw)
#
#     def training_step(self, train_batch, batch_idx):
#         lookback_len = self._kooplearn_model_weakref().lookback_len
#         encoded_batch = encode_contexts(train_batch, self.encoder)
#         if self.hparams.use_lstsq_for_evolution:
#             K = self._lstsq_evolution(encoded_batch)
#         else:
#             K = self.evolution_operator
#         evolved_batch = evolve_contexts(encoded_batch, lookback_len, K)
#         decoded_batch = decode_contexts(evolved_batch, self.decoder)
#
#         MSE = torch.nn.MSELoss()
#         # Reconstruction + prediction loss
#         rec_loss = MSE(train_batch.lookback(lookback_len), decoded_batch.lookback(lookback_len))
#         pred_loss = MSE(train_batch.lookforward(lookback_len), decoded_batch.lookforward(lookback_len))
#
#         alpha_rec = self.hparams.loss_weights.get("rec", 1.0)
#         alpha_pred = self.hparams.loss_weights.get("pred", 1.0)
#
#         loss = alpha_rec * rec_loss + alpha_pred * pred_loss
#         metrics = {
#             "train/reconstruction_loss": rec_loss.item(),
#             "train/prediction_loss":     pred_loss.item(),
#             }
#         if not self.hparams.use_lstsq_for_evolution:
#             # Linear loss
#             lin_loss = MSE(encoded_batch.data, evolved_batch.data)
#             metrics["train/linear_loss"] = lin_loss.item()
#             alpha_lin = self.hparams.loss_weights.get("lin", 1.0)
#             loss += alpha_lin * lin_loss
#
#         metrics["train/full_loss"] = loss.item()
#         self.log_dict(metrics, on_step=True, prog_bar=True, logger=True)
#         return loss
#
#     def transfer_batch_to_device(self, batch, device, dataloader_idx):
#         batch.data = batch.data.to(device)
#         return batch
#
#     def dry_run(self, batch: TensorContextDataset):
#         lookback_len = self._kooplearn_model_weakref().lookback_len
#         # Caution: this method is designed only for internal calling.
#         Z = encode_contexts(batch, self.encoder)
#         if self.hparams.use_lstsq_for_evolution:
#             evolution_operator = self._lstsq_evolution(Z)
#         else:
#             evolution_operator = self.evolution_operator
#         Z_evolved = evolve_contexts(Z, lookback_len, evolution_operator)
#         X_evol = decode_contexts(
#             Z_evolved, self.decoder
#             )  # Should fail if the shape is wrong
#         assert Z.shape == Z_evolved.shape
#
#         if batch.shape != X_evol.shape:
#             raise ShapeError(
#                 f"The shape of the evolved states {X_evol.shape[2:]} does not match the shape of the initial states "
#                 f"{batch.shape[2:]}. Please check that the decoder networks output tensors of the same shape as the "
#                 f"input tensors."
#                 )
