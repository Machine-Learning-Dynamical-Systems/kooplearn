import logging
import math
from functools import wraps
from typing import Optional, Union

import numpy as np
import scipy

from kooplearn._src.check_deps import check_torch_deps
from kooplearn._src.linalg import full_rank_lstsq
from kooplearn.data import TensorContextDataset  # noqa: E402
from kooplearn.models.ae.utils import flatten_context_data
from kooplearn.models.base_model import LatentBaseModel, LightningLatentModel
from kooplearn.utils import ModesInfo, check_if_resume_experiment

check_torch_deps()
import lightning  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DynamicAE(LatentBaseModel):
    """ Dynamic AutoEncoder introduced by :footcite:t:`Lusch2018`. This class also implements the variant introduced by
    :footcite:t:`Morton2018` in which the linear evolution of the embedded state is given by a least square model.

    Args:
       encoder (torch.nn.Module): Encoder network. Will be initialized as ``encoder(**encoder_kwargs)``.
       decoder (torch.nn.Module): Decoder network. Will be initialized as ``decoder(**decoder_kwargs)``.
       latent_dim (int): Latent dimension. It must match the dimension of the outputs of ``encoder``, as well as the
        dimensions of the inputs of ``decoder``.
       loss_weights (dict, optional): Weights of the different loss terms. Should be a dictionary containing either
        some or all the keys ``rec`` (reconstruction loss), ``pred`` (prediction loss) and ``lin`` (linear evolution
        loss). Defaults to ``{ "rec": 1.0, "pred": 1.0, "lin": 1.0}``.
       use_lstsq_for_evolution (bool, optional): If True, the linear evolution of the embedded state is given by a
       least square model. If ``use_lstsq_for_evolution == False``, this model reduces to
        :class:`kooplearn.models.DynamicAE`. Defaults to False.
       evolution_op_bias (bool, optional): If True, adds a bias term to the evolution operator. Adding a bias term is
        analogous to enforcing the evolution operator to model the constant function. Defaults to False.
       evolution_op_init_mode (str, optional): The initialization mode for the evolution operator. Defaults to "stable".
       encoder_kwargs (dict, optional): Dictionary of keyword arguments passed to the encoder network upon
        initialization. Defaults to ``{}``.
       decoder_kwargs (dict, optional): Dictionary of keyword arguments passed to the decoder network upon
        initialization. Defaults to ``{}``.

    Attributes:
       encoder (torch.nn.Module): The encoder network.
       decoder (torch.nn.Module): The decoder network.
       lin_decoder (torch.nn.Module or None): Linear decoder for mode decomposition. Fitted after learning the latent
        representation space. Defaults to None.
       latent_dim (int): The dimension of the latent space.
       evolution_op_bias (bool): Whether to add a bias term to the evolution operator.
       loss_weights (dict): Weights of the different loss terms.
       use_lstsq_for_evolution (bool): Whether to use a least square model for the linear evolution of the embedded
        state.
       linear_dynamics (torch.nn.Linear): The linear dynamics of the model.
       state_features_shape (tuple or None): The shape of the state/input features. Defaults to None.
       trainer (lightning.Trainer or None): The Lightning Trainer used to train the model. Defaults to None.
       lightning_model (LightningLatentModel or None): The Lightning model for training. Defaults to None.
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
        super().__init__()  # Initialize torch.nn.Module

        self.encoder = encoder(**encoder_kwargs)
        self.decoder = decoder(**decoder_kwargs)
        self.lin_decoder = None  # Linear decoder for mode decomp. Fitted after learning the latent representation space
        self.latent_dim = latent_dim
        self.evolution_op_bias = evolution_op_bias
        self.loss_weights = loss_weights
        self.use_lstsq_for_evolution = use_lstsq_for_evolution

        if use_lstsq_for_evolution:
            # self.linear_dynamics = torch.nn.Parameter(torch., requires_grad=False)
            raise NotImplementedError("use_lstsq_for_evolution = True is not implemented/tested yet.")
        else:
            # TODO: Should we add bias to hardcode the constant function modeling the evolution operator?
            # Adding a bias term is analog to enforcing the evolution operator to model the constant function.
            self.linear_dynamics = self.get_linear_dynamics_model(enforce_constant_fn=evolution_op_bias)
            self.initialize_evolution_operator(init_mode=evolution_op_init_mode)

        self.state_features_shape = None  # Shape of the state/input features
        # Lightning variables populated during fit
        self.trainer = None
        self.lightning_model = None
        self._is_fitted = False  # TODO: Automatically determined by looking at self.trainer.status.

    def fit(
            self,
            train_dataloaders: Optional[Union[DataLoader, list[DataLoader]]] = None,
            val_dataloaders: Optional[Union[DataLoader, list[DataLoader]]] = None,
            datamodule: Optional[lightning.LightningDataModule] = None,
            trainer: Optional[lightning.Trainer] = None,
            optimizer_fn: torch.optim.Optimizer = torch.optim.Adam,
            optimizer_kwargs: Optional[dict] = None,
            ):
        """Fits the Koopman AutoEncoder model. Accepts the same arguments as :meth:`lightning.Trainer.fit`,
        except for the ``model`` keyword, which is automatically set internally.

        Args:
            train_dataloaders: An iterable or collection of iterables specifying training samples.
                Alternatively, a :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.train_dataloader` hook.

            val_dataloaders: An iterable or collection of iterables specifying validation samples.

            datamodule: A :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.train_dataloader` hook.
        """
        self._is_fitted = False
        self.trainer = _default_lighting_trainer() if trainer is None else trainer
        assert isinstance(self.trainer, lightning.Trainer)

        # TODO: if user wants custom LightningModule, we should allow them to pass it.
        #   Making this module a class attribute leads to the problem of recursive parameter search between LatentVar
        #
        lightning_module = LightningLatentModel(
            latent_model=self,
            optimizer_fn=optimizer_fn,
            optimizer_kwargs=optimizer_kwargs,
            )

        self._check_dataloaders_and_shapes(datamodule, train_dataloaders, val_dataloaders)

        if self.loss_weights is None:  # Automatically set the loss weights if not provided
            self.loss_weights = dict(rec=1.0, pred=1.0, lin=2.0)

        # Weight evenly a delta error in state space as a delta error in latent observable space
        # A λ error in each dimension of the state/observable space leads to a mean square error:
        # err_state = λ * sqrt(input_dim)       ---             err_latent = λ * sqrt(latent_dim)
        # To balance the errors, we need to weight the latent error by the ratio sqrt(input_dim / latent_dim)
        input_dim = np.prod(self.state_features_shape)
        obs_dim_state_dim_ratio = math.sqrt(input_dim / self.latent_dim)
        self.loss_weights["lin"] *= obs_dim_state_dim_ratio

        # Fit the encoder, decoder and (optionally) the evolution operator =============================================
        ckpt_call = self.trainer.checkpoint_callback  # Get trainer checkpoint callback if any
        training_done, ckpt_path, best_path = check_if_resume_experiment(ckpt_call)
        # If trained model was saved, skip training and load trained model.
        if training_done:  # Load best model (according to validation loss)
            best_ckpt = torch.load(best_path)
            lightning_module.eval()
            lightning_module.load_state_dict(best_ckpt['state_dict'], strict=False)
            # from lightning.pytorch.trainer.states import TrainerStatus
            # self.trainer.state.status = TrainerStatus.FINISHED
            self._is_fitted = True
            logger.info(f"Skipping training. Loaded best model from {best_path}")
        else:
            self.trainer.fit(
                model=lightning_module,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
                )
            self._is_fitted = self.trainer.state.finished

        logger.info(f"Fitting linear decoder for mode decomposition")
        # TODO: remove from here, this seems to be experiment specific.
        if datamodule is not None and hasattr(datamodule, "augment"):
            datamodule.augment = False

        # Get latent observables of training dataset
        train_dataset = train_dataloaders.dataset if train_dataloaders is not None else datamodule.train_dataset
        train_dataloader = train_dataloaders if train_dataloaders is not None else datamodule.train_dataloader()
        predict_out = trainer.predict(model=lightning_module, dataloaders=train_dataloader, ckpt_path=best_path)
        Z = [out_batch['latent_obs'] for out_batch in predict_out]
        Z = TensorContextDataset(torch.cat([z.data for z in Z], dim=0))  # (n_samples, context_len, latent_dim)

        # # Fit the evolution operator using least squares =============================================================
        # Z_0 = Z.data[:, 0, :]
        # Z_1 = Z.data[:, 1, :]
        # evol_op, _ = full_rank_lstsq(X=Z_0.T, Y=Z_1.T, bias=False)
        # # Reset the evolution operator to the fitted one
        # self.linear_dynamics.weight.data = evol_op
        # print("\n ------------------- \n Fitted evolution operator: \n")

        # Fit linear decoder to perform dynamic mode decomposition =====================================================
        # TODO: add to checkpoint after training. Avoid recomputing this if already fitted.
        # Denote by φ(x) an eigenfunction of the evolution operator T = VΛV⁻¹, where V is the matrix of eigenvectors
        # and Λ is the diagonal matrix of eigenvalues. The vector of eigenfunctions is computed as
        # `Ψ(x_t)=V⁻¹z_t = V⁻¹Ψ(x_t)`. Where Ψ:X→Z is the encoder/obs_function. If we want to do mode decomposition
        # we have that z_t+1 = Σ_i V_[i,:] @ (λ_i · φ_i(x_t)). Because the learned decoder is non-linear we cannot
        # transfer the mode decomposition to the state-space as Ψ⁻¹(z_1,t + z_2,t) != Ψ⁻¹(z_1,t) + Ψ⁻¹(z_2,t).
        # Thus, after learning the representation space Z, we fit a *linear* decoder `Ψ⁻¹_lin : Z → X` to perform mode
        # decomposition by x_t+1 = Ψ⁻¹_lin(Σ_i V_[i,:] @ (λ_i · Ψ_i(x_t))).

        Z_flat = flatten_context_data(Z)  # (n_samples * context_len, latent_dim)
        X_flat = flatten_context_data(train_dataset).to(device=Z_flat.device)  # (n_samples * context_len, state_dim)

        # Store the linear_decoder=`Ψ⁻¹_lin: Z -> X` as a non-trainable `torch.nn.Linear` Module.
        self.lin_decoder = self.fit_linear_decoder(states=X_flat.T, latent_states=Z_flat.T)

        # TODO: update best model checkpoint to include the linear decoder and eigdecomposition cache.
        return lightning_module

    @wraps(LatentBaseModel.evolve_forward)
    def evolve_forward(self, state: TensorContextDataset) -> dict:
        # Evolve using LatentBaseModel implementation
        output = super().evolve_forward(state)

        # After fitted, additionally compute the predictions using the linear decoder.
        if self.is_fitted and self.lin_decoder is not None:
            pred_latent_obs = output["pred_latent_obs"]
            pred_state_lin_decoded = self.decode_contexts(latent_obs=pred_latent_obs, decoder=self.lin_decoder)
            output["pred_state_lin_decoded"] = pred_state_lin_decoded

        return output

    def modes(self,
              state: TensorContextDataset,
              predict_observables: bool = False,
              ):
        r"""Compute the mode decomposition of the state and/or observables

        Informally, if :math:`(\\lambda_i, \\xi_i, \\psi_i)_{i = 1}^{r}` are eigentriplets of the Koopman/Transfer
        operator, for any observable :math:`f` the i-th mode of :math:`f` at :math:`x` is defined as:
        :math:`\\lambda_i \\langle \\xi_i, f \\rangle \\psi_i(x)`. See :footcite:t:`Kostic2022` for more details.

        When applying mode decomposition to the latent observable states :math:`\mathbf{z}_t \in \mathbb{R}^l`,
        the modes are obtained using the eigenvalues and eigenvectors of the solution operator :math:`T`, as there is no
        need to approximate the inner product :math:`\\langle \\xi_i, f \\rangle` using a kernel/Data matrix.
        Consider that the evolution of the latent observable is given by:

        .. math::
        \mathbf{z}_{t+k} = T^k \mathbf{z}_t = (V \Lambda^k V^H) \mathbf{z}_t = \sum_{i=1}^{l} \lambda_i^k \langle
        v_i, \mathbf{z}_t \rangle v_i


        dd
        Note: when we compute the modes of the latent observable state, the mode decomposition reduces to a traditional
        mode decomposition using the eigenvectors and eigenvalues of the solution operator :math:`T`

        """

        assert self.is_fitted, \
            f"Instance of {self.__class__.__name__} is not fitted. Please call the `fit` method before calling `modes`"
        if predict_observables:
            raise NotImplementedError("Need to implement the approximation of the outer product / inner products "
                                      "between each dimension/function of the latent space and observables provided")

        eigvals = self.eig()

        # Encode # TODO: Should use the eig(eval_left_on=state), but fuck, that interface seems super confusing.
        state.to(device=self.evolution_operator.device, dtype=self.evolution_operator.dtype)
        # Compute the latent observables for the data (batch, context_len, latent_dim)
        z_t = self.encode_contexts(state=state, encoder=self.encoder).data.detach().cpu().numpy()

        # Left eigenfunctions are the inner products between the latent observables z_t and the right eigenvectors
        # `v_i` of the evolution operator T @ v_i = λ_i v_i. That is: eigfns[...,t, i] = <v_i, z_t> : i=1,...,l
        # Thus we have that z_t+dt = Σ_i[v_i * λ_i * <v_i, z_t>], where <v_i, z_t> is the projection to eigenfunction i.
        # v_i = eigvecs_r[:, i] = V[:,i], where T = V Λ V^-1
        eigvecs_r, eigvecs_r_inv = self._eigvecs_r.detach().cpu().numpy(), self._eigvecs_r_inv.detach().cpu().numpy()

        # We want to compute each z_t+dt_i = v_i * λ_i * <v_i, z_t>.
        # First we project the encoded latent obs state to the eigenbasis: z_t_eig = V^-1 z_t := [<v_1, z_t>, ...]
        z_t_eig = np.einsum("...il,...l->...i", np.linalg.inv(eigvecs_r), z_t)
        # # Evolve the eigenspaces z_t+dt_eig = Λ (V^-1 z_t)  :  z_t+dt_eig_i = λ_i * (V^-1 z_t)_i
        # z_t_dt_eig = z_t_eig # np.einsum("...l,l->...l", z_t_eig, eigvals)
        # # For each evolved eigenfunction λ_i * <v_i, z_t>, project back to the canonical basis, such that
        # # z_t+dt_i = v_i * λ_i * (V^-1 z_t)_i = v_i * λ_i * <v_i, z_t>.
        # z_t_dt_modes = np.einsum("...le,...e->...el", eigvecs_r, z_t_dt_eig)  # (..., mode_idx, latent_dim)

        modes_info = ModesInfo(dt=1.0,
                               eigvals=eigvals,
                               eigvecs_r=eigvecs_r,
                               state_eigenbasis=z_t_eig,
                               linear_decoder=self.lin_decoder
                               )
        # modes = modes_info.modes

        # Sanity check when linear_
        # z_t_rec = np.sum(modes, axis=-2)
        # assert np.allclose(z_t_rec - z_t, 0, atol=1e-5, rtol=1e-5), \
        #     f" {np.max(np.abs(z_t_rec - z_t))} : Modes do not reconstruct the latent observables."

        return modes_info

    def compute_loss_and_metrics(self,
                                 state: TensorContextDataset = None,
                                 pred_state: TensorContextDataset = None,
                                 latent_obs: TensorContextDataset = None,
                                 pred_latent_obs: TensorContextDataset = None,
                                 pred_state_lin_decoded: TensorContextDataset = None,
                                 # **kwargs
                                 ) -> dict[str, torch.Tensor]:
        """ Computes the loss and metrics for the DynamicAE model.

        The loss function is defined as the mean squared error between the predicted and actual states and latent
        observables.
        Specifically, the loss for a single "sample" is given by:

        .. math::
            L = \frac{1}{T} \sum_{t=1}^{T} ((\mathbf{x}_t - \hat{\mathbf{x}}_t)^2 + \alpha_{lin} (\mathbf{z}_t -
            \hat{\mathbf{z}}_t)^2)

        where:
        - :math:`\mathbf{x}_t` and :math:`\mathbf{z}_t` are the actual state and latent observables at time `t`
        respectively,
        - :math:`\hat{\mathbf{x}}_t` and :math:`\hat{\mathbf{z}}_t` are the predicted state and latent observables at
        time `t` respectively,
        - :math:`T` is the total number of time steps in the context window.

        Args:
            state (TensorContextDataset): The actual state to be encoded. This should be a trajectory of states
                :math:`(\mathbf{x}_t)_{t\\in\\mathbb{T}}` in the context window :math:`\\mathbb{T}`.
            pred_state (TensorContextDataset): The predicted state. This should be a trajectory of states
                :math:`(\hat{\mathbf{x}}_t)_{t\\in\\mathbb{T}}` in the context window :math:`\\mathbb{T}`.
            latent_obs (TensorContextDataset): The actual latent observables. This should be a trajectory of latent
            observables
                :math:`(\mathbf{z}_t)_{t\\in\\mathbb{T}}` in the context window :math:`\\mathbb{T}`.
            pred_latent_obs (TensorContextDataset): The predicted latent observables. This should be a trajectory of
            latent observables
                :math:`(\hat{\mathbf{z}}_t)_{t\\in\\mathbb{T}}` in the context window :math:`\\mathbb{T}`.
            pred_state_lin_decoded:
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the computed loss and metrics.
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

        if pred_state_lin_decoded is not None:  # If predictions are computed with linear decoder compute metrics.
            rec_lin_loss = MSE(state.lookback(lookback_len), pred_state_lin_decoded.lookback(lookback_len))
            pred_lin_loss = MSE(state.lookforward(lookback_len), pred_state_lin_decoded.lookforward(lookback_len))
            metrics["reconstruction_lin_dec_loss"] = rec_lin_loss.item()
            metrics["prediction_lin_dec_loss"] = pred_lin_loss.item()

        loss = alpha_rec * rec_loss + alpha_pred * pred_loss

        if self.use_lstsq_for_evolution:
            pass
        else:
            lin_loss = MSE(latent_obs.data, pred_latent_obs.data)
            loss += alpha_lin * lin_loss

            metrics["linear_dynamics_loss"] = lin_loss.item()

        metrics["loss"] = loss
        return metrics

    def get_linear_dynamics_model(self, enforce_constant_fn: bool = False) -> torch.nn.Module:
        if enforce_constant_fn:
            raise NotImplementedError("Need to handle bias term in evolution and eigdecomposition")
        return torch.nn.Linear(self.latent_dim, self.latent_dim, bias=enforce_constant_fn)

    def fit_linear_decoder(self, latent_states: torch.Tensor, states: torch.Tensor) -> torch.nn.Module:
        """Fit a linear decoder mapping the latent state space Z to the state space X."""
        use_bias = False  # TODO: Unsure if to enable. This can be another hyperparameter, or set to true by default.

        # Solve the least squares problem to find the linear decoder matrix and bias
        D, bias = full_rank_lstsq(X=latent_states, Y=states, bias=use_bias)

        # Check shape
        _expected_shape = (np.prod(self.state_features_shape), self.latent_dim)
        assert D.shape == _expected_shape, \
            f"Expected linear decoder shape {_expected_shape}, got {D.shape}"

        # Create a non-trainable linear layer to store the linear decoder matrix and bias term
        lin_decoder = torch.nn.Linear(in_features=self.latent_dim,
                                      out_features=np.prod(self.state_features_shape),
                                      bias=use_bias,
                                      dtype=self.evolution_operator.dtype)
        lin_decoder.weight.data = torch.tensor(D, dtype=torch.float32)
        lin_decoder.weight.requires_grad = False

        if use_bias:
            lin_decoder.bias.data = torch.tensor(bias, dtype=torch.float32)
            lin_decoder.bias.requires_grad = False

        return lin_decoder.to(device=self.evolution_operator.device)

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
            if self.evolution_op_bias:  # Set bias to zero
                self.transfer_op.bias.data = torch.zeros(self.latent_dim)
                self.transfer_op.bias.data = torch.zeros(self.latent_dim)

            if init_mode == "stable":  # This bias the modeling of slow eigfunctions.
                T = torch.eye(self.latent_dim)
                T += 0.001 * torch.randn(self.latent_dim, self.latent_dim)
                # Perform singular value decomposition, ensure that the singular values are all less than 1
                # T = U @ torch.diag(S / S.max()) @ V.T
                self.linear_dynamics.weight.data = T
            elif init_mode == "unit_circle":
                # Initialize the evolution operator with eigenvalues sampled uniformly on the unit circle.
                # As the z_t are real, the eigenvalues and their corresponding eigenvectors are complex conjugates.
                # TODO: This bias the system to model spurious yerky modes with super high frequency. It seems the
                #  best initialization would be to identify a maximum frequency of oscilation and pass it as a parameter
                #  of the initialization mode. E.g. unit_circle_T=30 where T=30[STEPS] is the minimum-period/max_freq.
                blocks = []
                remaining_dims = self.latent_dim
                if self.latent_dim % 2 == 1:
                    blocks.append(np.array([[1.0]]))  # 1x1 block for the real eigenvalue
                    remaining_dims -= 1

                # Sample the angle of the complex eigenvalues uniformly on the unit circle
                theta = np.linspace(0, np.pi / 6, remaining_dims // 2, endpoint=False)
                blocks += [np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) for t in theta]

                # Block diagonal matrix D
                D = scipy.linalg.block_diag(*blocks)
                # Add some white noise to the matrix D, to all off-block-diagonal elements
                # block_diag_mask = scipy.linalg.block_diag(*[np.ones_like(b) for b in blocks]).astype(bool)
                D += 0.001 * np.random.randn(self.latent_dim, self.latent_dim)  # * np.logical_not(block_diag_mask)
                A = D
                # Set the initial value of the evolution operator
                self.linear_dynamics.weight.data = torch.tensor(A, dtype=torch.float32)

                eigenvalues = np.linalg.eigvals(A)
                modulus = np.abs(eigenvalues)
                angle = np.angle(eigenvalues) * 180 / np.pi
                # Check if A is real
                is_real = np.isreal(A).all()
                # Check if all eigenvalues are on the unit circle
                # on_unit_circle = np.allclose(np.abs(eigenvalues), 1)
                # assert is_real and on_unit_circle, f"Real evol op {is_real}, unit circle {on_unit_circle}"
            else:
                logger.warning(f"Evolution operator init mode {init_mode} not implemented")
                return
            logger.info(f"Trainable evolution operator initialized with mode {init_mode}")

    @property
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
        # if self.trainer is None:
        #     return False
        # else:
        #     return self.trainer.state.finished
        return self._is_fitted

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


def _default_lighting_trainer() -> lightning.Trainer:
    return lightning.Trainer(accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                             devices='auto',
                             logger=None,
                             log_every_n_steps=1,
                             max_epochs=100,
                             enable_progress_bar=True)
