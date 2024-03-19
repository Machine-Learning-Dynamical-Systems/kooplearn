import logging
import weakref
from copy import deepcopy
from typing import Optional

import lightning
import numpy as np
import torch

from kooplearn._src.serialization import pickle_load, pickle_save
from kooplearn.abc import ContextWindowDataset, TrainableFeatureMap

logger = logging.getLogger("kooplearn")


class NNFeatureMap(TrainableFeatureMap):
    """Implements a generic Neural Network feature maps. Can be used in conjunction to :class:`kooplearn.models.Nonlinear` to learn a Koopman/Transfer operator from data. The NN feature map is trained using the :class:`lightning.LightningModule` API, and can be trained using the :class:`lightning.Trainer` API. See the `PyTorch Lightning documentation <https://pytorch-lightning.readthedocs.io/en/latest/>`_ for more information.

    Args:
        encoder (torch.nn.Module): Encoder network. Should be a subclass of :class:`torch.nn.Module`. Will be initialized as ``encoder(**encoder_kwargs)``.
        loss_fn (torch.nn.Module): Loss function from :class:`kooplearn.nn`
        optimizer_fn (torch.optim.Optimizer): Any optimizer from :class:`torch.optim.Optimizer`.
        trainer (lightning.Trainer): An initialized `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_ object used to train the NN feature map.
        lagged_encoder (Optional[torch.nn.Module], optional): Encoder network for the time-lagged data. Defaults to None. If None, the encoder network is used for time-lagged data as well. If not None, it will be initialized as ``lagged_encoder(**lagged_encoder_kwargs)``.
        encoder_kwargs (dict, optional): Dictionary of keyword arguments passed to the encoder network upon initialization. Defaults to ``{}``.
        loss_kwargs (dict): Dictionary of keyword arguments passed to the loss function at initialization. Defaults to ``{}``.
        optimizer_kwargs (dict): Dictionary of keyword arguments passed to the optimizer at initialization. Defaults to ``{}``.
        lagged_encoder_kwargs (dict, optional): Dictionary of keyword arguments passed to `lagged_encoder` upon initialization. Defaults to ``{}``.
        seed (int, optional): Seed of the internal random number generator. Defaults to None.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer_fn: torch.optim.Optimizer,
        trainer: lightning.Trainer,
        lagged_encoder: Optional[torch.nn.Module] = None,
        encoder_kwargs: dict = {},
        loss_kwargs: dict = {},
        optimizer_kwargs: dict = {},
        lagged_encoder_kwargs: dict = {},
        seed: Optional[int] = None,
    ):
        if seed is not None:
            lightning.seed_everything(seed)

        self.lightning_trainer = trainer
        self.lightning_module = PLModule(
            encoder,
            loss_fn,
            optimizer_fn,
            lagged_encoder=lagged_encoder,
            encoder_kwargs=encoder_kwargs,
            loss_kwargs=loss_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lagged_encoder_kwargs=lagged_encoder_kwargs,
            kooplearn_feature_map_weakref=weakref.ref(self),
        )
        self.seed = seed
        self._lookback_len = -1  # Dummy init value, will be determined at fit time.
        self._is_fitted = False

    @property
    def is_fitted(self):
        return self._is_fitted

    @property
    def lookback_len(self):
        return self._lookback_len

    def save(self, filename):
        """Serialize the model to a file.

        Args:
            filename (path-like or file-like): Save the model to file.
        """
        # Delete (un-picklable) weakref self.lightning_module._kooplearn_feature_map_weakref
        self.lightning_module._kooplearn_feature_map_weakref = None
        pickle_save(self, filename)

    @classmethod
    def load(cls, filename):
        """Load a serialized model from a file.

        Args:
            filename (path-like or file-like): Load the model from file.

        Returns:
            The loaded model.
        """
        restored_obj = pickle_load(cls, filename)
        # Restore the weakref
        restored_obj.lightning_module._kooplearn_feature_map_weakref = weakref.ref(
            restored_obj
        )
        return restored_obj

    def fit(
        self,
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule: Optional[lightning.LightningDataModule] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """Fits the NN feature map. Accepts the same arguments as :meth:`lightning.Trainer.fit`, except for the ``model`` keyword, which is automatically set to the NN feature map.

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
                "You cannot pass `train_dataloader` or `val_dataloaders` to `NNFeatureMap.fit(datamodule=...)`"
            )
        # Get the shape of the first batch to determine the lookback_len
        if train_dataloaders is None:
            assert isinstance(datamodule, lightning.LightningDataModule)
            for batch in datamodule.train_dataloader():
                assert isinstance(batch, ContextWindowDataset)
                context_len = batch.context_length
                break
        else:
            assert isinstance(train_dataloaders, torch.utils.data.DataLoader)
            for batch in train_dataloaders:
                assert isinstance(batch, ContextWindowDataset)
                context_len = batch.context_length
                break

        self._lookback_len = context_len - 1
        if verbose:
            print(
                f"Fitting {self.__class__.__name__}. Lookback window length set to {self.lookback_len}"
            )
        self.lightning_trainer.fit(
            model=self.lightning_module,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
        self._is_fitted = True

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = torch.from_numpy(X.copy(order="C")).float()
        self.lightning_module.eval()
        with torch.no_grad():
            embedded_X = self.lightning_module.encoder(
                X.to(self.lightning_module.device)
            )
            embedded_X = embedded_X.detach().cpu().numpy()
        return embedded_X


class PLModule(lightning.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer_fn: torch.optim.Optimizer,
        lagged_encoder: Optional[torch.nn.Module] = None,
        encoder_kwargs: dict = {},
        loss_kwargs: dict = {},
        optimizer_kwargs: dict = {},
        lagged_encoder_kwargs: dict = {},
        kooplearn_feature_map_weakref=None,
    ):
        super().__init__()

        self.save_hyperparameters(
            ignore=[
                "encoder",
                "loss_fn",
                "optimizer_fn",
                "lagged_encoder",
                "kooplearn_feature_map_weakref",
            ]
        )
        if "lr" in optimizer_kwargs:  # For Lightning's LearningRateFinder
            _tmp_opt_kwargs = deepcopy(optimizer_kwargs)
            self.lr = _tmp_opt_kwargs.pop("lr")
            self.opt_kwargs = _tmp_opt_kwargs
        else:
            self.lr = 1e-3
            logger.warning(
                "No learning rate specified. Using default value of 1e-3. You can specify the learning rate by passing it to the optimizer_kwargs argument."
            )

        self.encoder = encoder(**encoder_kwargs)
        if lagged_encoder is not None:
            self.lagged_encoder = lagged_encoder(**lagged_encoder_kwargs)
        else:
            self.lagged_encoder = self.encoder
        self._optimizer = optimizer_fn
        self._loss = loss_fn(**loss_kwargs)
        self._kooplearn_feature_map_weakref = kooplearn_feature_map_weakref

    def configure_optimizers(self):
        kw = self.opt_kwargs | {"lr": self.lr}
        return self._optimizer(self.parameters(), **kw)

    def training_step(self, train_batch: ContextWindowDataset, batch_idx):
        lookback_len = self._kooplearn_feature_map_weakref().lookback_len
        X, Y = train_batch.lookback(lookback_len), train_batch.lookback(
            lookback_len, slide_by=1
        )
        encoded_X, encoded_Y = self.forward(X), self.forward(Y, lagged=True)
        loss = self._loss(encoded_X, encoded_Y)
        metrics = {f"train/{self._loss.__class__.__name__}": loss.item()}
        self.log_dict(metrics, on_step=True, prog_bar=True, logger=True)
        return loss

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch.data = batch.data.to(device)
        return batch

    def forward(self, X: torch.Tensor, lagged: bool = False) -> torch.Tensor:
        # Caution: this method is designed only for internal calling by the DPNet feature map.
        lookback_len = X.shape[1]
        batch_size = X.shape[0]
        trail_dims = X.shape[2:]
        X = X.view(lookback_len * batch_size, *trail_dims)
        if lagged:
            encoded_X = self.lagged_encoder(X)
        else:
            encoded_X = self.encoder(X)
        encoded_trail_dims = encoded_X.shape[1:]
        encoded_X = encoded_X.view(batch_size, lookback_len, *encoded_trail_dims)
        return encoded_X.view(batch_size, -1)
