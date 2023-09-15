from kooplearn.abc import TrainableFeatureMap
import lightning
import torch
from typing import Optional
import os 
from pathlib import Path
import pickle
import numpy as np
import logging
from kooplearn.nn.functional import VAMP_score
logger = logging.getLogger('kooplearn')

class VAMPNet(TrainableFeatureMap):
    def __init__(
            self,
            lobe: torch.nn.Module, #As in the official implementation, see https://deeptime-ml.github.io/latest/api/generated/deeptime.decomposition.deep.VAMPNet.html#deeptime.decomposition.deep.VAMPNet
            optimizer_fn: torch.optim.Optimizer, optimizer_kwargs: dict,
            trainer: lightning.Trainer,
            lobe_kwargs: dict = {},
            lobe_timelagged: Optional[torch.nn.Module] = None,
            lobe_timelagged_kwargs: dict = {},
            schatten_norm: int = 2, 
            center_covariances: bool = True,
            seed: Optional[int] = None):
        
        lightning.seed_everything(seed)
        self._lookback_len = -1 #Dummy init value, will be determined at fit time.
        self.lightning_trainer = trainer
        self.lightning_module = VAMPModule(
            lobe,
            optimizer_fn, optimizer_kwargs,
            lobe_kwargs=lobe_kwargs,
            lobe_timelagged=lobe_timelagged,
            lobe_timelagged_kwargs=lobe_timelagged_kwargs,
            schatten_norm=schatten_norm,
            center_covariances=center_covariances
        )
        self.seed = seed
    
    @property
    def is_fitted(self):
        return self._is_fitted

    @property
    def lookback_len(self):
        return self._lookback_len
    
    #TODO: Test
    def save(self, path: os.PathLike):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        ckpt = path / 'lightning.ckpt'
        self.lightning_trainer.save_checkpoint(str(ckpt))
        del self.lightning_module
        model = path / 'kooplearn_model.pkl'
        with open (model, 'wb') as f:
            pickle.dump(self, f)  
    
    #TODO: Test
    @classmethod
    def load(cls, path: os.PathLike):
        path = Path(path)
        model = path / 'kooplearn_model.pkl'
        ckpt = path / 'lightning.ckpt'
        with open(model, 'rb') as f:
            restored_obj = pickle.load(f)
        assert isinstance(restored_obj, cls)
        restored_obj.lightning_module = VAMPModule.load_from_checkpoint(str(ckpt))
        return restored_obj

    def fit(self, 
            train_dataloaders = None,
            val_dataloaders = None,
            datamodule: Optional[lightning.LightningDataModule] = None,
            ckpt_path: Optional[str] = None):
        """Fits the VAMPNet feature map. Accepts the same arguments as :meth:`lightning.Trainer.fit`, except for the ``model`` keyword, which is automatically set to the VAMPNet feature map.

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
        if (train_dataloaders is not None or val_dataloaders is not None) and datamodule is not None:
            raise ValueError(
                "You cannot pass `train_dataloader` or `val_dataloaders` to `VAMPNet.fit(datamodule=...)`"
            )
        #Get the shape of the first batch to determine the lookback_len
        if train_dataloaders is None:
            assert isinstance(datamodule, lightning.LightningDataModule)   
            for batch in datamodule.train_dataloader():
                context_len = batch.shape[1]
                break
        else:
            assert isinstance(train_dataloaders, torch.utils.data.DataLoader)         
            for batch in train_dataloaders:
                context_len = batch.shape[1]
                break

        self._lookback_len = context_len - 1
        
        self.lightning_trainer.fit(model=self.lightning_module, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders, datamodule=datamodule, ckpt_path=ckpt_path)
        self._is_fitted = True
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = torch.from_numpy(X).float()
        X.to(self.lightning_module.device)
        self.lightning_module.eval()
        with torch.no_grad():
            embedded_X = self.lightning_module.lobe(X)
            embedded_X = embedded_X.detach().numpy()
        return embedded_X

class VAMPModule(lightning.LightningModule):
    def __init__(
        self,
        lobe: torch.nn.Module, #As in the official implementation, see https://deeptime-ml.github.io/latest/api/generated/deeptime.decomposition.deep.VAMPNet.html#deeptime.decomposition.deep.VAMPNet
        optimizer_fn: torch.optim.Optimizer, optimizer_kwargs: dict,
        lobe_kwargs: dict = {},
        lobe_timelagged: Optional[torch.nn.Module] = None,
        lobe_timelagged_kwargs: dict = {},
        schatten_norm: int = 2,
        center_covariances: bool = True):
        
        super().__init__()
        self.save_hyperparameters(ignore=["lobe", "optimizer_fn"])
        self.lobe = lobe(**lobe_kwargs)
        if lobe_timelagged is not None:
            self.lobe_timelagged = lobe_timelagged(**lobe_timelagged_kwargs)
        else:
            self.lobe_timelagged = self.lobe
        self._optimizer = optimizer_fn

    def configure_optimizers(self):
        return self._optimizer(self.parameters(), **self.hparams.optimizer_kwargs)
    
    def training_step(self, train_batch, batch_idx):
        X, Y = train_batch[:, :-1, ...], train_batch[:, 1:, ...]
        encoded_X, encoded_Y = self(X), self(Y)
        
        if self.hparams.center_covariances:
            encoded_X -= encoded_X.mean(dim=0, keepdim=True)
            encoded_Y -= encoded_Y.mean(dim=0, keepdim=True)

        _norm = torch.rsqrt(torch.tensor(encoded_X.shape[0]))
        encoded_X *= _norm
        encoded_Y *= _norm

        cov_X = torch.mm(encoded_X.T, encoded_X)
        cov_Y = torch.mm(encoded_Y.T, encoded_Y)
        cov_XY = torch.mm(encoded_X.T, encoded_Y)

        loss = -1*VAMP_score(cov_X, cov_Y, cov_XY, schatten_norm=self.hparams.schatten_norm)
        
        return loss
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        lookback_len = X.shape[1]
        batch_size = X.shape[0]
        X = X.view(lookback_len*batch_size, -1)
        encoded_X = self.lobe(X).view(batch_size, lookback_len, -1)
        return encoded_X.view(batch_size, -1)