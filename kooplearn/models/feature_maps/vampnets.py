from kooplearn.abc import TrainableFeatureMap
import lightning
import torch
from typing import Optional
import os 
from pathlib import Path
import pickle
import numpy as np
import logging
import weakref
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
        
        if seed is not None:
            lightning.seed_everything(seed)
        
        self.lightning_trainer = trainer
        self.lightning_module = VAMPModule(
            lobe,
            optimizer_fn, optimizer_kwargs,
            lobe_kwargs=lobe_kwargs,
            lobe_timelagged=lobe_timelagged,
            lobe_timelagged_kwargs=lobe_timelagged_kwargs,
            schatten_norm=schatten_norm,
            center_covariances=center_covariances,
            kooplearn_feature_map_weakref = weakref.ref(self)
        )
        self.seed = seed
        self._lookback_len = -1 #Dummy init value, will be determined at fit time.
        self._is_fitted = False
    
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
        #Save the trainer
        torch.save(self.lightning_trainer, path / 'lightning_trainer.bin')
        #Save the lightning checkpoint
        ckpt = path / 'lightning.ckpt'
        self.lightning_trainer.save_checkpoint(str(ckpt))
        del self.lightning_module
        del self.lightning_trainer
        model = path / 'kooplearn_model.pkl'
        with open (model, 'wb') as f:
            pickle.dump(self, f)  
    
    #TODO: Test
    @classmethod
    def load(cls, path: os.PathLike):
        path = Path(path)
        trainer = torch.load(path / 'lightning_trainer.bin')
        ckpt = path / 'lightning.ckpt'
        with open(path / 'kooplearn_model.pkl', 'rb') as f:
            restored_obj = pickle.load(f)
        assert isinstance(restored_obj, cls)
        restored_obj.lightning_trainer = trainer
        restored_obj.lightning_module = VAMPModule.load_from_checkpoint(str(ckpt))
        return restored_obj

    def fit(self, 
            train_dataloaders = None,
            val_dataloaders = None,
            datamodule: Optional[lightning.LightningDataModule] = None,
            ckpt_path: Optional[str] = None,
            verbose: bool = True):
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
        if verbose:
            print(f"Fitting {self.__class__.__name__}. Lookback window length set to {self.lookback_len}")
        self.lightning_trainer.fit(model=self.lightning_module, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders, datamodule=datamodule, ckpt_path=ckpt_path)
        self._is_fitted = True
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = torch.from_numpy(X.copy(order='C')).float()
        self.lightning_module.eval()
        with torch.no_grad():
            embedded_X = self.lightning_module.lobe(X.to(self.lightning_module.device))
            embedded_X = embedded_X.detach().cpu().numpy()
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
        center_covariances: bool = True,
        kooplearn_feature_map_weakref = None):
        
        super().__init__()
        self.save_hyperparameters(ignore=["lobe", "optimizer_fn", "kooplearn_feature_map_weakref"])
        if ('lr' in optimizer_kwargs) or ('learning_rate' in optimizer_kwargs): #For Lightning's LearningRateFinder
            self.lr = optimizer_kwargs.get('lr', optimizer_kwargs.get('learning_rate'))
        self.lobe = lobe(**lobe_kwargs)
        if lobe_timelagged is not None:
            self.lobe_timelagged = lobe_timelagged(**lobe_timelagged_kwargs)
        else:
            self.lobe_timelagged = self.lobe
        self._optimizer = optimizer_fn
        self._kooplearn_feature_map_weakref = kooplearn_feature_map_weakref

    def configure_optimizers(self):
        return self._optimizer(self.parameters(), **self.hparams.optimizer_kwargs)
    
    def training_step(self, train_batch, batch_idx):
        X, Y = train_batch[:, :-1, ...], train_batch[:, 1:, ...]
        encoded_X, encoded_Y = self.forward(X), self.forward(Y, time_lagged=True)
        
        if self.hparams.center_covariances:
            encoded_X = encoded_X - encoded_X.mean(dim=0, keepdim=True)
            encoded_Y = encoded_Y - encoded_Y.mean(dim=0, keepdim=True)

        _norm = torch.rsqrt(torch.tensor(encoded_X.shape[0]))
        encoded_X *= _norm
        encoded_Y *= _norm

        cov_X = torch.mm(encoded_X.T, encoded_X)
        cov_Y = torch.mm(encoded_Y.T, encoded_Y)
        cov_XY = torch.mm(encoded_X.T, encoded_Y)
        loss = -1*VAMP_score(cov_X, cov_Y, cov_XY, schatten_norm=self.hparams.schatten_norm)
        self.log('train/VAMP_score', -1.0*loss.item(), on_step=True, prog_bar=True, logger=True)
        return loss
    
    def forward(self, X: torch.Tensor, time_lagged: bool = False) -> torch.Tensor:
        # Caution: this method is designed only for internal calling by the VAMPNet feature map.
        lookback_len = X.shape[1]
        batch_size = X.shape[0]
        trail_dims = X.shape[2:]
        X = X.view(lookback_len*batch_size, *trail_dims)
        if time_lagged:
            encoded_X = self.lobe_timelagged(X)
        else:
            encoded_X = self.lobe(X)
        trail_dims = encoded_X.shape[1:]
        encoded_X = encoded_X.view(batch_size, lookback_len, *trail_dims)
        return encoded_X.view(batch_size, -1)