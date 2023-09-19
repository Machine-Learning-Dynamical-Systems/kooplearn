import os
import pickle
from pathlib import Path
from kooplearn.abc import BaseModel
from typing import Optional, Union, Callable
import weakref
import numpy as np
from scipy.linalg import eig
from kooplearn._src.utils import check_is_fitted
from kooplearn._src.check_deps import check_torch_deps
from kooplearn._src.context_window_utils import check_contexts
from kooplearn.models.ae.utils import _encode, _decode, _evolve
import logging
logger = logging.getLogger('kooplearn')
check_torch_deps()
import torch  # noqa: E402
import lightning  # noqa: E402

class LuschKutzBrunton(BaseModel):
    def __init__(
            self,
            encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            latent_dim: int,
            optimizer_fn: torch.optim.Optimizer, optimizer_kwargs: dict,
            trainer: lightning.Trainer,
            loss_weights: dict = {'rec': 1., 'pred': 1., 'lin': 1.},
            encoder_kwargs: dict = {},
            decoder_kwargs: dict = {},
            seed: Optional[int] = None):
        lightning.seed_everything(seed)
        self.lightning_trainer = trainer
        self.lightning_module = LuschKutzBruntonModule(
            encoder, decoder, latent_dim,
            optimizer_fn, optimizer_kwargs,
            loss_weights=loss_weights,
            encoder_kwargs=encoder_kwargs,
            decoder_kwargs=decoder_kwargs,
            kooplearn_model_weakref = weakref.ref(self)
        )
        self.seed = seed
        self._is_fitted = False
        #Todo: Add warning on lookback_len for this model

    def fit(self, 
            train_dataloaders = None,
            val_dataloaders = None,
            datamodule: Optional[lightning.LightningDataModule] = None,
            ckpt_path: Optional[str] = None):
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
        if (train_dataloaders is not None or val_dataloaders is not None) and datamodule is not None:
            raise ValueError(
                "You cannot pass `train_dataloader` or `val_dataloaders` to `VAMPNet.fit(datamodule=...)`"
            )
        #Get the shape of the first batch to determine the lookback_len
        if train_dataloaders is None:
            assert isinstance(datamodule, lightning.LightningDataModule)   
            for batch in datamodule.train_dataloader():
                self.check_shape(batch)
                self._fit_data_trail_dims = batch.shape[2:]
                break
        else:
            assert isinstance(train_dataloaders, torch.utils.data.DataLoader)         
            for batch in train_dataloaders:
                self.lightning_module._check_shape(batch)
                self._fit_data_trail_dims = batch.shape[2:]
                break
        
        self.lightning_trainer.fit(model=self.lightning_module, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders, datamodule=datamodule, ckpt_path=ckpt_path)
        self._is_fitted = True

    def _move_contexts_to_torch(self, data: np.ndarray):
        check_is_fitted('_fit_data_trail_dims', self)
        data = check_contexts(data, lookback_len=self.lookback_len)
        if data.shape[1] != self.lookback_len:
            logger.warning(f'The model can perform inference only on the lookback slices of the data. Contexts of length {data.shape[1]} are provided, while the lookback length is {self.lookback_len}. The contexts will be truncated to [:, :{self.lookback_len}, ...]')
            data = data[:, :self.lookback_len, ...]
        model_device = self.lightning_module.device
        return torch.from_numpy(data).float().to(model_device)

    def predict(self, data: np.ndarray, t: int = 1, observables: Optional[Union[Callable, np.ndarray]] = None):
        data = self._move_contexts_to_torch(data)
        n_samples = data.shape[0]
        lookback_len = data.shape[1]
        assert lookback_len == self.lookback_len
        trail_dims = data.shape[2:]
        with torch.no_grad():
            K = self.lightning_module.koopman_operator.weight
            data = data.view(n_samples*lookback_len, *trail_dims)
            encoded_data = self.lightning_module._encode(data)
            evolved_encoding = torch.mm(encoded_data, torch.matrix_power(K, t))
            evolved_encoding = evolved_encoding.view(n_samples, lookback_len, -1)
            evolved_data = self.lightning_module._decode(evolved_encoding)
            evolved_data = evolved_data.detach().cpu().numpy()[:, -1, ...] #Hardcoding the lookback len
        if observables is None:
            return evolved_data    
        elif callable(observables):
            return observables(evolved_data)
        else:
            raise NotImplementedError("Only callable observables or None are supported at the moment.")

    def modes(self, data: np.ndarray, observables: Optional[Union[Callable, np.ndarray]] = None):
        pass

    def eig(self, eval_left_on: Optional[np.ndarray] = None, eval_right_on: Optional[np.ndarray] = None):
        if hasattr(self, '_eig_cache'):
            w, vl, vr = self._eig_cache
        else:
            K = self.lightning_module.koopman_operator.weight
            K_np = K.detach().cpu().numpy()
            w, vl, vr = eig(K_np, left=True, right=True)
            self._eig_cache = w, vl, vr
        
        if eval_left_on is None and eval_right_on is None:
            # (eigenvalues,)
            return w
        else:
            raise NotImplementedError("Left and right eigenfunction evaluations are not supported at the moment.")

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
        restored_obj.lightning_module = LuschKutzBruntonModule.load_from_checkpoint(str(ckpt))
        return restored_obj

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def lookback_len(self) -> int:
        return 1
    
class LuschKutzBruntonModule(lightning.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_dim: int,
        optimizer_fn: torch.optim.Optimizer, optimizer_kwargs: dict,
        loss_weights: dict = {'rec': 1., 'pred': 1., 'lin': 1.},
        encoder_kwargs: dict = {},
        decoder_kwargs: dict = {},
        kooplearn_model_weakref: weakref.ReferenceType = None):

        super().__init__()
        self.save_hyperparameters(ignore=['kooplearn_model_weakref'])
        self.encoder = encoder(**encoder_kwargs)
        self.decoder = decoder(**decoder_kwargs)
        self.koopman_operator = torch.nn.Linear(latent_dim, latent_dim, bias=False)
        self._optimizer = optimizer_fn
        self._kooplearn_model_weakref = kooplearn_model_weakref

    def configure_optimizers(self):
        return self._optimizer(self.parameters(), **self.hparams.optimizer_kwargs)
    
    def training_step(self, train_batch, batch_idx):
        lookback_len = self._kooplearn_model_weakref().lookback_len
        encoded_batch = _encode(train_batch, self.encoder)
        evolved_batch = _evolve(encoded_batch, lookback_len, self.koopman_operator)
        decoded_batch = _decode(evolved_batch, self.decoder)

        MSE = torch.nn.MSELoss()
        #Reconstruction + prediction loss
        rec_loss = MSE(train_batch[:, :lookback_len, ...], decoded_batch[:, :lookback_len, ...])
        pred_loss = MSE(train_batch[:, lookback_len:, ...], decoded_batch[:, lookback_len:, ...])
        #Linear loss
        lin_loss = MSE(encoded_batch[:, lookback_len:, ...], evolved_batch[:, lookback_len:, ...])

        self.log_dict({
            'train/reconstruction_loss': rec_loss.item(),
            'train/prediction_loss': pred_loss.item(),
            'train/linear_loss': lin_loss.item()
        }, on_step=True, prog_bar=True, logger=True)

        alpha_rec = self.hparams.loss_weights.get('rec', 1.)
        alpha_pred = self.hparams.loss_weights.get('pred', 1.)
        alpha_lin = self.hparams.loss_weights.get('lin', 1.)

        loss = alpha_rec*rec_loss + alpha_pred*pred_loss + alpha_lin*lin_loss
        return loss
    
    def _check_shape(self, batch: torch.Tensor):
        assert batch.ndim >= 3, f"Batch must have at least 3 dimensions corresponding to (batch_size, context_len, *feature_dims), got {batch.ndim}."
        # Caution: this method is designed only for internal calling.
        Z = self._encode(batch)
        Y = self._evolve(Z)
        X_evol = self._decode(Y) #Should fail if the shape is wrong
        assert Z.shape == Y.shape
        assert batch.shape == X_evol.shape   