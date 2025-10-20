import torch
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureMapEmbedder(BaseEstimator, TransformerMixin):
    """
    sklearn-style transformer wrapping a PyTorch encoder (and optional decoder).

    Parameters
    ----------
    encoder : torch.nn.Module
        Neural network mapping input data to latent space.
    decoder : torch.nn.Module, optional
        Neural network mapping latent space back to input space.
    device : str, optional
        Device for computation ('cpu' or 'cuda'). Defaults to auto-detect.
    """

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module = None, device: str | None = None):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        if self.decoder is not None:
            self.decoder.to(self.device)

    def fit(self, X=None, y=None):
        """No fitting needed unless encoder/decoder are trainable elsewhere."""
        # sklearn API requires fit(), so we return self.
        return self

    def transform(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Encode data using the neural encoder."""
        self.encoder.eval()
        X_tensor = self._to_tensor(X)
        with torch.no_grad():
            Z = self.encoder(X_tensor)
        return Z.cpu().numpy()

    def inverse_transform(self, Z: np.ndarray | torch.Tensor) -> np.ndarray:
        """Decode data using the neural decoder, if available."""
        if self.decoder is None:
            raise AttributeError("No decoder provided for inverse_transform.")
        self.decoder.eval()
        Z_tensor = self._to_tensor(Z)
        with torch.no_grad():
            X_rec = self.decoder(Z_tensor)
        return X_rec.cpu().numpy()

    def _to_tensor(self, array: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Helper: ensure input is a float tensor on the correct device."""
        if isinstance(array, np.ndarray):
            tensor = torch.from_numpy(array.copy(order="C")).float()
        else:
            tensor = array.float()
        return tensor.to(self.device)

    def __repr__(self):
        return (f"NeuralTransformer(encoder={self.encoder.__class__.__name__}, "
                f"decoder={self.decoder.__class__.__name__ if self.decoder else None}, "
                f"device='{self.device}')")