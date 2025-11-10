from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
from flax import nnx
from sklearn.base import BaseEstimator, TransformerMixin

# Define type aliases for clarity
ArrayLike = Union[np.ndarray, jnp.ndarray]


class NnxFeatureMapEmbedder(BaseEstimator, TransformerMixin):
    """
    sklearn-style transformer wrapping NNX Modules (encoder/decoder).

    This class mirrors the original PyTorch-based :class:`kooplearn.torch.FeatureMapEmbedder`,
    using `Flax' NNX framework <https://flax.readthedocs.io/en/stable/>`_. It accepts stateful ``nnx.Module`` instances, JIT-compiles
    their forward pass, and uses eval mode for inference.

    The interface follows the scikit-learn ``TransformerMixin`` pattern,
    providing ``fit`` and ``transform`` methods.

    Parameters
    ----------
    encoder : nnx.Module
        A NNX Module instance mapping input data to latent space.
        Its ``__call__`` method will be JIT-compiled.
    decoder : nnx.Module, optional
        A NNX Module instance mapping latent space back to input space.
        Its ``__call__`` method will be JIT-compiled. If not provided, only
        encoding (``transform``) is supported.

    Examples
    --------
    Example usage with the modern Flax NNX API:

    .. code-block:: python

        import jax
        import jax.numpy as jnp
        from flax import nnx
        import numpy as np

        class SimpleEncoder(nnx.Module):
            def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
                self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        # 1. Initialize module and state
        rngs = nnx.Rngs(0)
        encoder_module = SimpleEncoder(in_features=5, out_features=10, rngs=rngs)

        # 2. Create the transformer
        transformer = NnxFeatureMapEmbedder(encoder=encoder_module)

        # 3. Use it
        data = np.random.rand(100, 5).astype(np.float32)
        latent_features = transformer.transform(data)

        print(latent_features.shape)
        # (100, 10)

    Notes
    -----
    Internally, the encoder and decoder forward passes are wrapped in
    ``nnx.jit`` for efficient computation. Stateful ``nnx.Module`` instances
    are handled correctly by maintaining their parameter/state dictionaries.
    """

    def __init__(
        self,
        encoder: nnx.Module,
        decoder: Optional[nnx.Module] = None,
    ):
        self.encoder = encoder
        self.decoder = decoder

        # JIT-compile the module's call method for performance.
        # nnx.jit handles the static/dynamic split of the module.
        self.jitted_encoder_call = nnx.jit(self.encoder)

        self.jitted_decoder_call = None
        if self.decoder is not None:
            self.jitted_decoder_call = nnx.jit(self.decoder)

    def fit(self, X: ArrayLike = None, y: ArrayLike = None) -> "NnxFeatureMapEmbedder":
        """
        No fitting is performed by this transformer.
        The encoder/decoder are assumed to be pre-trained.
        """
        # sklearn API requires fit(), so we return self.
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Encode data using the JAX NNX encoder."""
        X_jax = self._to_array(X)

        # Use nnx.eval_mode for inference (e.g., sets dropout/batchnorm to eval)
        if hasattr(self.encoder, "eval"):
            self.encoder.eval()
        Z = self.jitted_encoder_call(X_jax)

        return np.asarray(Z)

    def inverse_transform(self, Z: ArrayLike) -> np.ndarray:
        """Decode data using the JAX NNX decoder, if available."""
        if self.decoder is None or self.jitted_decoder_call is None:
            raise AttributeError("No decoder provided for inverse_transform.")

        Z_jax = self._to_array(Z)

        # Use nnx.eval_mode for inference
        if hasattr(self.decoder, "eval"):
            self.decoder.eval()
        X_rec = self.jitted_decoder_call(Z_jax)

        return np.asarray(X_rec)

    def _to_array(self, array: ArrayLike) -> jnp.ndarray:
        """Helper: ensure input is a float32 JAX array."""
        return jnp.asarray(array, dtype=jnp.float32)

    def __repr__(self) -> str:
        """Return a string representation of the transformer."""
        encoder_name = self.encoder.__class__.__name__
        decoder_name = "None"
        if self.decoder is not None:
            decoder_name = self.decoder.__class__.__name__

        return f"NnxFeatureMapEmbedder(encoder={encoder_name}, decoder={decoder_name})"
