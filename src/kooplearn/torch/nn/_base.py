"""Loss functions for representation learning."""

from torch import Tensor
from torch.nn import Module

from kooplearn.torch.nn import _functional as F

__all__ = [
    "AutoEncoderLoss",
    "SpectralContrastiveLoss",
    "VampLoss",
]


class VampLoss(Module):
    r"""Variational Approach for learning Markov Processes (VAMP) score by :cite:t:`Wu2019`.

    .. math::

        \mathcal{L}(x, y) = -\sum_{i} \sigma_{i}(A)^{p} \qquad \text{where}~A = \big(x^{\top}x\big)^{\dagger/2}x^{\top}y\big(y^{\top}y\big)^{\dagger/2}.

    Args:
        schatten_norm (int, optional): Computes the VAMP-p score with ``p = schatten_norm``. Defaults to 2.
        center_covariances (bool, optional): Use centered covariances to compute the VAMP score. Defaults to True.
    """

    def __init__(
        self,
        schatten_norm: int = 2,
        center_covariances: bool = True,
    ) -> None:
        super().__init__()
        self.schatten_norm = schatten_norm
        self.center_covariances = center_covariances

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass of VAMP loss.

        Args:
            x (Tensor): Features for x.
            y (Tensor): Features for y.

        Raises:
            NotImplementedError: If ``schatten_norm`` is not 1 or 2.

        Shape:
            ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

            ``y``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
        """
        return F.vamp_loss(
            x,
            y,
            self.schatten_norm,
            self.center_covariances,
        )


class SpectralContrastiveLoss(Module):
    r"""Spectral contrastive loss based originally introduced by :cite:t:`haochen2021provable`, and adopted for evolution operators in :cite:t:`turri2025self, jeong2025efficient`

    .. math::

        \mathcal{L}(x, y) = \frac{1}{N(N-1)}\sum_{i \neq j}\langle x_{i}, y_{j} \rangle^2 - \frac{2}{N}\sum_{i=1}\langle x_{i}, y_{i} \rangle.

    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass of the L2 contrastive loss.

        Args:
            x (Tensor): Input features.
            y (Tensor): Output features.


        Shape:
            ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.

            ``y``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
        """
        return F.spectral_contrastive_loss(x, y)


class AutoEncoderLoss(Module):
    r"""Single-step Dynamic Autoencoder (DAE) loss introduced by :cite:t:`Lusch2018`.

    This loss combines three objectives to train dynamic autoencoders:

    1. **Reconstruction loss** — measures how well the autoencoder reconstructs inputs.
    2. **Linearity loss** — enforces linear evolution in latent space.
    3. **Prediction loss** — penalizes errors between predicted and actual encoded outputs.

    The total loss is a weighted sum:

    .. math::
        \mathcal{L} =
        \alpha_\mathrm{rec} \, \|x - \phi^{-1}(\phi(x)) \|^2 +
        \alpha_\mathrm{lin} \, \|\phi(y) - K\phi(x) \|^2 +
        \alpha_\mathrm{pred} \, \|y - \phi^{-1}(K\phi(x))\|^2

    where :math:`\phi^{-1}(\phi(x))` is the reconstruction of :math:`x`,
    :math:`\phi(y)` is the encoded output,
    :math:`K\phi(x)` is the evolved input latent representation,
    and :math:`\phi^{-1}(K\phi(x))` is the predicted decoded output.
    """

    def __init__(
        self,
        alpha_rec: float = 1.0,
        alpha_lin: float = 1.0,
        alpha_pred: float = 1.0,
    ) -> None:
        r"""Initialize the Dynamic Autoencoder (DAE) loss.

        Parameters
        ----------
        alpha_rec : float, default=1.0
            Weight for the reconstruction term :math:`\|x - \phi^{-1}(\phi(x)) \|^2`.
        alpha_lin : float, default=1.0
            Weight for the linearity term :math:`\|\phi(y) - K\phi(x) \|^2`.
        alpha_pred : float, default=1.0
            Weight for the prediction term :math:`\|y - \phi^{-1}(K\phi(x))\|^2`.
        """
        super().__init__()
        self.alpha_rec = alpha_rec
        self.alpha_lin = alpha_lin
        self.alpha_pred = alpha_pred

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        x_rec: Tensor,
        y_enc: Tensor,
        x_evo: Tensor,
        y_pred: Tensor,
    ) -> Tensor:
        """Compute the Dynamic Autoencoder loss.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(N, D)``, where ``N`` is the batch size
            and ``D`` is the feature dimension.
        y : torch.Tensor
            Output (target) features. Same shape as ``x``.
        x_rec : torch.Tensor
            Reconstructed version of the input ``x`` produced by the decoder. Same shape as ``x``.
        y_enc : torch.Tensor
            Encoded latent representation of the target ``y``.
        x_evo : torch.Tensor
            Evolved latent representation obtained by applying the learned
            linear operator to the latent encoding of ``x``. Same shape as ``x``.
        y_pred : torch.Tensor
            Predicted decoded output corresponding to the evolved latent state. Same shape as ``x``.

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the total dynamic autoencoder loss.
        """
        return F.dynamic_ae_loss(
            x,
            y,
            x_rec,
            y_enc,
            x_evo,
            y_pred,
            self.alpha_rec,
            self.alpha_lin,
            self.alpha_pred,
        )
