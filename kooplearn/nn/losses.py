from kooplearn._src.check_deps import check_torch_deps

check_torch_deps()

import torch  # noqa: E402

import kooplearn.nn.functional as F  # noqa: E402

__all__ = ["VAMPLoss", "DPLoss"]


class VAMPLoss(torch.nn.Module):
    def __init__(self, schatten_norm: int = 2, center_covariances: bool = True):
        """Initializes the Variational Approach for learning Markov Processes (VAMP) loss by :footcite:t:`Wu2019`.

        Args:
            schatten_norm (int, optional): Computes the VAMP-p score with ``p = schatten_norm``. Defaults to 2.
            center_covariances (bool, optional): Use centered covariances to compute the VAMP score. Defaults to True.

        Raises:
            NotImplementedError: If ``schatten_norm`` is not 1 or 2.

        """
        super().__init__(self)
        if schatten_norm not in [1, 2]:
            raise NotImplementedError(f"Schatten norm {schatten_norm} not implemented")
        self.schatten_norm = schatten_norm
        self.center_covariances = center_covariances

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """Compute the VAMP loss function

        Args:
            X (torch.Tensor): Covariates for the initial time steps.
            Y (torch.Tensor): Covariates for the evolved time steps.
        """
        return -F.vamp_score(
            X,
            Y,
            schatten_norm=self.schatten_norm,
            center_covariances=self.center_covariances,
        )


class DPLoss(torch.nn.Module):
    def __init__(
        self, metric_deformation: float = 1.0, center_covariances: bool = True
    ):
        """Initializes the (Relaxed) Deep Projection loss by :footcite:t:`Kostic2023DPNets`.

        Args:
            metric_deformation (float, optional): Strength of the metric metric deformation loss: Defaults to 1.0.
            center_covariances (bool, optional): Use centered covariances to compute the VAMP score. Defaults to True.

        """
        super().__init__(self)
        self.metric_deformation = metric_deformation
        self.center_covariances = center_covariances

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """Compute the Deep Projection loss function

        Args:
            X (torch.Tensor): Covariates for the initial time steps.
            Y (torch.Tensor): Covariates for the evolved time steps.
        """
        return -F.deepprojection_score(
            X,
            Y,
            metric_deformation=self.metric_deformation,
            center_covariances=self.center_covariances,
        )