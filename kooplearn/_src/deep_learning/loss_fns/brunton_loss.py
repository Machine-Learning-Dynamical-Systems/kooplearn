import torch


def brunton_loss(
        decoded_encoded_x_value: torch.Tensor, x_value: torch.Tensor,  # loss reconstruction
        advanced_encoded_m_x_value: torch.Tensor, encoded_m_y_value: torch.Tensor,  # loss linear dynamics
        decoded_advanced_encoded_m_x_value: torch.Tensor, m_y_value: torch.Tensor,  # loss future state prediction
        alpha_1: float = 1.0, alpha_2: float = 1.0  # alpha_3 is weight decay in optimizer
) -> torch.Tensor:
    """Computes the loss used in [1]

    Parameters:
        decoded_encoded_x_value: value of x after being encoded and then decoded.
        x_value: value of x.
        advanced_encoded_m_x_value: value of x (m consecutive values) after being encoded and then advanced of one time
            step.
        encoded_m_y_value: value of y (m consecutive values) after being encoded.
        decoded_advanced_encoded_m_x_value: value of x (m consecutive values) after being encoded, advanced of one time
            step and then decoded.
        m_y_value: value of y (m consecutive values).
        alpha_1: Weight of the reconstruction and future state prediction loss terms. Same notation as in [1].
        alpha_2: Weight of the infinity loss term. Same notation as in [1].

    Shapes:
        decoded_encoded_x_value: (..., 1, n_features, n_timesteps)
        x_value: (..., 1, n_features, n_timesteps)
        advanced_encoded_m_x_value: (..., m, n_features, n_timesteps)
        encoded_m_y_value: (..., m, n_features, n_timesteps)
        decoded_advanced_encoded_m_x_value: (..., m, n_features, n_timesteps)
        m_y_value: (..., m, n_features, n_timesteps)

    Returns:
        Loss value.

    [1] Lusch, Bethany, J. Nathan Kutz, and Steven L. Brunton. “Deep Learning for Universal Linear Embeddings of
    Nonlinear Dynamics.” Nature Communications 9, no. 1 (November 23, 2018): 4950.
    https://doi.org/10.1038/s41467-018-07210-0.
    """
    loss_reconstruction = torch.nn.functional.mse_loss(decoded_encoded_x_value, x_value)
    loss_linear_dynamics = torch.nn.functional.mse_loss(advanced_encoded_m_x_value, encoded_m_y_value)
    loss_future_state_prediction = torch.nn.functional.mse_loss(decoded_advanced_encoded_m_x_value, m_y_value)
    loss_inf = ((decoded_encoded_x_value - x_value).norm(p=float('inf')) +
                (decoded_advanced_encoded_m_x_value[..., 0, :, :] - m_y_value[..., 0, :, :]).norm(p=float('inf')))
    return alpha_1 * (loss_reconstruction + loss_future_state_prediction) + loss_linear_dynamics + alpha_2 * loss_inf
