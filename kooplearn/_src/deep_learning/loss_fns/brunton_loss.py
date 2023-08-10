import torch


def brunton_loss(decoded_encoded_x_value, x_value,  # loss reconstruction
                 advanced_encoded_m_x_value, encoded_m_y_value,  # loss linear dynamics
                 decoded_advanced_encoded_m_x_value, m_y_value,  # loss future state prediction
                 alpha_1=1, alpha_2=1):  # alpha_3 is weight decay in optimizer
    loss_reconstruction = torch.nn.functional.mse_loss(decoded_encoded_x_value, x_value)
    loss_linear_dynamics = torch.nn.functional.mse_loss(advanced_encoded_m_x_value, encoded_m_y_value)
    loss_future_state_prediction = torch.nn.functional.mse_loss(decoded_advanced_encoded_m_x_value, m_y_value)
    loss_inf = ((decoded_encoded_x_value - x_value).norm(p=float('inf')) +
                (decoded_advanced_encoded_m_x_value[..., 0, :, :] - m_y_value[..., 0, :, :]).norm(p=float('inf')))
    return alpha_1 * (loss_reconstruction + loss_future_state_prediction) + loss_linear_dynamics + alpha_2 * loss_inf
