import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, flatten_input=False, output_activation_fn=nn.Identity(),
                 activation_fn=nn.ReLU()):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.output_activation_fn = output_activation_fn
        self.activation_fn = activation_fn
        layer_dims = [input_dim] + hidden_dims
        if flatten_input:
            module_list = [nn.Flatten(start_dim=-2, end_dim=-1)]
        else:
            module_list = []
        for i, dim in enumerate(layer_dims[:-1]):
            module_list.extend([nn.Linear(dim, layer_dims[i + 1]), activation_fn])
        module_list.extend([nn.Linear(layer_dims[-1], output_dim), output_activation_fn])
        self.sequential = nn.Sequential(*module_list)

    def forward(self, data):
        # dimensions convention (..., channels, temporal_dim)
        x = data['x_value']
        model_output = {
            'x_encoded': self.sequential(x),
        }
        return model_output
