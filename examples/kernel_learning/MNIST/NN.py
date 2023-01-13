
import torch

class MNIST_feature_map(torch.nn.Module):
    image_dim = 28
    def __init__(self, out_features=32):
        super(MNIST_feature_map, self).__init__()
        self.num_features = out_features
        self.conv1 = torch.nn.Sequential(         
            torch.nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            torch.nn.ReLU(),                      
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(16) 
        )
        self.conv2 = torch.nn.Sequential(         
            torch.nn.Conv2d(16, 32, 5, 1, 2),     
            torch.nn.ReLU(),                      
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(32)                
        )
        # fully connected layer, output dim num_features
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(32 * 7 * 7, self.num_features - 1), #Out layer has num_features - 1 as the mapping to the sphere make us gain a dimension.
            torch.nn.BatchNorm1d(self.num_features - 1)
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, MNIST_feature_map.image_dim, MNIST_feature_map.image_dim)) # Expect flattened data of shape [n_samples, image_dim**2]
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), - 1)       
        output = self.lin(x)