
import torch
from kornia.enhance import ZCAWhitening, linear_transform

class MNIST_feature_map(torch.nn.Module):
    image_dim = 28
    def __init__(self, out_features=32, momentum = 0.1):
        super(MNIST_feature_map, self).__init__()
        self.num_features = out_features
        self.momentum = momentum
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
        self.lin = torch.nn.Linear(32 * 7 * 7, self.num_features)
        #Batch withening
        self.batch_whitening = ZCAWhitening(detach_transforms = True)

        
        self.register_buffer('running_mean', torch.zeros(self.num_features))
        self.register_buffer('running_covar', torch.eye(self.num_features))

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, MNIST_feature_map.image_dim, MNIST_feature_map.image_dim)) # Expect flattened data of shape [n_samples, image_dim**2]
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.lin(x.view(x.size(0), - 1))
        if self.training:
            self.batch_whitening.fit(x)
            self.running_mean = (1 - self.momentum)*self.running_mean + self.momentum*self.batch_whitening.mean_vector
            self.running_covar = (1 - self.momentum)*self.running_mean + self.momentum*self.batch_whitening.transform_matrix
        return linear_transform(x, self.running_covar, self.running_mean)
