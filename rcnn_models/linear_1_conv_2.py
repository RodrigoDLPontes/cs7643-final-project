import torch.nn as nn

class BackboneTail(nn.Module):

    def __init__(self):
        super(BackboneTail, self).__init__()
        self.conv1 = nn.Conv2d(1024, 256, 3)
        self.conv2 = nn.Conv2d(256, 64, 3)
        self.linear = nn.Linear(64 * 5 * 10, 7)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, act_maps):
        '''
        Run mini-batch of activation maps through model.

        Args:
            act_maps (Tensor): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the activation map height
                W is the activation map width

        Returns:
            A Tensor of size (N, num_labels) specifying the score
            for each example and a certain number of cars.
        '''
        N = act_maps.shape[0]
        output = self.relu(self.conv1(act_maps))
        output = self.maxpool(self.relu(self.conv2(output)))
        output = self.linear(output.view(N, -1))
        return output