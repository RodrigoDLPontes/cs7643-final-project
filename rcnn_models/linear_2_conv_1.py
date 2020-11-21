import torch.nn as nn

class BackboneTail(nn.Module):

    def __init__(self):
        super(BackboneTail, self).__init__()
        self.conv = nn.Conv2d(1024, 64, 3)
        self.linear1 = nn.Linear(64 * 6 * 11, 256)
        self.linear2 = nn.Linear(256, 7)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, act_maps):
        '''
        Run mini-batch of images through model.

        Args:
            images (Tensor): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A Tensor of size (N, num_labels) specifying the score
            for each example and a certain number of cars.
        '''
        N = act_maps.shape[0]
        output = self.maxpool(self.relu(self.conv(act_maps)))
        output = self.relu(self.linear1(output.view(N, -1)))
        output = self.linear2(output)
        return output