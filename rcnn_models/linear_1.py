import torch.nn as nn

class BackboneTail(nn.Module):

    def __init__(self):
        super(BackboneTail, self).__init__()
        self.linear = nn.Linear(1024 * 14 * 24, 7)
        
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
        output = self.linear(act_maps.view(N, -1))
        return output