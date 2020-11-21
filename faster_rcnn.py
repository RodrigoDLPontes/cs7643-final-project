import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo as zoo
from detectron2.engine import DefaultPredictor

class FasterRCNN():

    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
        cfg.MODEL.WEIGHTS = zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
        self.predictor = DefaultPredictor(cfg)
        
    def forward(self, image):
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
        with torch.no_grad():
            output = self.predictor(image)
        return torch.sum(output['instances'].pred_classes == 2).cpu().item()
