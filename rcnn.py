import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo as zoo
from detectron2.engine import DefaultPredictor

class RCNN():

    def __init__(self, rcnn_type="M50"):
        types = {
            "M50": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
            "F50": "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
        }
        cfg = get_cfg()
        cfg.merge_from_file(zoo.get_config_file(types[rcnn_type]))
        cfg.MODEL.WEIGHTS = zoo.get_checkpoint_url(types[rcnn_type])
        self.predictor = DefaultPredictor(cfg)
        
    def forward(self, image):
        with torch.no_grad():
            output = self.predictor(image)
        return torch.sum(output['instances'].pred_classes == 2).cpu().item()
