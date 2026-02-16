import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow
# from IPython import display
import PIL
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog


from detectron2.data.datasets import register_coco_instances
register_coco_instances("roboflow_dataset_train", {}, "roboflow_dataset_detectron2/train/_annotations.coco.json", "roboflow_dataset_detectron2/train")
register_coco_instances("roboflow_dataset_val", {}, "roboflow_dataset_detectron2/valid/_annotations.coco.json", "roboflow_dataset_detectron2/valid")
register_coco_instances("roboflow_dataset_test", {}, "roboflow_dataset_detectron2/test/_annotations.coco.json", "roboflow_dataset_detectron2/test")


from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("roboflow_dataset_train",)
cfg.DATASETS.TEST = ("roboflow_dataset_val",)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# uncomment below to train
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
