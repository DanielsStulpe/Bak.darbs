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


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


import pickle
with open("cfg.pkl", "wb") as f:
    pickle.dump(cfg, f)

my_dataset_test_metadata = MetadataCatalog.get("physics_train")
from detectron2.utils.visualizer import ColorMode
dataset_dicts = DatasetCatalog.get("physics_test")
for d in random.sample(dataset_dicts, 5):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=my_dataset_test_metadata, 
                   scale=0.5, 
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("physics_test", ("bbox", "segm"), False, output_dir="./output/")
test_loader = build_detection_test_loader(cfg, "physics_test")
inference_on_dataset(trainer.model, test_loader, evaluator)
# another equivalent way to evaluate the model is to use `trainer.test`