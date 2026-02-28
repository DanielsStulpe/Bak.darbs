"""
References:
    - https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset
    - https://github.com/pytorch/vision/tree/main/references/detection
    - https://cocodataset.org/#detection-eval
    - https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools
    - https://mmdetection.readthedocs.io/en/v2.10.0/_modules/mmdet/datasets/coco.html
"""

import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import json
import os

from rcnn_data import PotholeDataset


RESULTS_DIR = "retinanet_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


@torch.inference_mode()
def evaluate(model, data_loader, device, ann_file):
    model.eval()
    predicts = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())

            boxes = output["boxes"].to(device)
            scores = output["scores"].to(device)
            labels = output["labels"].to(device)

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                predicts.append({
                    "image_id": image_id,
                    "category_id": int(label.item()),
                    "bbox": [
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1
                    ],
                    "score": float(score.item())
                })

    # Save detections
    with open(os.path.join(RESULTS_DIR, "retinanet_test_predicts.json"), "w") as f:
        json.dump(predicts, f)

    # Load COCO GT
    coco_gt = COCO(ann_file)

    # Load detections
    coco_dt = coco_gt.loadRes(os.path.join(RESULTS_DIR, "retinanet_test_predicts.json"))

    # Evaluate
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats


def collate_fn(batch):
    return tuple(zip(*batch))


num_classes = 2

model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None)

# Map the saved weights to the current device
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_retinanet.pth"), weights_only=True))
model.to(device)

test_dataset = PotholeDataset(
    img_folder="roboflow_dataset_coco/test",
    ann_file="roboflow_dataset_coco/test/_annotations.coco.json"
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

ann_file = "roboflow_dataset_coco/test/_annotations.coco.json"
results = evaluate(model, test_loader, device, ann_file)


with open(os.path.join(RESULTS_DIR, "results.txt"), mode="w") as file:
    file.write("mAP50: {:.4f}\n".format(results[0]))
    file.write("mAP50-95: {:.4f}\n".format(results[1]))
    file.write("Recall: {:.4f}\n".format(results[8]))


print("mAP50: {:.4f}".format(results[0]))
print("mAP50-95: {:.4f}".format(results[1]))
print("Recall: {:.4f}".format(results[8]))