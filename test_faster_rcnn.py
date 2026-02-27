"""
References:
    - https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset
    - https://github.com/pytorch/vision/tree/main/references/detection
    - https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/detection/mean_ap.py#L50-L689
"""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import os

from rcnn_data import PotholeDataset

iou_thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
rec_thresholds = [0.25 + 0.01 * i for i in range(76)]

@torch.inference_mode()
def evaluate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(class_metrics=True, iou_thresholds=iou_thresholds, rec_thresholds=rec_thresholds)

    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        outputs = model(images)

        preds = []
        for output in outputs:
            preds.append({
                "boxes": output["boxes"].to(device),
                "scores": output["scores"].to(device),
                "labels": output["labels"].to(device)
            })

        target = []
        for t in targets:
            target.append({
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device)
            })

        metric.update(preds, target)

    results = metric.compute()
    return results


def collate_fn(batch):
    return tuple(zip(*batch))


results_dir = "faster_rcnn_results"
os.makedirs(results_dir, exist_ok=True)

num_classes = 2

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Map the saved weights to the current device
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
model.load_state_dict(torch.load(os.path.join(results_dir, "best_faster_rcnn.pth"), weights_only=True))
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

results = evaluate(model, test_loader, device)


with open(os.path.join(results_dir, "results.txt"), mode="w") as file:
    file.write("mAP50: {:.4f}\n".format(results["map_50"].item()))
    file.write("mAP50-95: {:.4f}\n".format(results["map"].item()))
    file.write("Precision: {:.4f}\n".format(results["map_per_class"].item()))
    file.write("Recall: {:.4f}\n".format(results["mar_100_per_class"].item()))


print("mAP50: {:.4f}".format(results["map_50"].item()))
print("mAP50-95: {:.4f}".format(results["map"].item()))
print("Precision: {:.4f}".format(results["map_per_class"].item()))
print("Recall: {:.4f}".format(results["mar_100_per_class"].item()))