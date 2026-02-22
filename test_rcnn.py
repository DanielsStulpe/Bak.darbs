import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from rcnn_data import PotholeDataset

def evaluate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision()

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]

            # Get predictions
            outputs = model(images)

            # Move predictions and targets to CPU for metric
            preds = []
            for output in outputs:
                preds.append({
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                    "labels": output["labels"].cpu()
                })

            targets_cpu = []
            for t in targets:
                targets_cpu.append({
                    "boxes": t["boxes"],
                    "labels": t["labels"]
                })

            metric.update(preds, targets_cpu)

    results = metric.compute()
    return results

def collate_fn(batch):
    return tuple(zip(*batch))


num_classes = 2

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Map the saved weights to the current device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.load_state_dict(torch.load("best_fasterrcnn.pth", map_location=device))
model.to(device)

val_dataset = PotholeDataset(
    img_folder="roboflow_dataset_coco/valid",
    ann_file="roboflow_dataset_coco/valid/_annotations.coco.json"
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)

results = evaluate(model, val_loader, device)

print("mAP@0.5: {:.4f}".format(results["map_50"].item()))
print("mAP@0.5:0.95: {:.4f}".format(results["map"].item()))
print("Precision: {:.4f}".format(results["precision"].item()))
print("Recall: {:.4f}".format(results["recall"].item()))