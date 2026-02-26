import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from rcnn_data import PotholeDataset

def evaluate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(extended_summary=True)

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


def compute_yolo_style_pr(model, data_loader, device, conf_thresh=0.001, iou_thresh=0.7):
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                preds_boxes = output["boxes"]
                preds_scores = output["scores"]
                preds_labels = output["labels"]

                gt_boxes = target["boxes"].to(device)
                gt_labels = target["labels"].to(device)

                # Filter by confidence threshold
                keep = preds_scores >= conf_thresh
                preds_boxes = preds_boxes[keep]
                preds_labels = preds_labels[keep]

                matched_gt = []

                if len(preds_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(preds_boxes, gt_boxes)

                    for pred_idx in range(len(preds_boxes)):
                        max_iou, gt_idx = ious[pred_idx].max(0)

                        if max_iou >= iou_thresh and gt_idx.item() not in matched_gt:
                            total_tp += 1
                            matched_gt.append(gt_idx.item())
                        else:
                            total_fp += 1

                    total_fn += len(gt_boxes) - len(matched_gt)

                else:
                    total_fp += len(preds_boxes)
                    total_fn += len(gt_boxes)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)

    return precision, recall


def collate_fn(batch):
    return tuple(zip(*batch))


num_classes = 2

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Map the saved weights to the current device
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
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

print("mAP50: {:.4f}".format(results["map_50"].item()))
print("mAP50-95: {:.4f}".format(results["map"].item()))

mp, mr = compute_yolo_style_pr(model, val_loader, device, conf_thresh=0.25)

print("Precision (YOLO-style): {:.4f}".format(mp))
print("Recall (YOLO-style): {:.4f}".format(mr))