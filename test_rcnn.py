import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from rcnn_data import PotholeDataset

def evaluate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(class_metrics=True)

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


def compute_yolo_mp_mr(model, dataloader, device, iou_threshold=0.5, conf_threshold=0.001):
    model.eval()

    all_stats = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for pred, target in zip(outputs, targets):
                pred_boxes = pred["boxes"].cpu()
                pred_scores = pred["scores"].cpu()
                pred_labels = pred["labels"].cpu()

                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()

                # Filter by confidence threshold
                keep = pred_scores >= conf_threshold
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                correct = torch.zeros(len(pred_boxes))

                if len(gt_boxes) and len(pred_boxes):
                    ious = box_iou(pred_boxes, gt_boxes)
                    max_iou, max_idx = ious.max(1)

                    for i in range(len(pred_boxes)):
                        if max_iou[i] >= iou_threshold and pred_labels[i] == gt_labels[max_idx[i]]:
                            correct[i] = 1

                all_stats.append((
                    correct,
                    torch.ones(len(pred_boxes)),  # all predictions counted
                    gt_labels
                ))

    # Concatenate stats
    correct = torch.cat([x[0] for x in all_stats])
    pred_total = torch.cat([x[1] for x in all_stats])
    gt_total = torch.cat([torch.ones(len(x[2])) for x in all_stats])

    TP = correct.sum().item()
    FP = pred_total.sum().item() - TP
    FN = gt_total.sum().item() - TP

    mp = TP / (TP + FP + 1e-16)
    mr = TP / (TP + FN + 1e-16)

    return mp, mr


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

# results = evaluate(model, val_loader, device)

mp, mr = compute_yolo_mp_mr(model, val_loader, device)

# print("mAP50: {:.4f}".format(results["map_50"].item()))
# print("mAP50-95: {:.4f}".format(results["map"].item()))
print("Precision:", round(mp, 4))
print("Recall:", round(mr, 4))