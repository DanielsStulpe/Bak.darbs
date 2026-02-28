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
import time
import os

from rcnn_data import PotholeDataset


RESULTS_DIR = "fcos_results"
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
    with open(os.path.join(RESULTS_DIR, "fcos_predicts.json"), "w") as f:
        json.dump(predicts, f)

    # Load COCO GT
    coco_gt = COCO(ann_file)

    # Load detections
    coco_dt = coco_gt.loadRes(os.path.join(RESULTS_DIR, "fcos_predicts.json"))

    # Evaluate
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats


def collate_fn(batch):
    return tuple(zip(*batch))


# load a model pre-trained on COCO
model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")


train_dataset = PotholeDataset(
    img_folder="roboflow_dataset_coco/train",
    ann_file="roboflow_dataset_coco/train/_annotations.coco.json"
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

val_dataset = PotholeDataset(
    img_folder="roboflow_dataset_coco/valid",
    ann_file="roboflow_dataset_coco/valid/_annotations.coco.json"
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn
)

val_ann_file = "roboflow_dataset_coco/valid/_annotations.coco.json"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)


num_epochs = 100

best_map = 0.0

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    lr_scheduler.step()

    print(f"========== Epoch {epoch} Training Loss: {total_loss} ==========")

    results = evaluate(model, val_loader, device, val_ann_file)

    map5095 = results[0]
    map50 = results[1]

    print(f"Validation mAP@0.5: {map50:.4f}")
    print(f"Validation mAP@0.5:0.95: {map5095:.4f}")

    # Save best model based on mAP@0.5
    if map50 > best_map:
        best_map = map50
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_fcos.pth"))
        print("Best model (by mAP@0.5) saved!")


training_time = time.time() - start_time
model_size_mb = os.path.getsize(os.path.join(RESULTS_DIR, "best_fcos.pth")) / (1024 * 1024)
params = sum(p.numel() for p in model.parameters()) / 1e6


with open(os.path.join(RESULTS_DIR, "results.txt"), mode="w") as file:
    file.write(f"Total training time: {training_time:.2f} seconds\n")
    file.write(f"Model size: {model_size_mb:.2f} MB\n")
    file.write(f"Total parameters: {params:.2f}M\n")


print(f"Total training time: {training_time:.2f} seconds")
print(f"Model size: {model_size_mb:.2f} MB")
print(f"Total parameters: {params:.2f}M")
