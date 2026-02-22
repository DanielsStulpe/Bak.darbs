import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from rcnn_data import PotholeDataset


# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


def collate_fn(batch):
    return tuple(zip(*batch))

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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.00001,
    # momentum=0.9,
    weight_decay=0.0005
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

num_epochs = 100

from torchmetrics.detection.mean_ap import MeanAveragePrecision

def evaluate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision()

    with torch.no_grad():
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

            targets_cpu = []
            for t in targets:
                targets_cpu.append({
                    "boxes": t["boxes"].to(device),
                    "labels": t["labels"].to(device)
                })

            metric.update(preds, targets_cpu)

    results = metric.compute()
    return results


best_map = 0.0

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

    print(f"Epoch {epoch} Training Loss: {total_loss}")

    # VALIDATION STEP
    results = evaluate(model, val_loader, device)

    map50 = results["map_50"].item()
    map5095 = results["map"].item()

    print(f"Validation mAP@0.5: {map50:.4f}")
    print(f"Validation mAP@0.5:0.95: {map5095:.4f}")

    # Save best model based on mAP@0.5
    if map50 > best_map:
        best_map = map50
        torch.save(model.state_dict(), "best_fasterrcnn.pth")
        print("Best model (by mAP@0.5) saved!")