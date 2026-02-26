import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import os
import time

from rcnn_data import PotholeDataset


results_dir = "fcos_results"
os.makedirs(results_dir, exist_ok=True)


# load a model pre-trained on COCO
model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.0025,
    # momentum=0.9,
    weight_decay=0.0005
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)

num_epochs = 100


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
        torch.save(model.state_dict(), os.path.join(results_dir, "best_fcos.pth"))
        print("Best model (by mAP@0.5) saved!")


training_time = time.time() - start_time
model_size_mb = os.path.getsize(os.path.join(results_dir, "best_fcos.pth")) / (1024 * 1024)
params = sum(p.numel() for p in model.parameters()) / 1e6


with open(os.path.join(results_dir, "results.txt"), mode="w") as file:
    file.write(f"Total training time: {training_time:.2f} seconds\n")
    file.write(f"Model size: {model_size_mb:.2f} MB\n")
    file.write(f"Total parameters: {params:.2f}M\n")


print(f"Total training time: {training_time:.2f} seconds")
print(f"Model size: {model_size_mb:.2f} MB")
print(f"Total parameters: {params:.2f}M")
