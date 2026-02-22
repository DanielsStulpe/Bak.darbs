import torch
import torchvision
from torchvision.datasets import CocoDetection
from pycocotools import mask as coco_mask
import os
import numpy as np
from PIL import Image


class PotholeDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, annotations = super().__getitem__(idx)

        # Convert grayscale images to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")  # <-- Force 3 channels

        image_id = self.ids[idx]

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in annotations:
            xmin, ymin, width, height = ann["bbox"]
            xmax = xmin + width
            ymax = ymin + height

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann["iscrowd"])

            # For Mask R-CNN
            if "segmentation" in ann and ann["segmentation"]:
                seg = ann["segmentation"]

                if isinstance(seg, list):
                    # polygon format
                    rles = coco_mask.frPyObjects(seg, img.height, img.width)
                    rle = coco_mask.merge(rles)
                elif isinstance(seg, dict):
                    # already RLE
                    rle = seg
                else:
                    continue

                mask = coco_mask.decode(rle)
                masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        if masks:
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            target["masks"] = masks

        img = torchvision.transforms.functional.to_tensor(img)

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target