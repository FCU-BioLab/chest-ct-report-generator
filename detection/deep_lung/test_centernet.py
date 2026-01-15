#!/usr/bin/env python3
import torch
from pathlib import Path
from detection.deep_lung.model import get_model
from detection.deep_lung.dataset import LungNodule3DDataset
from torch.utils.data import DataLoader

def collate_fn(batch):
    return tuple(zip(*batch))

device = 'cuda'
dataset = LungNodule3DDataset(Path('cache/deep_lung_cache/train'), split='train', augment=False)
loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

model = get_model().to(device).train()

# Get one batch
images, targets = next(iter(loader))
images = torch.stack(images).to(device)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

print("Targets:", [len(t['boxes']) for t in targets])

# Forward pass
loss_dict = model(images, targets)
print("Loss dict:")
for k, v in loss_dict.items():
    print(f"  {k}: {v.item() if hasattr(v, 'item') else v}")

# Eval mode
model.eval()
with torch.no_grad():
    detections = model(images)
    print("Detections:", [len(d['boxes']) for d in detections])
