from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse 
import os

from dataset import RefCocoBatch, RefCocoConfig, RefCocoDataset
from model.clip_detector.detector import CLIPDetector
from util.metrics import get_iou
from util.parallel import setup, cleanup 

import torch
import torch.optim as optim

# TODO: set DataParallel

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="clip_detector", type=str, help="Model to test (default: baseline)")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
parser.add_argument("--dataset", default=None, type=str, required=True, help="Path to RefCocoG dataset.")
parser.add_argument("--device", default="cuda", type=str, help="Device to use")
parser.add_argument("--checkpoint_dir", default=".", type=str, help="Path to store the model checkpoint")

args = parser.parse_args()

print("[-] Loading the dataset...")

# Loading data
cfg = RefCocoConfig({
    "path": args.dataset
})

train_dataset = RefCocoDataset(config = cfg, phase = "train")
test_dataset = RefCocoDataset(config = cfg, phase = "test")
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=RefCocoDataset.batchify)

print("[-] Loading the model...")

# Loading model
if args.model == "clip_detector":
    model = CLIPDetector(device=args.device)
else:
    raise Exception("No valid model selected.")

print("[-] Starting to train...")

# Training
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
loss_fn = torch.nn.HuberLoss()
import numpy as np 

# per epoch: 11 mins on a single RTX 3070 Laptop
epochs = 1
overall_loss = []

for epoch in range(epochs):

    epoch_loss = []
    for batch in tqdm(train_dataloader):
    
        optimizer.zero_grad()
        
        # logits_img, logits_txt = model(batch.images, batch.sentences)
        # loss = loss_fn(logits_img)
        
        bboxes = model(batch.images, batch.sentences)
        loss = loss_fn(bboxes, batch.bboxes.to(args.device))

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.detach().cpu().numpy())
        overall_loss.append(loss.detach().cpu().numpy())
        
    print(f"Epoch loss: \t{np.mean(epoch_loss)}\n")
    torch.save(model.state_dict, os.path.join(args.checkpoint_dir, "model.pt"))
