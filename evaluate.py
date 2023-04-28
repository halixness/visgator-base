from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse 

from dataset import RefCocoBatch, RefCocoConfig, RefCocoDataset
from model.baseline.yoloclip import YOLOClip
from util.metrics import get_iou
from util.parallel import setup, cleanup 

# TODO: set DataParallel

print("[-] Loading the dataset...")

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="baseline", type=str, help="Model to test (default: baseline)")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
parser.add_argument("--dataset", default=None, type=str, required=True, help="Path to RefCocoG dataset.")
parser.add_argument("--device", default="cuda", type=str, help="Device to use")

args = parser.parse_args()

# Loading data
cfg = RefCocoConfig({
    "path": args.dataset
})

train_dataset = RefCocoDataset(config = cfg, phase = "train")
test_dataset = RefCocoDataset(config = cfg, phase = "test")
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=RefCocoDataset.batchify)

print("[-] Loading the model...")

# Loading model
if args.model == "baseline":
    model = YOLOClip(device=args.device)
else:
    raise Exception("No valid model selected.")

print("[-] Starting the evaluation...")

model_iou = []
for batch in tqdm(train_dataloader):
    
    bboxes = model(batch.images, batch.sentences)

    if len(bboxes) == 0:
        model_iou.append(0)  
    else:
        for i, obj in enumerate(bboxes):
            
            if obj is not None:
                _, img_height, img_width = batch.images[i].shape
                
                # Pred
                xmin, ymin, xmax, ymax = obj[0][:4].cpu().numpy()
                width = xmax - xmin
                height = ymax - ymin
                xcenter = xmin + width/2
                ycenter = ymin + height/2

                pred_box = {"x1": xmin, "x2": xmax, "y1": ymin, "y2": ymax}
                
                true_box = {"x1": batch.bboxes[i][0], "x2": batch.bboxes[i][0] + batch.bboxes[i][2], "y1": batch.bboxes[i][1], "y2": batch.bboxes[i][1] + batch.bboxes[i][3]}

                # Ground
                width = true_box["x2"] - true_box["x1"]
                height = true_box["y2"] - true_box["y1"]
                xcenter = true_box["x1"] + width/2
                ycenter = true_box["y1"] + height/2
                
                # Score & plot
                iou = get_iou(pred_box, true_box)
                model_iou.append(iou)

print(f"Average IoU for model \"{args.model}\": {np.mean(model_iou)}")
            