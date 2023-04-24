"""
    Ideas:
    1) Fine-tuning CLIP by matching referential captions (e.g. dog on the left) with masked images (e.g. outside the area with the dog on the left detected by YOLO it's all black)
    2) Same but without masking
"""

import os
import clip
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

class YOLOClip(torch.nn.Module):

    def __init__(
        self, 
        clip_img_encoder='ViT-B/32', 
        yolo_repo='ultralytics/yolov5', 
        yolo_version='yolov5s', 
        device=None,
        CONF_THRESHOLD=.5
    ):
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.CONF_THRESHOLD = CONF_THRESHOLD

        self.CLIP, self.clip_preprocess = clip.load(clip_img_encoder, self.device)
        self.YOLO = torch.hub.load(yolo_repo, yolo_version, pretrained=True, verbose=False).to(self.device)

    def get_bounding_boxes(self, images):
        """
            Get YOLO bounding boxes & labels
            images:     Tensor(N, w, h, c)
        """
        # (N, top-K, xmin, ymin, xmax, ymax, conf, class)
        return [self.YOLO(img).xyxy[0] for img in images]  
    

    def forward(self, images, texts):
        """
            From images, choose the YOLO-extracted bbox that best matches texts with CLIP score 
            images:         [PillowImage]
            texts:          [str]
            
            bbox_results:   [(xmin, xmax, ymin, ymax, confidence, val),(...),...]
        """
        PILToTensor = T.ToTensor()
        TensorToPIL = T.ToPILImage()

        # YOLO bulk processing all images
        #      xmin    ymin    xmax   ymax  confidence  class
        yolo_images = [TensorToPIL(i) for i in images]
        bounding_boxes = self.get_bounding_boxes(yolo_images)
        bbox_results = []

        with torch.no_grad():
        
            # For each image
            for i, image in enumerate(images):

                # For each bbox in img
                objects = []
                for j, detected in enumerate(bounding_boxes[i]):

                    # BBox & confidence
                    xmin, ymin, xmax, ymax = detected[:4].int().tolist()
                    confidence = detected[4].double()

                    if confidence > self.CONF_THRESHOLD:                            
                        # Crop the img to get bbox data
                        # c, h, w
                        detected_img = image[:, ymin:ymax, xmin:xmax]
                        objects.append(self.clip_preprocess(TensorToPIL(detected_img)).to(self.device))

                # If any obj detected
                if len(objects) > 0:

                    objects = torch.stack(objects).to(self.device)
                    objects_features = self.CLIP.encode_image(objects) # (K, 512)
            
                    text_features = self.CLIP.encode_text(clip.tokenize(texts[i]).to(self.device)) # (1, 512)

                    # Compute bboxes-caption score and return the best one
                    similarity = (100.0 * objects_features @ text_features.T)
                    idx = torch.argmax(similarity).cpu().numpy()

                    # First is the predicted one
                    final_boxes = [bounding_boxes[i][idx]] + [b for i, b in enumerate(bounding_boxes[i]) if i != idx]
                    bbox_results.append(final_boxes)

                else:
                    bbox_results.append(None)

            return bbox_results