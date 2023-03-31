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

class YOLOClip(torch.nn.Module):

    def __init__(
        self, 
        clip_img_encoder='ViT-B/32', 
        yolo_repo='ultralytics/yolov5', 
        yolo_version='yolov5s', 
        device=None
    ):
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.CLIP, self.clip_preprocess = clip.load(clip_img_encoder, self.device)
        self.YOLO = torch.hub.load(yolo_repo, yolo_version, pretrained=True, verbose=False).to(self.device)

    def get_bounding_boxes(self, images):
        """
            Get YOLO bounding boxes & labels
            images:     Tensor(N, w, h, c)
        """
        results = self.YOLO(images)
        return results.xyxy  # (N, top-K, xmin, ymin, xmax, ymax, conf, class)
    

    def forward(self, images, texts):
        """
            Given an image, extract objects and compute text scores
            images:     PillowImage
            texts:      Tensor(N, k, d)
        """
        #      xmin    ymin    xmax   ymax  confidence  class
        bounding_boxes = self.get_bounding_boxes(images)
        
        PILToTensor = T.ToTensor()
        TensorToPIL = T.ToPILImage()
        img = PILToTensor(images).to(self.device)
        # img = torch.permute(img, (1, 2, 0)) # c, h, w => h, w, c
        
        identified_objs = []
        
        # Masked obj imgs
        for i, detected in enumerate(bounding_boxes[0]):

            # extract bounding box
            xmin, ymin, xmax, ymax = detected[:4].int().tolist()
            confidence = detected[4].double()

            # c, h, w
            mask = torch.ones_like(img).to(self.device)            
            mask[:, :, :] = 0
            mask[:, ymin:ymax, xmin:xmax] = confidence

            detection_img = torch.mul(img / 255, mask) * 255

            # Pillow Img, bbox list
            identified_objs.append((TensorToPIL(detection_img), detected[:4].int().tolist()))
            plt.imsave(f"img_{i}.png", torch.permute(detection_img, (1,2,0)).cpu().numpy())
            
        # Extract CLIP scores
        with torch.no_grad():

            # Images
            obj_features = []
            for obj in identified_objs:
                features = self.CLIP.encode_image(
                    self.clip_preprocess(obj[0]).unsqueeze(0).to(self.device)
                )
                features /= features.norm(dim=-1, keepdim=True)
                obj_features.append(features)
                
            obj_features = torch.cat(obj_features)

            # Text
            text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(self.device)
            text_features = self.CLIP.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * obj_features @ text_features.T).softmax(dim=-1)

        # For each caption, pick best bbox
        results = []
        for i in range(len(texts)):
            idx = torch.argmax(similarity[:, i])
            results.append(identified_objs[idx][1])

            print(f"{texts[i]}: \t img_{idx} \t {similarity[idx, i]}")

        return results