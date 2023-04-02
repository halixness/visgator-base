"""
    Ideas:
    1) Concatenate sinusoidal embeddings of bounding boxes to the img, project to text embedding size and maximize with contrastive learning.
    2)
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
        device=None
    ):
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.PILToTensor = T.ToTensor()
        self.TensorToPIL = T.ToPILImage()

        # Embedding size: 512
        self.CLIP, self.clip_preprocess = clip.load(clip_img_encoder, self.device)
        self.YOLO = torch.hub.load(yolo_repo, yolo_version, pretrained=True, verbose=False).to(self.device)

        # Add ReLU?
        self.bbox_proj = torch.nn.Linear(512 + 4, 512).to(self.device)

    def position_encoding(self, n_positions, embedding_dim):
        """ 
            Init the sinusoid position encoding table 
            https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/4f4a192f0fd272102c8852b00b1007dffd292b90/transformer/Models.py#L11
        """
        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2*i/embedding_dim) for i in range(embedding_dim)]
            if pos != 0 else np.zeros(embedding_dim) for pos in range(n_positions)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1

        return torch.from_numpy(position_enc).type(torch.FloatTensor)


    def get_bounding_boxes(self, image):
        """
            Get YOLO bounding boxes & labels
            image:     Tensor(N, w, h, c)
        """
        results = self.YOLO(image)
        return results.xyxy  # (N, top-K, xmin, ymin, xmax, ymax, conf, class)
    
    def contrastive_loss(self, similarity, i, j, tau = 1):
        """ 
            Very simple one-positive contrastive loss
            similarity:     a pre-computed distance matrix
            i:              reference element
            j:              positive example
        """
        row = torch.exp(similarity[i, :] / tau)
        return -torch.log(
            row[j] / (torch.sum(row) - row[j])
        )

    def forward(self, image, texts):
        """
            Given an image, extract objects and compute text scores
            image:     PillowImage
            texts:      Tensor(N, k, d)
        """

        # CLIP image features
        img = self.PILToTensor(image).to(self.device)
        # img = torch.permute(img, (1, 2, 0)) # c, h, w => h, w, c
        img_features = self.CLIP.encode_image(
            self.clip_preprocess(self.TensorToPIL(img)).unsqueeze(0).to(self.device)
        )
        img_features /= img_features.norm(dim=-1, keepdim=True) 

        # YOLO bbox detection
        #      xmin    ymin    xmax   ymax  confidence  class
        bounding_boxes = self.get_bounding_boxes(image)

        identified_objs = []
        for i, detected in enumerate(bounding_boxes[0]):
            xmin, ymin, xmax, ymax = detected[:4].int().tolist()
            identified_objs.append([xmin, ymin, xmax, ymax])
            
        # Extract CLIP scores
        with torch.no_grad():

            # For bbox => encoding vector
            bbox_embeddings = []
            for obj in identified_objs:

                # Positional encodings added to normalized bbox coordinates
                bbox = torch.Tensor([
                    xmin / image.width, 
                    xmax / image.width,
                    ymin / image.height,
                    ymax / image.height
                ]).type(torch.float16)

                bbox = (bbox + self.position_encoding(4, 1)[:, 0]).to(self.device)
                
                # Concatenate to content img and project to multimodal embedding space
                bbox_embedding = self.bbox_proj(
                    torch.cat((img_features, bbox.unsqueeze(0)), dim = 1)
                ).to(self.device)
                
                bbox_embeddings.append(bbox_embedding)
                
            bbox_embeddings = torch.cat(bbox_embeddings).type(torch.float16)

            # Text
            text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(self.device)
            text_features = self.CLIP.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # returning list(bboxes), (N, K) N = # bboxes, K = # captions
        similarity = (100.0 * bbox_embeddings @ text_features.T).softmax(dim=-1)
        return identified_objs, similarity