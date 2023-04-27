"""
    Ideas:
    1) 
    2)
"""

import os
import clip
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T


class CLIPDetector(torch.nn.Module):
    def __init__(
        self, 
        clip_img_encoder='ViT-B/32', 
        device=None
    ):
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else: self.device = device

        self.PILToTensor = T.ToTensor()
        self.TensorToPIL = T.ToPILImage()

        # Embedding size: 512
        self.CLIP, self.clip_preprocess = clip.load(clip_img_encoder, self.device)
        self.CLIP = self.CLIP.type(torch.float32)

        # Freezing params
        self.CLIP.requires_grad = False
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.bbox_proj = torch.nn.Linear(512 * 2, 4).type(self.CLIP.dtype).to(self.device)
        self.activation = torch.nn.ReLU().type(self.CLIP.dtype).to(self.device)


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


    def contrastive_loss(self, similarity, i, j, tau = 1, dim=0):
        """ 
            Very simple one-positive contrastive loss
            similarity:     a pre-computed distance matrix
            i:              reference element
            j:              positive example
            tau:            temperature param
            dim:            0 = row (bbox -> captions), 1 = column (caption -> bboxes)
        """
        if dim == 1:
            similarity = torch.permute(similarity, (1, 0))

        row = torch.exp(similarity[i, :] / tau)
        return -torch.log(
            row[j] / (torch.sum(row) - row[j])
        )
    

    def inference(self, imgs, captions):
        with torch.no_grad():
            sents = torch.stack([clip.tokenize(s) for s in captions])[:,0,:]

            image_features = self.CLIP.encode_image(imgs.to(self.device))
            text_features = self.CLIP.encode_text(sents.to(self.device))

            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) # 16, 512
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) # 16, 512

            # 16, 4
            x = torch.cat((image_features, text_features), dim = 1)
            bboxes = self.activation(
                self.bbox_proj(x)
            )
        
            # Rescaling
            rescaled_bboxes = torch.zeros_like(bboxes).to(self.device)
            for i, bbox in enumerate(bboxes):
                xmin, xmax, ymin, ymax = bbox # normalized bbox
                xmin = (xmin * imgs[0].shape[2]).to(torch.int32)
                xmax = (xmax * imgs[0].shape[2]).to(torch.int32)
                ymin = (xmax * imgs[0].shape[1]).to(torch.int32)
                ymax = (xmax * imgs[0].shape[1]).to(torch.int32)
                rescaled_bboxes[i] = torch.Tensor([xmin, xmax, ymin, ymax]).to(self.device)
            
        return rescaled_bboxes


    def forward(self, imgs, captions):
        
        sents = torch.stack([clip.tokenize(s) for s in captions])[:,0,:].to(self.device)
        preprocessed_imgs = torch.stack([self.clip_preprocess(self.TensorToPIL(i)) for i in imgs]).to(self.device)

        image_features = self.CLIP.encode_image(preprocessed_imgs)
        text_features = self.CLIP.encode_text(sents)

        # normalize inputs
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # 16, 512
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # 16, 512

        # 16, 4
        x = torch.cat((image_features, text_features), dim = 1)
        bboxes = self.activation(
            self.bbox_proj(x)
        )
        
        return bboxes

        # 16, c, h, w
        """
        masked_imgs = torch.zeros_like(imgs).to(self.device)
        for i, bbox in enumerate(bboxes):
            xmin, xmax, ymin, ymax = bbox # normalized bbox
            xmin = (xmin * imgs[0].shape[2]).to(torch.int32)
            xmax = (xmax * imgs[0].shape[2]).to(torch.int32)
            ymin = (xmax * imgs[0].shape[1]).to(torch.int32)
            ymax = (xmax * imgs[0].shape[1]).to(torch.int32)
            masked_imgs[i, :, ymin:ymax, xmin:xmax] = imgs[i, :, ymin:ymax, xmin:xmax]

        masked_features = self.CLIP.encode_image(masked_imgs)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * masked_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ masked_features.t()

        return logits_per_image, logits_per_text
        """
