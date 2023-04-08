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
    
    def encode_captions(self, texts):
        """
            Batch of sentences -> batch of CLIP sent embeddings
            texts:     List[String]
        """
        return torch.stack(
            [self.CLIP.encode_text(clip.tokenize(t).to(self.device)) for t in texts]
        ).to(self.device)[:, 0, :]
    

    def encode_image_caption(self, image, text):
        """
            Given an image and a caption, compute the bounding box and the img embedding (for contrastive loss)
            image:     Tensor(c, h, w)
            texts:     String
        """

        # CLIP image features
        image = image.type(self.CLIP.dtype)
        img_features = self.CLIP.encode_image(image)[0]
        img_features /= img_features.norm(dim=-1, keepdim=True)

        txt_features = self.CLIP.encode_text(
            clip.tokenize(text).to(self.device)
        )[0]
        txt_features /= txt_features.norm(dim=-1, keepdim=True)

        x = torch.cat((img_features, txt_features))
        x = self.activation(
            self.bbox_proj(x)
        )

        # Mask the input with the predicted bbox
        xmin, xmax, ymin, ymax = x.type(torch.int16)
        embedding = torch.zeros_like(image)
        embedding[:, :, ymin:ymax, xmin:xmax] = image[:, :, ymin:ymax, xmin:xmax]

        return x, self.CLIP.encode_image(embedding)