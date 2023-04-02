from architectures.clip_detector.yoloclip import YOLOClip
import torch
from PIL import Image
import requests
from io import BytesIO

img = Image.open("dogs.jpg")

model = YOLOClip()

identified_objs, scores = model(img, ["the white dog on the left", "the brown dog on the right",])

print(model.contrastive_loss(scores, 0, 1))