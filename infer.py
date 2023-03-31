from architectures.baseline.yoloclip import YOLOClip
import torch
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO

img = Image.open("dogs.jpg")

model = YOLOClip()

bboxes = model(img, ["the white dog on the left", "the brown dog on the right",])
print(bboxes)