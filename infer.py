from architectures.baseline.yoloclip import YOLOClip
import torch
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO

img = Image.open("dogs.jpg")

model = YOLOClip()

bboxes = model(img, ["the dog on the left", "the on the right",])
print(bboxes)