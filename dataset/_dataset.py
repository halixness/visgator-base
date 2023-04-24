##
##
##

import json
import pickle
from typing import Literal, List

import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.utils import data
from transformers import CLIPProcessor

from ._config import RefCocoConfig
from ._utils import RefCocoBatch, RefCocoBatchSample, RefCocoSample


class RefCocoDataset(data.Dataset[RefCocoBatchSample]):
    _samples: List[RefCocoSample]
    _negative_sentences: int
    _processor: CLIPProcessor

    def __init__(
        self, config: RefCocoConfig, phase: Literal["train", "val", "test"]
    ) -> None:
        super().__init__()

        self._processor = CLIPProcessor.from_pretrained(config.preprocessing)
        self._negative_sentences = config.negative_sentences
        self._samples = self._get_samples(config, phase)

    def _get_samples(
        self, config: RefCocoConfig, phase: Literal["train", "val", "test"]
    ) -> List[RefCocoSample]:
        refs_path = config.path / f"annotations/refs({config.split}).p"
        instances_path = config.path / "annotations/instances.json"
        images_path = config.path / "images"

        info = {}
        with open(refs_path, "rb") as pf, open(instances_path, "r") as jf:
            refs = pickle.load(pf)
            instances = json.load(jf)

        for ref in refs:
            sentences = [sent["raw"] for sent in ref["sentences"]]
            info[ref["image_id"]] = {"sentences": sentences}

        for annotation in instances["annotations"]:
            info[annotation["image_id"]]["bbox"] = annotation["bbox"]

        for image in instances["images"]:
            info[image["id"]]["path"] = images_path / image["file_name"]

        samples = []

        for sample_info in info.values():
            if ref["split"] != phase:
                continue

            sample = RefCocoSample(
                path=sample_info["path"],  # type: ignore
                bbox=sample_info["bbox"],
                sentences=sample_info["sentences"],
            )

            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> RefCocoBatchSample:
        sample = self._samples[index]
        img = torchvision.io.read_image(str(sample.path))

        # preventing 1 channel images => stick to RGB
        if img.size(0) == 1: img = img.repeat(3, 1, 1)

        res = self._processor(images=img, return_tensors="pt", padding=True)
        image: Tensor = res["pixel_values"].squeeze(0)

        bbox = torch.tensor(sample.bbox)

        sentences: List[str] = [
            sample.sentences[np.random.choice(len(sample.sentences), 1)[0]]
        ]
        for _ in range(self._negative_sentences):
            neg_sample = self._samples[np.random.choice(len(self._samples), 1)[0]]
            sent = neg_sample.sentences[np.random.choice(len(sample.sentences), 1)[0]]
            sentences.append(sent)

        return RefCocoBatchSample(image, sentences, bbox)

    @staticmethod
    def batchify(
        batch: List[RefCocoBatchSample],
    ) -> RefCocoBatch:
        images = [sample.image for sample in batch]
        sentences = [sample.sentences for sample in batch]
        bboxes = [sample.bbox for sample in batch]

        return RefCocoBatch(
            images=torch.stack(images),
            sentences=sentences,
            bboxes=torch.stack(bboxes),
        )
