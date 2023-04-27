##
##
##

import json
import pickle
from typing import Literal, List

import numpy as np
import torch
import torchvision
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

        images = {}
        for image in instances["images"]:
            images[image["id"]] = images_path / image["file_name"]

        for ref in refs:
            if ref["split"] != phase:
                continue

            sentences = [sent["raw"] for sent in ref["sentences"]]
            if info.get(ref["ann_id"]) is not None:
                info[ref["ann_id"]]["sentences"].append(sentences)
            else:
                info[ref["ann_id"]] = {
                    "path": images[ref["image_id"]],
                    "sentences": [sentences],
                }

        for annotation in instances["annotations"]:
            if annotation["id"] in info:
                info[annotation["id"]]["bbox"] = annotation["bbox"]

        samples = []

        for sample_info in info.values():
            path = sample_info["path"]
            bbox = sample_info["bbox"]
            for sent in sample_info["sentences"]:
                sample = RefCocoSample(
                    path=path,  # type: ignore
                    bbox=bbox,
                    sentences=sent,
                )

                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> RefCocoBatchSample:
        sample = self._samples[index]
        img = torchvision.io.read_image(str(sample.path))

        # preventing 1 channel images => stick to RGB
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)

        bbox = torch.tensor(sample.bbox)

        sentences: List[str] = [
            sample.sentences[np.random.choice(len(sample.sentences), 1)[0]]
        ]
        for _ in range(self._negative_sentences):
            neg_sample = self._samples[np.random.choice(len(self._samples), 1)[0]]
            sent = neg_sample.sentences[np.random.choice(len(sample.sentences), 1)[0]]
            sentences.append(sent)

        return RefCocoBatchSample(img, sentences, bbox)

    @staticmethod
    def batchify(
        batch: List[RefCocoBatchSample],
    ) -> RefCocoBatch:
        images = [sample.image for sample in batch]
        sentences = [sample.sentences for sample in batch]
        bboxes = [sample.bbox for sample in batch]

        return RefCocoBatch(
            images=images,
            sentences=sentences,
            bboxes=torch.stack(bboxes),
        )
