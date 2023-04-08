##
##
##

from dataclasses import dataclass
from pathlib import Path

from jaxtyping import Float
from torch import Tensor
from typing import List

@dataclass(init=True, frozen=True)
class RefCocoSample:
    path: Path
    bbox: List[float]
    sentences: List[str]


@dataclass(init=True, frozen=True)
class RefCocoBatchSample:
    image: Float[Tensor, "C H W"]
    sentences: List[str]
    bbox: Float[Tensor, "4"]


@dataclass(init=True, frozen=True)
class RefCocoBatch:
    images: Float[Tensor, "B C H W"]
    sentences: List[List[str]]
    bboxes: Float[Tensor, "B 4"]
