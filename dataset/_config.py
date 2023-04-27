##
##
##

import enum
from pathlib import Path
from typing import Any, Dict
from typing_extensions import Self


class RefCocoSplit(enum.Enum):
    GOOGLE = "google"
    UMD = "umd"

    @classmethod
    def from_string(cls, split: str):
        return cls[split.upper().strip()]

    def __str__(self) -> str:
        return self.value


class RefCocoConfig:
    path: Path
    split: RefCocoSplit
    negative_sentences: int

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.path = Path(cfg["path"])
        self.split = RefCocoSplit.from_string(cfg.get("split", "umd"))
        self.negative_sentences = cfg.get("negative_examples", 0)

    @classmethod
    def default(cls, path: str) -> Self:
        return cls({"path": path})
