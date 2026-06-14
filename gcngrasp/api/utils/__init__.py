from __future__ import annotations

import os
import time

import torch

from .path_extension import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def clean_cached_properties(obj: object,
                            attributes: list[str]) -> object:
    """ Clear the cached properties of an object. """
    for attr in attributes:
        if attr in obj.__dict__: delattr(obj, attr)
    return obj


def download_checkpoint(url: str,
                        filename: str = None) -> Path:
    filename = filename or url.split("/")[-1]
    dst = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, dst)
    return dst.resolve()


def get_tensor_args():
    import torch

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return device, dtype


def huggingface_model_path(repo_id: str,
                           check: bool = False) -> Path:
    """ Download the model from Hugging Face. """
    path = Path(f"~/.cache/huggingface/hub/models--{repo_id.replace('/', '--')}/snapshots").expanduser()
    if check or not path.exists():
        os.system(f"hf download {repo_id}")
    return next(path.iterdir())


class Timer(str):
    momentum: float = None
    precision: int = 3
    result: dict[str, float] = {}

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type: return False
        k = str(self)
        cost = time.time() - self.t0

        if self.momentum is None:
            self.result[k] = cost
            self.to_dict(log=True)
        else:
            self.result[k] = self.result.get(k, cost) * self.momentum + cost * (1 - self.momentum)

    @classmethod
    def reset(cls):
        cls.result = {}

    @classmethod
    def set_momentum(cls, m: float = None):
        cls.momentum = m

    @classmethod
    def to_dict(cls, log: bool = False):
        res = {k: cls.result[k] for k in sorted(cls.result)}
        for k, v in res.items():
            res[k] = round(v, cls.precision)
        if log: print(f"{__class__.__name__}: {res}")
        return res
