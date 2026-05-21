from dataclasses import dataclass, asdict
from functools import cached_property, partial
from typing import Callable, Type

from .api import Path


def get_model_config(version: str) -> str:
    return {
        "1": "config/model_v1.yaml",
        "2": "config/model_v2.yaml",
        "2aff": "config/model_v2aff.yaml",
        "gpt": "config/model_graspgpt.yaml"
    }[version]


def get_model_name(version: str) -> str:
    return {
        "1": "GCNGrasp-v1",
        "2": "GCNGrasp-v2",
        "2aff": "GCNGrasp-v2-aff",
        "gpt": "GraspGPT"
    }[version]


def get_model_type(version: str) -> Type["pl.LightningModule"]:
    if version == "1":
        from .model.gcngrasp_v1 import GCNGraspV1
        return GCNGraspV1

    elif version in ("2", "2ema", "2aff"):
        from .model.gcngrasp_v2 import GCNGraspV2
        return GCNGraspV2

    elif version == "gpt":
        from .model.graspgpt import GraspGPT
        return GraspGPT

    else:
        raise ValueError(f"Invalid model version: {version}")


def get_org_ckpt(version: str,
                 fold: str) -> str:
    root = {
        "1": "GCNGrasp-v1-org",
        "gpt": "GraspGPT-org"
    }
    if root.get(version):
        return f"checkpoints/{root[version]}/{fold}.ckpt"


def get_subset_type(version: str) -> str:
    return {
        "2aff": "Affordance"
    }.get(version, "TOG")


def get_train_config(version: str) -> str:
    return {
        "1": "config/train_v1.yaml",
        "2": "config/train_v2.yaml",
        "2aff": "config/train_v2aff.yaml",
    }.get(version)


@dataclass
class ConfigCollector:
    model_file: Path
    train_file: Path
    data_file: Path = None

    fold: str = None
    ckpt_best: Path = None
    ckpt_last: Path = None

    def __post_init__(self):
        for k, v in asdict(self).items():
            if k in ("fold",): continue
            if v is not None: setattr(self, k, Path(v))

    @cached_property
    def batch_size(self) -> int:
        return self.train_config["train"]["batch_size"]

    @cached_property
    def data_config(self) -> dict:
        return self.data_file.yaml()

    @cached_property
    def database(self):
        from .dataset.task_grasp import Database
        return Database(self.data_config)

    @cached_property
    def model_config(self) -> dict:
        return self.model_file.yaml()

    @cached_property
    def model_type(self):
        return get_model_type(self.version)

    @cached_property
    def subset_type(self) -> str:
        return get_subset_type(self.version)

    @cached_property
    def train_config(self) -> dict:
        return self.train_file.yaml()

    @cached_property
    def version(self) -> str:
        return str(self.model_config["version"])

    @classmethod
    def from_version(cls, version: str):
        return cls(
            model_file=get_model_config(version),
            train_file=get_train_config(version),
            data_file="config/data.yaml"
        )

    @classmethod
    def from_path(cls, path: Path):
        path = Path(path)
        ckpt_path = path / "checkpoints"
        return cls(
            model_file=next(path.glob("model*.yaml")),
            train_file=next(path.glob("train*.yaml")),
            data_file="config/data.yaml",
            fold=Path(path / "hparams.yaml").yaml()["fold"],
            ckpt_best=ckpt_path / "best.ckpt",
            ckpt_last=ckpt_path / "last.ckpt",
        )

    def func_main_loop(self, args, trainer_kwargs) -> Callable:
        import pytorch_lightning as pl
        import torch

        # dataset, trainer
        trainer_kwargs = dict(**trainer_kwargs, sync_batchnorm=True)
        batch_size = self.batch_size

        # main loop
        if args.train or args.resume:
            trainer = pl.Trainer(**trainer_kwargs, devices=args.devices or "auto")
            main_loop = trainer.fit
            ckpt_path = self.ckpt_last

            print("World size:", trainer.world_size)
            batch_size //= trainer.world_size
            assert batch_size * trainer.world_size == self.batch_size

        else:
            trainer = pl.Trainer(**trainer_kwargs, num_nodes=1, devices=[torch.cuda.device_count() - 1])
            main_loop = trainer.test
            ckpt_path = self.ckpt_best
            batch_size //= args.test_bs_div

        datamodule = self.database.as_datamodule(
            args.fold, batch_size=batch_size, subset_type=self.subset_type)
        return partial(main_loop, datamodule=datamodule, ckpt_path=ckpt_path)
