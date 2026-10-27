from __future__ import annotations

import argparse
import random

import numpy as np
import torch

import gcngrasp
import gcngrasp.utils.lit_extension as lite
from gcngrasp.utils import analysis
from gcngrasp.utils import utils
from gcngrasp.utils.functional import ItemStack
from gcngrasp.utils.loss_fcn import LossFunction, dist_peak

# warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision("medium")


class MainModule(lite.VanillaModule):

    def __init__(self,
                 cfg_col: gcngrasp.ConfigCollector,
                 fold: str):
        self.cfg_col = cfg_col
        super().__init__(self.cfg_col.train_file)
        self.save_hyperparameters("fold")

        self.model = self.cfg_col.model_type(self.cfg_col.model_config)
        self.loss_fcn = LossFunction(self.config["train"]["loss_weight"],
                                     gripper_info=self.cfg_col.model_config.get("input", {}).get("gripper"))
        self.item_stack = {stage: {mode: ItemStack() for mode in ("part", "full")} for stage in ("train", "val", "test")}

    def forward(self, stage, batch):
        infer_kwargs = dict(objs=batch["obj"], tasks=batch["task"])
        # if stage == "test": infer_kwargs["test_time_augment"] = 4

        pcd_part, pcd_full, grasp = batch["pcd_part"], batch["pcd_full"], batch["grasp"]
        ret_tea = None
        batch_stu = batch

        def rotate_inputs(pcd):
            # Affordance targets and geometry must stay in the same frame.
            if "peak_aff" in batch:
                pcd, rotated_grasp, peak = utils.random_rotate(pcd, grasp, batch["peak_aff"])
                return pcd, rotated_grasp, dict(batch, peak_aff=peak)
            pcd, rotated_grasp = utils.random_rotate(pcd, grasp)
            return pcd, rotated_grasp, batch

        if self.training:
            pcd_infer = (pcd_full if random.random() < self.config["p_pcd_full"] else pcd_part).clone()
            pcd_infer, grasp_infer, batch_stu = rotate_inputs(pcd_infer)
            ret_stu = self.model(pcd_infer, grasp_infer, **infer_kwargs)

        else:
            with torch.inference_mode():
                pcd_infer, grasp_infer, batch_stu = rotate_inputs(pcd_part)
                ret_stu = self.model(pcd_infer, grasp_infer, **infer_kwargs)
                if stage == "test":
                    pcd_infer, grasp_infer, batch_tea = rotate_inputs(pcd_full)
                    ret_tea = self.model(pcd_infer, grasp_infer, **infer_kwargs)

                    self.item_stack[stage]["full"].push(analysis.dump_prediction(batch, ret_tea["score"]))
                    if ret_tea.get("affordance"):
                        dp = dist_peak(ret_tea.get("affordance"), batch_tea)
                        if len(dp):
                            self.log(f"{stage}/affF/dist_peak", dp.mean(), batch_size=len(dp), prog_bar=True, on_step=False, on_epoch=True)

        # dump prediction
        self.item_stack[stage]["part"].push(analysis.dump_prediction(batch, ret_stu["score"].detach()))
        if ret_stu.get("affordance"):
            dp = dist_peak(ret_stu.get("affordance"), batch_stu)
            if len(dp):
                self.log(f"{stage}/affP/dist_peak", dp.mean(), batch_size=len(dp), prog_bar=True, on_step=False, on_epoch=True)

        if stage != "test": return self.loss_fcn(batch_stu, ret_stu, ret_tea)

    def on_fit_start(self):
        super().on_fit_start()
        self.copy_into_project(self.cfg_col.model_file)
        self.copy_into_project(self.cfg_col.data_file)

    def _on_shared_epoch_end(self, stage: str):
        etype = self.hparams["fold"][0]
        if etype not in "otc": etype = "o"
        metrics_all = {}

        # evaluation
        for m, e in self.item_stack[stage].items():
            if not e: continue
            items = e.pop(self.all_gather)
            items = items[items[:, -1].long().bool(), :-1].cpu().numpy()

            metrics = {}
            if stage != "test":
                metrics = self.cfg_col.database.evaluate(items, etype=etype)

            else:
                np.savetxt(self.project / f"test_pred_{m}.txt", items)
                for etype in "otc":
                    ms = self.cfg_col.database.evaluate(items, etype=etype)
                    for k in list(ms):
                        if k.startswith("mAP"): ms[f"{k}_{etype}"] = ms.pop(k)
                    metrics.update(ms)

            prefix = f"{stage}/{m}/"
            metrics_all.update({prefix + k: v for k, v in metrics.items()})

        # fuse metrics for optimal model selection
        if stage == "val":
            monitor_tog = self.config["train"]["metric"]["monitor"]
            monitor_aff = self.trainer.callback_metrics.get("val/affP/dist_peak")
            if monitor_aff is not None:
                metrics_all[monitor_tog] -= monitor_aff * .01

        for k, v in metrics_all.items():
            self.log(k, v, prog_bar=True, on_step=False, on_epoch=True)

    def _shared_step(self, stage, batch, batch_idx):
        ret = self(stage, batch)
        if ret:
            self.log("/".join([stage, "loss"]), ret["loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=ret["bs_grasp"])
            return ret["loss"]

    @classmethod
    def from_command(cls, args) -> "MainModule":
        org_ckpt = gcngrasp.get_org_ckpt(args.test, args.fold)
        if org_ckpt:
            # load origin GCNGrasp-v1
            cfg_col = gcngrasp.ConfigCollector.from_version("1")
            module = cls(cfg_col, args.fold)
            module.model.load_org_ckpt(org_ckpt)
        else:
            # initialize model
            p = args.resume or args.test
            cfg_col = gcngrasp.ConfigCollector.from_path(p) if p else gcngrasp.ConfigCollector.from_version(args.train)
            if args.train_config:
                cfg_col.train_file = gcngrasp.Path(args.train_config)
            if p and args.fold is None: args.fold = cfg_col.fold
            module = cls(cfg_col, args.fold)
        return module


def parse_args():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", type=str, help="model version trained from scratch")
    mode.add_argument("--resume", type=str, help="resume model path")
    mode.add_argument("--test", type=str, help="test model path")

    parser.add_argument("--fold", type=str, help="data fold, e.g. o0, t1")
    parser.add_argument("--devices", type=int, nargs="+", help="GPU devices")
    parser.add_argument("--test-bs-div", type=int, default=1, help="test batch size divider")
    parser.add_argument("--train-config", type=str, help="training YAML override for a new training run")
    parser.add_argument("--skip-cuda-wait", action="store_true", help="start on a shared GPU without waiting for other Python processes")
    args = parser.parse_args()
    if args.train and args.fold is None:
        parser.error("--fold is required with --train")
    if args.train_config and not args.train:
        parser.error("--train-config can only be used with --train")
    return args


if __name__ == '__main__':
    args = parse_args()
    if not args.skip_cuda_wait:
        utils.wait_for_cuda(args.devices, sleep=60, re_pattern="python")

    # main
    module = MainModule.from_command(args)
    module.cfg_col.database.pcds
    func = module.cfg_col.func_main_loop(args, module.trainer_kwargs())
    func(module)
