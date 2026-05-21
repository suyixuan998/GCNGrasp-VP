import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def bce(logits: dict,
        label: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits[mask], label[mask],
        weight=weight if weight is None else weight[mask]
    )


def dist_peak(logits: dict,
              batch: dict,
              k: int = 3,
              relative: bool = True) -> np.ndarray:
    mask = batch["label"].any(dim=-1)
    if not mask.any(): return np.array([])

    peak_aff = batch["peak_aff"][mask]
    bi = torch.arange(len(peak_aff), device=peak_aff.device)[:, None].repeat(1, k)
    ni = torch.topk(logits["logits"][mask], k=k, dim=1).indices

    xyz = logits["xyz"][mask]
    dist = torch.norm(xyz - peak_aff[:, None], dim=-1)
    ret = (dist[bi, ni] - dist.min(dim=-1, keepdim=True).values).mean(dim=-1)

    if relative:
        ret /= torch.norm(xyz - xyz.mean(dim=1, keepdims=True), dim=-1).max(dim=-1).values * 2
    return ret.cpu().numpy()


def zero_loss(loss: torch.Tensor):
    return loss - loss.detach()


class LossFunction(nn.Module):

    def __init__(self,
                 weight: dict,
                 gripper_info: dict = None):
        super().__init__()
        self.weight = weight
        self.gripper_info = gripper_info

        self.hyp = weight.get("hyp", {})
        if self.hyp: weight.pop("hyp")

    def forward(self,
                batch: dict,
                ret_stu: dict,
                ret_tea: dict):
        label = batch["label"]
        mask = batch["mask"]
        assert mask.dtype == torch.bool
        ret = {"loss": {}}

        # loss: TOG
        if self.weight.get("tog"):
            ret["bs_grasp"] = mask.sum().item()
            ret["loss"]["tog"] = bce(ret_stu["logits"], label.float(), mask)

        # loss: consist
        if self.weight.get("consist"):
            ret["loss"]["consist"] = 0.
            if ret_tea:
                ret["loss"]["consist"] = zero_loss(F.mse_loss(ret_stu["logits"][mask], ret_tea["logits"][mask]))

        # loss: affordance
        if self.weight.get("affordance"):
            affordance = ret_stu["affordance"]

            weight = label.any(dim=-1).float()
            xyz = affordance["xyz"]
            scale = torch.norm(xyz - xyz.mean(dim=1, keepdims=True), dim=-1).max(dim=-1).values * 2

            pred = (affordance["score"].unsqueeze(-1) * affordance["xyz"]).sum(dim=1)
            ret["loss"]["affordance"] = zero_loss(
                weight @ torch.square((pred - batch["peak_aff"]) / scale.unsqueeze(-1)).sum(dim=-1) / (weight.sum() + 1e-6)
            )

        ret["loss"] = sum(w * ret["loss"][n] for n, w in self.weight.items())
        return ret
