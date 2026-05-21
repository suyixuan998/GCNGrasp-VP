from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import gcngrasp
from gcngrasp.utils import utils

np.set_printoptions(precision=4, suppress=True)


def analysis_result():
    from gcngrasp.utils import analysis

    # file = "runs/lightning_logs/o2-test-all/test_pred_aff.txt"
    file = "runs/custom/weight/test_pred_part.txt"
    preds = analysis.TaskGraspPrediction(file=file, dataset=CFG_COL.database)
    preds.ap_matrix()


def plot_cost(metrics: list[str] = ["cost-infer (s)", "GPU-mem (GB)"]):
    device = torch.device("cuda")
    n_grasp = (25 * np.arange(1, 7)).tolist()

    res = {}
    for version in ("1", "gpt", "2", "2aff"):
        cfg_col = gcngrasp.ConfigCollector.from_version(version)
        method = gcngrasp.get_model_name(version)
        tmp = {"n_grasp": n_grasp}

        module = cfg_col.model_type(cfg_col.model_config).to(device)
        for b in tqdm(n_grasp, desc=method):
            args = module.example_input(1, b, cfg_col.data_config["n_points"])

            for k, v in utils.info_cost(module, *args).items():
                tmp.setdefault(k, []).append(v)
        res[method] = tmp
        del module
    print(res)

    colors = ["orange", "gold", "limegreen", "deepskyblue"]
    linestyles = [":", "--", "-", "--"]
    markders = ["o", "", "", "*"]
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i + 1)
        # if "mem" in metric: fig.set_yscale("log")

        plt.xlabel("n-grasp")
        plt.ylabel(metric)

        for j, (method, infos) in enumerate(res.items()):
            plt.plot(infos["n_grasp"], infos[metric], label=method, color=colors[j], marker=markders[j], linestyle=linestyles[j])
        plt.grid(True)
        plt.legend()
    plt.show()


def sample_batch(batch_size: int,
                 seed: int = 102,
                 verbose: bool = True):
    """ Sample a batch. """
    import pytorch_lightning as pl

    datamodule = CFG_COL.database.as_datamodule("o0", batch_size=batch_size)
    pl.seed_everything(seed, verbose=verbose)

    batch = next(iter(datamodule.train_dataloader()))
    if verbose: utils.info_batch(batch)
    return batch


def sample_Teb():
    from gcngrasp.api.utils.camera import Pinhole
    from gcngrasp.system import process_config
    from gcngrasp.system.data_type.action import Action
    from gcngrasp.system.data_type.workspace import IrregularWorkspace

    process_config("config/system/params.yaml", with_api=False)
    camera = Pinhole.from_yaml("config/system/RS-D435i.yaml")
    ws = IrregularWorkspace("config/system/franka_ws.npy", r_decay=.9)

    # camera position
    focus = np.array([.43, -.15, .05])
    pos = ws.sample_lookat(focus=focus,
                           dist_min=camera.dist_observe(radius=.14, border_margin=.1),
                           n=10,
                           elev_range=(.2, 1.2))

    # gripper
    Tebs = np.stack([Action.from_lookat(p, focus).Teb for p in pos])
    gcngrasp.Path("runs/Tebs.npy").numpy(Tebs)

    # visualize
    fig = plt.subplot(projection="3d")
    fig.scatter(*pos.T, s=1, c="r")
    ws.visualize(fig)
    fig.axis("equal")
    plt.show()


def test_dataset():
    import open3d as o3d
    from gcngrasp.api.utils import o3d_extension as o3de
    from gcngrasp.dataset import task_grasp

    database: task_grasp.Database = CFG_COL.database
    obj = database.load_obj(238)
    tid = database.tasks.index("flip")

    # ins_cvt = InstructionConverter(database, model="qwen3.6-plus-2026-04-02")
    # print(ins_cvt("翻转锅"))

    pcd = obj.point_cloud(as_o3d=True)
    grasps = [
        obj.mesh_grasp(tid),
        obj.mesh_grasp(),
        []
    ]
    i = o3de.farthest_point_sampling(obj.grasps[:, :3, 3], 8)
    # i = np.random.randint(len(obj.grasps), size=10)
    for j in range(2):
        grasps[j] = [grasps[j][k] for k in i]
    for g in grasps:
        o3d.visualization.draw_geometries([pcd] + g)


@torch.inference_mode()
def test_tog_net():
    """ Test TOG net. """
    device = torch.device("cuda")

    module = CFG_COL.model_type(CFG_COL.model_config).to(device)
    print(module)
    # module._trainer = True
    # module.load_org_ckpt(gcngrasp.get_org_ckpt(CFG_COL.version, "o0"))

    # print(module(*module.example_input(2, 3, 1024)))
    batch = sample_batch(batch_size=2)
    args = batch["pcd_full"].to(device), batch["grasp"].to(device), batch["obj"], batch["task"]
    print(args[1].shape)
    module(*args)


def vis_tog_label(oid: int):
    database = CFG_COL.database
    obj = database.load_obj(oid)

    otid = np.arange(len(database.tasks))[:, None].repeat(2, axis=1)
    otid[:, 0] = oid
    otid = otid[database.is_gold_ot(otid)]

    for t in otid[:, 1]:
        print(f"{t}: {database.tasks[t]}")
        obj.visualize(task_id=t)


if __name__ == '__main__':
    CFG_COL = gcngrasp.ConfigCollector.from_version("2aff")

    test_dataset()
