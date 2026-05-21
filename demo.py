from __future__ import annotations

from pathlib import Path

import open3d as o3d

from gcngrasp.system import process_config
from gcngrasp.system import virtual_robot
from gcngrasp.system.api import FUNC_API
from gcngrasp.system.scene import Scene, Frame

ASSETS = list(Path("assets").iterdir())


def find_batch(obj_id: int,
               task_id: int):
    """ Find a batch that satisfies the condition. """
    from gcngrasp.dataset import task_grasp

    database: task_grasp.Database = TOG_MODEL_WRAPPER.cfg_col.database

    if isinstance(task_id, str):
        task_id = database.tasks.index(task_id)
    # database.load_obj(obj_id).visualize(grasp=False, task_id=task_id)
    batch = database.find_batch(obj_id, task_id)

    mask = batch["mask"]
    print(f"obj_id: {obj_id}, task_id: {task_id}, mask: {mask.sum()} / {len(mask)}")
    return batch


def inference_dataset(obj_id: int,
                      task_id: int):
    import gcngrasp.api.utils.o3d_extension as o3de
    from gcngrasp.dataset import task_grasp
    from gcngrasp.utils.analysis import draw_prediction

    # target obj and task
    database: task_grasp.Database = TOG_MODEL_WRAPPER.cfg_col.database
    batch = find_batch(obj_id, task_id)

    # load batch
    obj = database.load_obj(batch["ot_id"][0])
    mask = batch["mask"]
    grasp_ids = batch["grasp_id"][mask]
    grasp = obj.grasps[grasp_ids]

    # inference
    tog, field = TOG_MODEL_WRAPPER(batch["pcd_full"], batch["grasp"][mask], batch["obj"], batch["task"])

    # downsample
    i = o3de.farthest_point_sampling(grasp[:, :3, 3], num_samples=10)

    # visualize
    geos = draw_prediction(obj, grasp_ids[i], tog[i]) + [
        field.to_o3d() if field else o3de.from_pcd_array(obj.point_cloud(), color_scale=1.)
    ]
    o3d.visualization.draw_geometries(geos)


def inference_img(config: dict,
                  frame: Frame,
                  args: list):
    scene = Scene(config, target_obj=args[0], tog_model_args=args[1:3])
    scene.integrate(frame)

    # print(scene.tog_poses[0])
    o3d.visualization.draw_geometries(scene.to_o3d(tog=True))


if __name__ == '__main__':
    CONFIG = process_config("config/system/params.yaml")
    TOG_MODEL_WRAPPER = FUNC_API.functions["tog"]

    inference_img(CONFIG, *virtual_robot.frame_from_folder(ASSETS[1]))

    # inference_dataset(20, "pour")
