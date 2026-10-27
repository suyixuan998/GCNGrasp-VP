import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

import gcngrasp
from gcngrasp.dataset.task_grasp import TaskGrasp, representative_point
from main import MainModule


class LocalTaskGraspTests(unittest.TestCase):
    def test_representative_point_separates_positive_and_negative_centroids(self):
        centers = np.array([[0., 0., 0.], [1., 0., 0.], [5., 0., 0.]], dtype=np.float32)
        grasps = np.repeat(centers[:, None], 6, axis=1)
        np.testing.assert_array_equal(representative_point(grasps, [1, 1, 0]), centers[0])
        np.testing.assert_array_equal(representative_point(grasps, [1, 1, 1]), centers[1])
        np.testing.assert_array_equal(representative_point(grasps, [0, 0, 0]), np.zeros(3))

    def test_conflicting_rows_preserved_and_padding_masked(self):
        obj = SimpleNamespace(control_points=np.zeros((2, 6, 3), dtype=np.float32),
                              points=np.eye(3, dtype=np.float32),
                              partials=np.ones((1, 3, 3), dtype=np.float32))
        class ObjectWithLength(SimpleNamespace):
            def __len__(self):
                return len(self.control_points)
        obj = ObjectWithLength(**vars(obj))
        db = SimpleNamespace(tasks=["handover"], names={1: "001_example"},
                             bs_grasp=4, n_points=3, seed=2025,
                             load_obj=lambda oid: obj)
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "train_split.txt"
            split.write_text("001_example-0-handover:1\n001_example-0-handover:-1\n001_example-1-handover:-2\n")
            dataset = TaskGrasp(db, split)
            batch = dataset[0]
        self.assertEqual(len(dataset.annotation_keys), 2)
        self.assertEqual(int(batch["mask"].sum()), 3)
        self.assertEqual(int(batch["label"].sum()), 1)
        self.assertEqual(batch["grasp_id"][batch["mask"]].tolist(), [0, 0, 1])
        self.assertFalse(bool(batch["mask"][-1]))

    def test_affordance_target_rotates_with_grasp_in_train_and_validation(self):
        module = MainModule(gcngrasp.ConfigCollector.from_version("2aff"), "o0")
        class GeometryEcho(nn.Module):
            def forward(self, pcd, grasp, **kwargs):
                self.grasp = grasp
                return {"score": torch.zeros(grasp.shape[:2]), "logits": torch.zeros(grasp.shape[:2])}
        class CheckTarget(nn.Module):
            def forward(self, batch, student, teacher):
                torch.testing.assert_close(batch["peak_aff"], module.model.grasp[:, 0, 0])
                return {"loss": student["logits"].sum(), "bs_grasp": 2}
        module.model = GeometryEcho()
        module.loss_fcn = CheckTarget()
        peak = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        batch = {"obj": ["001_example"] * 2, "task": ["handover"] * 2,
                 "pcd_part": torch.rand(2, 32, 3), "pcd_full": torch.rand(2, 32, 3),
                 "grasp": peak[:, None, None].expand(2, 1, 6, 3).clone(),
                 "peak_aff": peak.clone(), "ot_id": torch.tensor([[1, 0], [1, 0]]),
                 "grasp_id": torch.zeros(2, 1, dtype=torch.long),
                 "label": torch.ones(2, 1), "mask": torch.ones(2, 1, dtype=torch.bool)}
        for training, stage in [(True, "train"), (False, "val")]:
            module.train(training)
            module(stage, batch)
            self.assertFalse(torch.allclose(module.model.grasp[:, 0, 0], peak))
        torch.testing.assert_close(batch["peak_aff"], peak)


if __name__ == "__main__":
    unittest.main()
