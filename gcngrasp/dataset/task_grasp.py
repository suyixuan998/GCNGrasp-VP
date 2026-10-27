"""Local TaskGrasp adapter, supplied because the upstream loader is absent.

Uses the official splits verbatim (o=held-out class, i=held-out instance,
t=held-out task). Partial views are synthetic hidden-point-removal views of
the fused scan, not the author's unpublished preprocessing. Affordance targets
follow Eq. (10) of https://arxiv.org/html/2606.19091v1.
"""
from __future__ import annotations

import hashlib
import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset

from ..api.utils import skmetrics
from ..utils.lit_extension import LazyDataModule


# Unique points of TaskGrasp's seven-point polyline, ordered for model_v2aff.
# Annotation poses already use the coordinate frame centered at the fused scan.
GRIPPER_POINTS = np.array([
    [-.10, 0., 0.], [-.03, 0., 0.],
    [-.03, .07, 0.], [-.03, -.07, 0.],
    [.03, .07, 0.], [.03, -.07, 0.],
], dtype=np.float32)


def parse_line(line):
    description, label = line.strip().rsplit(":", 1)
    obj, grasp, task = description.split("-", 2)
    # Matches the original TaskGrasp parse_line: only label 1 is positive.
    return obj, int(grasp), task, int(label) == 1


def representative_point(grasps, labels):
    """Eq. (10); omit the negative-centroid term when all grasps are positive."""
    centers = grasps.mean(axis=1)
    labels = np.asarray(labels, dtype=bool)
    if not labels.any():
        return np.zeros(3, dtype=np.float32)  # masked by LossFunction
    distance = np.linalg.norm(centers - centers[labels].mean(axis=0), axis=1)
    if (~labels).any():
        distance -= np.linalg.norm(centers - centers[~labels].mean(axis=0), axis=1)
    return centers[distance.argmin()].astype(np.float32)


def sample_points(points, n, rng):
    return points[rng.choice(len(points), n, replace=len(points) < n)].copy()


@dataclass
class TaskGraspObject:
    name: str
    points: np.ndarray
    partials: np.ndarray
    grasps: np.ndarray
    control_points: np.ndarray

    def __len__(self):
        return len(self.grasps)

    def point_cloud(self, as_o3d=False):
        if not as_o3d:
            return self.points
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points))


class Database:
    def __init__(self, config):
        self.config = dict(config)
        self.root = Path(config["root"]).expanduser().resolve()
        for required in ["scans", "splits_final", "task1_results.txt", "misc.pkl"]:
            if not (self.root / required).exists():
                raise FileNotFoundError(f"Missing TaskGrasp asset: {self.root / required}")
        with (self.root / "misc.pkl").open("rb") as f:
            self.map_obj2cls = pickle.load(f)[-1]
        rows = [line.strip().split("-", 2) for line in
                (self.root / "task1_results.txt").read_text().splitlines() if line.strip()]
        self.tasks = sorted({task for _, task, _ in rows})
        self.names = {int(name.split("_", 1)[0]): name for name in self.map_obj2cls}
        self.classes = sorted(set(self.map_obj2cls.values()))
        self.class_ids = {oid: self.classes.index(self.map_obj2cls[name])
                          for oid, name in self.names.items()}
        self.gold_pairs = {(int(name.split("_", 1)[0]), self.tasks.index(task))
                           for name, task, label in rows if label in ("True", "Weak True")}
        self.n_points = int(config.get("n_points", 4096))
        self.bs_grasp = int(config.get("bs_grasp", 25))
        self.seed = int(config.get("seed", 2025))
        self.n_views = int(config.get("partial_views", 8))
        if min(self.n_points, self.bs_grasp, self.n_views) < 1:
            raise ValueError("n_points, bs_grasp and partial_views must be positive")
        self.cache_dir = Path(config.get("cache_dir", "runs/taskgrasp_cache"))

    def _load_geometry(self, oid, name):
        path = self.root / "scans" / name
        pc_file = path / "fused_pc_clean.npy"
        key = (str(pc_file), pc_file.stat().st_mtime_ns, self.n_points,
               self.n_views, self.seed, "hpr-v1")
        cache = self.cache_dir / f"{name}-{hashlib.sha256(repr(key).encode()).hexdigest()[:12]}.npz"
        if cache.is_file():
            with np.load(cache) as data:
                points, partials = data["points"], data["partials"]
        else:
            # Use the full original mean, as in TaskGrasp's GCNLoader.
            points = np.load(pc_file)[:, :3]
            if not np.isfinite(points).all():
                raise ValueError(f"Nonfinite coordinates in {pc_file}")
            points = (points - points.mean(axis=0)).astype(np.float32)
            rng = np.random.default_rng(self.seed + oid)
            points = sample_points(points, min(len(points), self.n_points * 2), rng)
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            radius = np.linalg.norm(points, axis=1).max()
            if radius <= 0:
                raise ValueError(f"Degenerate point cloud: {pc_file}")
            views = []
            for i in range(self.n_views):
                z = .15 + .7 * (i + .5) / self.n_views
                angle = i * np.pi * (3 - np.sqrt(5))
                direction = np.array([np.sqrt(1 - z * z) * np.cos(angle),
                                      np.sqrt(1 - z * z) * np.sin(angle), z])
                _, visible = cloud.hidden_point_removal(direction * radius * 3, radius * 100)
                if not visible:
                    raise ValueError(f"Empty synthetic view: {name}/{i}")
                views.append(sample_points(points[np.asarray(visible)], self.n_points, rng))
            partials = np.stack(views)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache, points=points, partials=partials)

        grasp_files = sorted((path / "grasps").glob("*/grasp.npy"), key=lambda p: int(p.parent.name))
        grasp_ids = [int(p.parent.name) for p in grasp_files]
        if grasp_ids != list(range(len(grasp_ids))) or not grasp_files:
            raise ValueError(f"Missing/noncontiguous annotated grasps in {path}")
        grasps = np.stack([np.load(p) for p in grasp_files]).astype(np.float32)
        control = GRIPPER_POINTS @ grasps[:, :3, :3].transpose(0, 2, 1) + grasps[:, None, :3, 3]
        return TaskGraspObject(name, points, partials, grasps, control)

    @cached_property
    def pcds(self):
        print(f"Loading {len(self.names)} TaskGrasp objects; synthetic partial views cached in {self.cache_dir}", flush=True)
        objects = {}
        for i, (oid, name) in enumerate(sorted(self.names.items()), 1):
            objects[oid] = self._load_geometry(oid, name)
            if i % 20 == 0 or i == len(self.names):
                print(f"Loaded geometry: {i}/{len(self.names)}", flush=True)
        return objects

    def load_obj(self, oid):
        return self.pcds[int(oid)]

    def is_gold_ot(self, pairs):
        return np.array([tuple(map(int, pair)) in self.gold_pairs for pair in pairs], dtype=bool)

    def as_datamodule(self, fold, batch_size=1, subset_type="TOG"):
        if subset_type not in ("TOG", "Affordance"):
            raise ValueError(f"Unsupported subset type: {subset_type}")
        if not fold or len(fold) != 2 or fold[0] not in "oitg" or fold[1] not in "0123":
            raise ValueError(f"Invalid fold {fold!r}; use o0..o3, i0..i3, t0..t3 or g0..g3")
        split_dir = self.root / "splits_final" / fold[0] / fold[1]
        datasets = [TaskGrasp(self, split_dir / f"{stage}_split.txt", train=stage == "train")
                    for stage in ("train", "val", "test")]
        # Audit official partitions before starting an expensive fit.
        keys = [dataset.annotation_keys for dataset in datasets]
        if any(keys[i] & keys[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f"Overlapping grasp annotations across partitions in {split_dir}")
        if fold[0] in "oit":
            def heldout(dataset):
                return {self.map_obj2cls[name] if fold[0] == "o" else
                        name if fold[0] == "i" else self.tasks[tid]
                        for name, _, tid, _ in dataset.groups}
            units = [heldout(dataset) for dataset in datasets]
            if any(units[i] & units[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                raise ValueError(f"Held-out {fold[0]} units leak across partitions in {split_dir}")
        return LazyDataModule(*(lambda dataset=dataset: dataset for dataset in datasets),
                              batch_size=batch_size, num_workers=int(self.config.get("num_workers", 0)))

    def evaluate(self, items, etype="o"):
        # items: object id, task id, grasp id, binary label, score
        pairs = items[:, :2].astype(int)
        labels, scores = items[:, 3].astype(int), items[:, 4]
        predictions = scores >= .5
        metrics = {name: float(func(labels, predictions))
                   for name, func in skmetrics.METRICS_HARD.items()}
        metrics["AP"] = float(skmetrics.skm.average_precision_score(labels, scores))
        if len(np.unique(labels)) > 1:
            metrics["AUC"] = float(skmetrics.skm.roc_auc_score(labels, scores))
        gold = self.is_gold_ot(pairs)
        if gold.any() and labels[gold].any():
            group = (pairs[:, 1] if etype == "t" else
                     np.array([self.class_ids[oid] for oid in pairs[:, 0]]) if etype == "c" else pairs[:, 0])
            metrics["mAP"] = float(skmetrics.mAP_score(group[gold], labels[gold], scores[gold]))
            metrics["mAP_ot"] = float(skmetrics.mAP_score(pairs[gold], labels[gold], scores[gold]))
        return metrics


class TaskGrasp(Dataset):
    def __init__(self, database, split_file, train=False):
        self.database = database
        self.train = train
        annotations = defaultdict(list)
        self.annotation_keys = set()
        duplicate_count = 0
        for line in Path(split_file).read_text().splitlines():
            if not line.strip():
                continue
            name, gid, task, label = parse_line(line)
            oid, tid = int(name.split("_", 1)[0]), database.tasks.index(task)
            if database.names.get(oid) != name:
                raise ValueError(f"Unknown object {name} in {split_file}")
            if not 0 <= gid < len(database.load_obj(oid)):
                raise ValueError(f"Unknown grasp {name}/{gid} in {split_file}")
            if (oid, tid, gid) in self.annotation_keys:
                duplicate_count += 1
            self.annotation_keys.add((oid, tid, gid))
            # Official splits contain repeated handover annotations, including
            # conflicting labels. Preserve every row, just as GCNLoader does.
            annotations[(name, oid, tid)].append((gid, label))
        self.groups = []
        for (name, oid, tid), values in sorted(annotations.items()):
            rows = np.array(sorted(values), dtype=np.int64)
            peak = representative_point(database.load_obj(oid).control_points[rows[:, 0]], rows[:, 1])
            for start in range(0, len(rows), database.bs_grasp):
                self.groups.append((name, oid, tid, (rows[start:start + database.bs_grasp], peak)))
        if not self.groups:
            raise ValueError(f"Empty partition: {split_file}")
        print(f"{Path(split_file).parent}: {Path(split_file).stem}: {len(self.groups)} groups, "
              f"{sum(len(values) for values in annotations.values())} rows, "
              f"{duplicate_count} repeated annotations (preserved)", flush=True)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        name, oid, tid, (rows, peak) = self.groups[index]
        obj = self.database.load_obj(oid)
        # Validation/test use fixed sampling; training follows Lightning worker seeds.
        rng = np.random if self.train else np.random.default_rng(self.database.seed + index)
        ids = rows[:, 0]
        count, total = len(ids), self.database.bs_grasp
        ids = np.pad(ids, (0, total - count), mode="edge")
        labels = np.pad(rows[:, 1], (0, total - count)).astype(np.float32)
        view = int(rng.randint(len(obj.partials)) if self.train else rng.integers(len(obj.partials)))
        return {
            "obj": name, "task": self.database.tasks[tid],
            "ot_id": torch.tensor([oid, tid], dtype=torch.long),
            "pcd_full": torch.from_numpy(sample_points(obj.points, self.database.n_points, rng)),
            "pcd_part": torch.from_numpy(obj.partials[view].copy()),
            "grasp": torch.from_numpy(obj.control_points[ids].copy()),
            "grasp_id": torch.from_numpy(ids.copy()),
            "label": torch.from_numpy(labels),
            "mask": torch.arange(total) < count,
            "peak_aff": torch.from_numpy(peak.copy()),
        }
