from __future__ import annotations

import copy
import pickle
import time
import warnings
from dataclasses import dataclass, field

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml
from pathlib import Path

T_CAM_IMU = np.array([[0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1]], dtype=np.float32)


def binary(self: Path, data=None, **kwargs):
    return pickle.loads(self.read_bytes(), **kwargs) \
        if data is None else self.write_bytes(pickle.dumps(data, **kwargs))


@dataclass
class Frame:
    camera: "Camera"
    color: np.ndarray = field(repr=False)
    depth: np.ndarray = field(repr=False)
    Tcw: np.ndarray = field(default_factory=T_CAM_IMU.copy)
    t: float = field(default_factory=time.time)

    color_encoding: str = "BGR"
    sensor_depth: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.color, np.ndarray):
            self.color = cv2.imread(str(self.color))
        assert self.color is not None

        if self.sensor_depth:
            self.depth = self.camera.process_depth(self.depth)

    def concat(self,
               cmap: str = "Spectral_r",
               axis: int = 1) -> np.ndarray:
        cmap = plt.get_cmap(cmap)
        images = [self.cvt_color("RGB"), self.depth]

        for i, img in enumerate(images):
            if img.ndim == 2:
                img = (img - img.min()) / (img.max() - img.min())
                images[i] = np.uint8(cmap(img)[..., :3] * 255)
        return np.concatenate(images, axis=axis)

    def cvt_color(self, encoding: str) -> np.ndarray:
        return cv2.cvtColor(
            self.color, getattr(cv2, f"COLOR_{self.color_encoding}2{encoding}")
        ) if self.color_encoding != encoding else self.color

    @property
    def depth_mask(self) -> np.ndarray:
        return self.depth > 0

    @property
    def pcd_array(self) -> np.ndarray:
        return np.concatenate([self.pointmap, self.cvt_color("RGB") / 255], axis=-1)

    @property
    def pointmap(self) -> np.ndarray:
        return self.camera.pointmap(self.depth)

    def resize(self, img_size: list[int]):
        ret = copy.copy(self)
        ret.camera.resize(img_size, mode="scale")
        ret.color = cv2.resize(ret.color, img_size)
        ret.depth = cv2.resize(ret.depth, img_size)
        return ret

    @property
    def Twc(self) -> np.ndarray:
        return np.linalg.inv(self.Tcw)

    def to_pcd(self,
               mask: np.ndarray = None,
               to_world: bool = True) -> o3d.geometry.PointCloud:
        mask_ = self.depth_mask
        if mask is not None: mask_ &= mask
        pcd = self.pcd_array[mask_]

        ret = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd[:, :3]))
        ret.colors = o3d.utility.Vector3dVector(pcd[..., 3:])

        if to_world: ret = ret.transform(self.Tcw)
        return ret


@dataclass
class FrameWithConfidence(Frame):
    conf: np.ndarray = field(default=None, repr=False)
    conf_thresh: float = None

    def __post_init__(self):
        assert self.conf is not None

    @property
    def depth_mask(self) -> np.ndarray:
        return self.conf > self.conf_thresh

    def resize(self, img_size: list[int]):
        ret = super().resize(img_size)
        ret.conf = cv2.resize(ret.conf, img_size)
        return ret


@dataclass
class Camera:
    img_size: list[int]
    intrinsics: list[float]
    dist_coeffs: list[float] = None
    # Optional parameters for depth cameras
    depth_scale: float = None
    depth_range: list[float] = None

    def __post_init__(self):
        assert len(self.img_size) == 2, "Image size must be a tuple of (width, height)"
        assert len(self.intrinsics) == 4
        self._pixels = np.stack(np.meshgrid(*map(np.arange, self.img_size)), axis=-1)
        self._unproj: np.ndarray = None

        if self.depth_range is not None:
            assert len(self.depth_range) == 2, "Depth range must be a tuple of (min_depth, max_depth)"

    @classmethod
    def from_dict(cls, cfg):
        cam_type = cfg.pop("type")
        if cls is __class__:
            cam_type = eval(cam_type)
        else:
            if cam_type != cls.__name__:
                warnings.warn(f"Camera type mismatch: expected '{cls.__name__}', got '{cam_type}'")
            cam_type = cls
        return cam_type(**cfg)

    @classmethod
    def from_yaml(cls, file):
        with open(file, 'r') as f:
            cfg = yaml.load(f.read(), Loader=yaml.Loader)
        return cls.from_dict(cfg)

    @property
    def camera_matrix(self) -> np.ndarray:
        """Camera matrix"""
        fx, fy, cx, cy = self.intrinsics
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    def contains_depth(self, depth: np.ndarray) -> np.ndarray:
        """ Check if depth map is within the range """
        return np.ones_like(depth, dtype=bool) if self.depth_range is None \
            else (depth >= self.depth_range[0]) & (depth <= self.depth_range[1])

    def contains_pcd(self, pcd: np.ndarray) -> np.ndarray:
        """ Check if point cloud is within the image size """
        pixels = np.round(self.project(pcd)).astype(int)
        return self.contains_pixels(pixels) & self.contains_depth(pcd[..., 2])

    def contains_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """ Check if pixels are within the image size """
        return (pixels >= 0).all(axis=-1) & (pixels < self.img_size).all(axis=-1)

    def normalized_plane(self, img: np.ndarray) -> np.ndarray:
        """ Unproject image to normalized plane """
        w, h = map((-1).__add__, self.img_size)
        xyxy = self.unproject(np.array([[0, h / 2], [w / 2, 0], [w, h / 2], [w / 2, h]])
                              )[:, :2].flatten()[[0, 3, 4, 7]]
        pcd = self._pixels / self.img_size * (xyxy[2:] - xyxy[:2]) + xyxy[:2]
        pcd = np.concatenate([pcd, np.ones_like(pcd[..., :1])], axis=-1)
        pixels = self.project(pcd).astype(np.float32)
        return cv2.remap(img, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR)

    def pointmap(self, depth: np.ndarray = None) -> np.ndarray:
        """ Generate point cloud from depth map """
        return self._unproj.copy() if depth is None else self._unproj * depth[..., None]

    def project(self, pcd: np.ndarray) -> np.ndarray:
        """ Project 3D points to 2D image plane """
        raise NotImplementedError

    def process_depth(self,
                      depth_or_file: str | np.ndarray) -> np.ndarray:
        """ Process depth map from file or numpy array """
        if not isinstance(depth_or_file, np.ndarray):
            depth_or_file = cv2.imread(depth_or_file, cv2.IMREAD_UNCHANGED)
        depth_or_file = cv2.resize(depth_or_file, self.img_size)

        if "int" in str(depth_or_file.dtype):
            depth_or_file = depth_or_file.astype(np.float32) / self.depth_scale

        if self.depth_range is not None:
            depth_or_file = np.where(
                (depth_or_file >= self.depth_range[0]) & (depth_or_file <= self.depth_range[1]),
                depth_or_file, 0
            )
            valid_radio = (depth_or_file > 0).astype(np.float32).mean().item()
            if valid_radio < .05:
                warnings.warn(f"Depth map has too few valid points ({valid_radio:.2%}).")
        return depth_or_file

    def render_depth(self,
                     pcd: np.ndarray,
                     return_vis_ratio: bool = False) -> np.ndarray:
        """
        Rendering the depth map corresponding to 3D points
        :returns: pixels, depth, [vis_ratio]
        """
        assert pcd.shape[-1] == 3 and pcd.ndim == 2, "Point cloud must be a Nx3 array."
        pixels = np.round(self.project(pcd)).astype(int)
        is_vis = np.where(self.contains_pixels(pixels))[0]

        p_u, p_id = np.unique(pixels[is_vis], return_inverse=True, axis=0)
        d = np.full(len(p_u), np.inf, dtype=pcd.dtype)
        np.minimum.at(d, p_id, pcd[..., 2][is_vis])

        ret = (p_u, d)
        if return_vis_ratio: ret += (len(is_vis) / len(pixels),)
        return ret

    def undistort(self, img: np.ndarray) -> np.ndarray:
        """ Undistort image """
        return img.copy()

    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        """ Unproject 2D image plane to 3D points """
        pixels = pixels.astype(np.float32)
        return cv2.remap(
            self._unproj[..., :2], pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR
        ).reshape(*pixels.shape)


@dataclass
class Pinhole(Camera):

    def __post_init__(self):
        super().__post_init__()
        if self.dist_coeffs:
            K = self.camera_matrix
            self._undist_map = cv2.initUndistortRectifyMap(K, np.array(self.dist_coeffs), None, K, self.img_size, cv2.CV_32FC1)
        # cache for unprojection
        self._unproj = np.concatenate([(self._pixels - self.intrinsics[2:]) / self.intrinsics[:2],
                                       np.ones_like(self._pixels[..., :1])], axis=-1).astype(np.float32)

    def dist_observe(self,
                     radius: float,
                     border_margin: float) -> float:
        assert 0 <= border_margin < 1, "Border margin must be in [0, 1)"
        tan_theta = np.tan(self.fov.min() / 2).item()
        return max(radius + (self.depth_range[0] if self.depth_range else 0),
                   radius / (1 - border_margin * 2) / tan_theta)

    @property
    def fov(self) -> np.ndarray:
        return 2 * np.arctan(np.array(self.img_size, dtype=np.float32) / 2 / self.intrinsics[:2])

    def project(self, pcd: np.ndarray) -> np.ndarray:
        pcd = pcd.astype(np.float32)
        return pcd[..., :2] / pcd[..., 2:] * self.intrinsics[:2] + self.intrinsics[2:]

    def resize(self,
               img_size: list[int],
               mode: str):
        if img_size == self.img_size: return
        assert not self.dist_coeffs, "Pinhole camera does not support resizing with distortion coefficients."
        intrinsics = np.array(self.intrinsics)
        # resize
        if mode == "scale":
            r = np.array(img_size) / self.img_size
            intrinsics *= [*r, *r]
        # crop
        elif mode == "crop":
            intrinsics[2:] -= (np.array(img_size) - self.img_size) / 2
        else:
            raise ValueError(f"Unknown param: {mode}")
        self.img_size = list(img_size)
        self.intrinsics = intrinsics.tolist()
        self.__post_init__()

    def undistort(self, img: np.ndarray):
        if not self.dist_coeffs: return img
        return cv2.remap(img, *self._undist_map, interpolation=cv2.INTER_LINEAR)


@dataclass
class KannalaBrandt(Camera):
    fov_max: float = 180

    def __post_init__(self):
        super().__post_init__()
        assert len(self.dist_coeffs) == 4
        # cache for unprojection
        coords = (self._pixels - self.intrinsics[2:]) / self.intrinsics[:2]
        coords /= self._solveWZ(coords)[..., None]
        self._unproj = np.concatenate([coords, np.ones_like(coords[..., :1])], axis=-1).astype(np.float32)

    def _computeR(self, theta: np.ndarray) -> np.ndarray:
        theta2 = np.square(theta)
        return theta + theta2 * (
                self.dist_coeffs[0] + theta2 * (
                self.dist_coeffs[1] + theta2 * (
                self.dist_coeffs[2] + theta2 * self.dist_coeffs[3])))

    def _solveWZ(self,
                 coords: np.ndarray,
                 niter: int = 20) -> np.ndarray:
        R = np.linalg.norm(coords, axis=-1)
        # wz = lim_{theta -> 0} R / tan(theta) = 1
        wz = np.ones_like(R)
        theta = np.deg2rad(self.fov_max)
        is_valid = (R > 1e-6) & (R < self._computeR(theta))
        R = R[is_valid]
        theta = np.full_like(R, theta)
        for i in range(niter):
            # Minimize: (poly(theta) - R)^2
            thetai = [np.square(theta)]
            for j in range(3): thetai.append(thetai[-1] * thetai[0])
            thetai = np.stack(thetai, axis=-1)
            ki_thetai = thetai * self.dist_coeffs
            e = theta * (1 + ki_thetai.sum(axis=-1)) - R
            # grad = (poly(theta) - R) / poly'(theta)
            theta -= e / (1 + ki_thetai @ np.arange(3, 10, 2))
        wz[is_valid] = R / np.tan(theta)
        return wz

    def project(self, pcd: np.ndarray) -> np.ndarray:
        pcd = pcd.astype(np.float32)
        R = self._computeR(np.arctan2(np.linalg.norm(pcd[..., :2], axis=-1, keepdims=True), pcd[..., 2:3]))
        psi = np.arctan2(pcd[..., 1], pcd[..., 0])
        polars = np.stack([np.cos(psi), np.sin(psi)], axis=-1)
        return polars * R * self.intrinsics[:2] + self.intrinsics[2:]


@dataclass
class ObservationSequence:
    root: Path
    img_ext: str = "png"

    def __post_init__(self):
        self.root = Path(self.root)
        (self.root / "color").mkdir(parents=True, exist_ok=True)
        (self.root / "depth").mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(list(self.root.glob(f"color/*.{self.img_ext}")))

    def load(self) -> list[dict]:
        ret = []
        for color in sorted(self.root.glob(f"color/*.{self.img_ext}")):
            t = color.stem
            struct = {"color": color, "depth": self.root / "depth" / color.name, "t": t}

            metadata = self.root / "metadata" / f"{t}.bin"
            if metadata.exists(): struct.update(binary(metadata))
            ret.append(struct)
        return ret

    def save_frame(self,
                   t: float,
                   color: np.ndarray,
                   depth: np.ndarray,
                   **metadata):
        t = f"{t:.6f}"
        img_name = f"{t}.{self.img_ext}"
        cv2.imwrite(str(self.root / "color" / img_name), color)
        cv2.imwrite(str(self.root / "depth" / img_name), depth)
        if metadata:
            f = self.root / "metadata" / f"{t}.bin"
            f.parent.mkdir(parents=True, exist_ok=True)
            binary(f, metadata)


if __name__ == '__main__':
    ROOT = Path("../..").resolve()


    def test_pinhole():
        cam: Pinhole = Camera.from_yaml(ROOT / "config/RS-D435i.yaml")
        print(cam)
        img = cv2.imread(ROOT / "assets/cup2-c.png")

        tar_size = [518, 392]
        cam.resize(tar_size, "scale")
        print(cam)
        print(cam.fov)
        img = cv2.resize(img, tar_size)

        return cam.normalized_plane(img)


    def test_kb():
        cam = Camera.from_yaml(ROOT / "config/Kannala-Brandt.yaml")
        print(cam)
        img = cv2.imread(ROOT / "assets/tum-vi.png")
        return cam.normalized_plane(img)


    plt.imshow(test_pinhole()[..., ::-1]), plt.show()
