from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pyrealsense2 as rs

DEVICES = list(rs.context().query_devices())


def process_img_frame(frame: rs.frame) -> np.ndarray:
    return np.asanyarray(frame.get_data())


def process_motion_frame(frame: rs.frame) -> np.ndarray:
    vec: rs.vector = frame.as_motion_frame().get_motion_data()
    return np.array([vec.x, vec.y, vec.z])


@dataclass
class BasicDevice:
    device_id: int = 0
    max_retry: int = 5
    timeout_ms: int = 1000

    cfg: rs.config = field(default_factory=rs.config, init=False)
    pipe: rs.pipeline = field(default_factory=rs.pipeline, init=False)

    def __post_init__(self):
        self.cfg.enable_device(self.device.get_info(rs.camera_info.serial_number))
        self.profile = self.pipe.start(self.cfg)

    def batch_call(self, duration: float):
        t1 = time.time() + duration
        data = []
        while time.time() < t1:
            data.append(self())
        return [np.stack(x) for x in zip(*data)]

    @property
    def device(self) -> rs.device:
        return DEVICES[self.device_id]

    def get_frames(self) -> rs.composite_frame:
        for i in range(self.max_retry):
            try:
                return self.pipe.wait_for_frames(timeout_ms=self.timeout_ms)
            except RuntimeError as e:
                print(f"Retrying due to error: {e}")
                self.pipe.stop()
                self.profile = self.pipe.start(self.cfg)
                continue
        raise RuntimeError("Failed to get frames.")

    def run(self, callback):
        while callback(self()): pass


@dataclass
class RGBDCamera(BasicDevice):
    img_size: tuple[int, int] = (640, 480)
    fps: int = 30
    use_align: bool = True

    def __post_init__(self):
        self.align = rs.align(rs.stream.color) if self.use_align else None
        self.cfg.enable_stream(rs.stream.color, *self.img_size, rs.format.bgr8, self.fps)
        self.cfg.enable_stream(rs.stream.depth, *self.img_size, rs.format.z16, self.fps)
        super().__post_init__()
        print(f"{__class__.__name__} device-{self.device_id}:", self.cam_info)

    @cached_property
    def cam_info(self) -> dict:
        frame = self.get_frames().get_color_frame()
        data: rs.intrinsics = frame.profile.as_video_stream_profile().intrinsics

        return {
            "name": self.device.get_info(rs.camera_info.name),
            "img_size": self.img_size,
            "intrinsics": [data.fx, data.fy, data.ppx, data.ppy],
            "dist_coeffs": data.coeffs,
            "depth_scale": 1 / self.profile.get_device().first_depth_sensor().get_depth_scale()
        }

    def __call__(self):
        frames = self.get_frames()
        t = frames.get_timestamp()
        if self.align: frames = self.align.process(frames)
        frames = frames.get_color_frame(), frames.get_depth_frame()
        return t, *map(process_img_frame, frames)


@dataclass
class IMU(BasicDevice):
    fps = 200

    def __post_init__(self):
        self.cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, self.fps)
        self.cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, self.fps)
        super().__post_init__()

    def __call__(self):
        frames = self.get_frames()
        t = frames.get_timestamp()
        frames = [frames.first_or_default(s, rs.format.motion_xyz32f) for s in (rs.stream.accel, rs.stream.gyro)]
        return t, *map(process_motion_frame, frames)


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    import open3d as o3d
    from camera import Frame, Camera
    from pathlib import Path

    ROOT = Path("../../..").resolve()

    cam1 = Camera.from_yaml(ROOT / "config/system/RS-D435i.yaml")
    dev1 = RGBDCamera(device_id=0, use_align=False)

    t0 = time.time()
    frame = [None]


    def callback(frames, runtime=3):
        t, color, depth = frames
        frame[0] = Frame(cam1, color, depth)

        plt.imshow(frame[0].concat())
        plt.pause(1e-3)
        return time.time() - t0 <= runtime


    dev1.run(callback)
    plt.close()

    frame: Frame = frame[0]
    o3d.visualization.draw_geometries([frame.to_pcd()])
