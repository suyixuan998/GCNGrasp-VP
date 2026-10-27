"""Regression coverage for invalid depth and padding in mask refinement."""
import unittest
import warnings

import numpy as np
import supervision as sv

from gcngrasp.api.utils.sv_extension import refine_masks_by_ns


class MaskRefinementTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((7, 7))
        self.points = np.stack([x * .003, y * .003, np.ones_like(x)], axis=-1).astype(np.float32)
        self.mask = np.ones((7, 7), dtype=bool)

    def expected_mask(self):
        expected = np.zeros_like(self.mask)
        valid = self.mask & np.isfinite(self.points).all(axis=-1) & (self.points[..., 2] > 0)
        for y, x in np.argwhere(valid):
            count = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if (dy == 0 and dx == 0) or not (0 <= yy < 7 and 0 <= xx < 7):
                        continue
                    if valid[yy, xx] and np.linalg.norm(self.points[yy, xx] - self.points[y, x]) <= .02:
                        count += 1
            expected[y, x] = count >= 7
        return expected

    def check_refinement(self):
        detections = sv.Detections(xyxy=np.array([[0, 0, 7, 7]]), mask=self.mask[None].copy())
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = refine_masks_by_ns(detections, self.points)
        np.testing.assert_array_equal(result.mask[0], self.expected_mask())

    def test_dense_surface_and_padded_border(self):
        self.check_refinement()

    def test_zero_negative_nan_infinity_and_mask_holes(self):
        self.points[1, 1] = 0
        self.points[2, 2, 2] = -1
        self.points[3, 3, 0] = np.nan
        self.points[4, 4] = np.inf
        self.mask[1, 4] = False
        self.check_refinement()

    def test_all_invalid_depth(self):
        self.points[:] = 0
        self.check_refinement()


if __name__ == "__main__":
    unittest.main()
