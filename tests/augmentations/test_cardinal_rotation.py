"""
Tests for CardinalRotationAugmentation (90/180/270 degree lossless rotations).

Covers:
- Deterministic rotation for each k value (1, 2, 3)
- Coordinate transforms (antenna position, azimuth)
- Spatial consistency between input/output/mask/floor_plan
- Full-cycle identity (4 rotations = original)
- Probability gate (p=0, p=1)
- Edge cases (None fields, square images)
- Statistical distribution of rotation angles
"""
import random
import unittest
from unittest.mock import patch

import torch

from src.utils.indoor.augmentations import (
    BaseAugmentation,
    CardinalRotationAugmentation,
)
from src.utils.indoor.types import RadarSample

from .conftest import make_sample, clone_sample


class TestCardinalRotationAugmentation(unittest.TestCase):
    """Tests for CardinalRotationAugmentation (90/180/270 degree lossless rotations)."""

    # ------------------------------------------------------------------
    # Deterministic rotation tests (one per k value)
    # ------------------------------------------------------------------

    def _apply_rotation_k(self, k: int, sample: RadarSample) -> RadarSample:
        """Force a specific rotation by mocking random.choice."""
        aug = CardinalRotationAugmentation(p=1.0)
        with patch("src.utils.indoor.augmentations.random.choice", return_value=k):
            return aug(sample)

    # --- k = 1  (90 deg CCW) ---

    def test_rotation_90_input_img(self):
        s = make_sample(H=4, W=6)
        r = self._apply_rotation_k(1, s)

        # torch.rot90 k=1 dims=(1,2): new shape (C, W, H) = (3, 6, 4)
        self.assertEqual(r.input_img.shape, (3, 6, 4))
        self.assertEqual(r.H, 6)
        self.assertEqual(r.W, 4)

    def test_rotation_90_output_img(self):
        s = make_sample(H=4, W=6)
        orig_output = s.output_img.clone()
        r = self._apply_rotation_k(1, s)

        self.assertEqual(r.output_img.shape, (6, 4))
        expected = torch.rot90(orig_output, 1, (0, 1))
        self.assertTrue(torch.equal(r.output_img, expected))

    def test_rotation_90_mask(self):
        s = make_sample(H=4, W=6)
        orig_mask = s.mask.clone()
        r = self._apply_rotation_k(1, s)

        expected = torch.rot90(orig_mask, 1, (0, 1))
        self.assertTrue(torch.equal(r.mask, expected))

    def test_rotation_90_floor_plan(self):
        s = make_sample(H=4, W=6)
        orig_fp = s.floor_plan.clone()
        r = self._apply_rotation_k(1, s)

        expected = torch.rot90(orig_fp, 1, (0, 1))
        self.assertTrue(torch.equal(r.floor_plan, expected))

    def test_rotation_90_antenna_coords(self):
        s = make_sample(H=4, W=6, x_ant=2.0, y_ant=1.0)
        r = self._apply_rotation_k(1, s)

        # k=1: new_x = y_ant, new_y = old_W - x_ant - 1
        self.assertEqual(r.x_ant, 1.0)
        self.assertEqual(r.y_ant, 6 - 2.0 - 1)  # 3.0

    def test_rotation_90_azimuth(self):
        s = make_sample(azimuth=45.0)
        r = self._apply_rotation_k(1, s)
        self.assertAlmostEqual(r.azimuth, (45.0 + 90) % 360)

    # --- k = 2  (180 deg) ---

    def test_rotation_180_input_img(self):
        s = make_sample(H=4, W=6)
        r = self._apply_rotation_k(2, s)

        # 180 rotation preserves shape
        self.assertEqual(r.input_img.shape, (3, 4, 6))
        self.assertEqual(r.H, 4)
        self.assertEqual(r.W, 6)

    def test_rotation_180_output_img(self):
        s = make_sample(H=4, W=6)
        orig_output = s.output_img.clone()
        r = self._apply_rotation_k(2, s)

        expected = torch.rot90(orig_output, 2, (0, 1))
        self.assertTrue(torch.equal(r.output_img, expected))

    def test_rotation_180_mask(self):
        s = make_sample(H=4, W=6)
        orig_mask = s.mask.clone()
        r = self._apply_rotation_k(2, s)

        expected = torch.rot90(orig_mask, 2, (0, 1))
        self.assertTrue(torch.equal(r.mask, expected))

    def test_rotation_180_floor_plan(self):
        s = make_sample(H=4, W=6)
        orig_fp = s.floor_plan.clone()
        r = self._apply_rotation_k(2, s)

        expected = torch.rot90(orig_fp, 2, (0, 1))
        self.assertTrue(torch.equal(r.floor_plan, expected))

    def test_rotation_180_antenna_coords(self):
        s = make_sample(H=4, W=6, x_ant=2.0, y_ant=1.0)
        r = self._apply_rotation_k(2, s)

        # k=2: new_x = old_W - x_ant - 1, new_y = old_H - y_ant - 1
        self.assertEqual(r.x_ant, 6 - 2.0 - 1)  # 3.0
        self.assertEqual(r.y_ant, 4 - 1.0 - 1)  # 2.0

    def test_rotation_180_azimuth(self):
        s = make_sample(azimuth=200.0)
        r = self._apply_rotation_k(2, s)
        self.assertAlmostEqual(r.azimuth, (200.0 + 180) % 360)  # 20.0

    # --- k = 3  (270 deg CCW = 90 CW) ---

    def test_rotation_270_input_img(self):
        s = make_sample(H=4, W=6)
        r = self._apply_rotation_k(3, s)

        self.assertEqual(r.input_img.shape, (3, 6, 4))
        self.assertEqual(r.H, 6)
        self.assertEqual(r.W, 4)

    def test_rotation_270_output_img(self):
        s = make_sample(H=4, W=6)
        orig_output = s.output_img.clone()
        r = self._apply_rotation_k(3, s)

        expected = torch.rot90(orig_output, 3, (0, 1))
        self.assertTrue(torch.equal(r.output_img, expected))

    def test_rotation_270_mask(self):
        s = make_sample(H=4, W=6)
        orig_mask = s.mask.clone()
        r = self._apply_rotation_k(3, s)

        expected = torch.rot90(orig_mask, 3, (0, 1))
        self.assertTrue(torch.equal(r.mask, expected))

    def test_rotation_270_floor_plan(self):
        s = make_sample(H=4, W=6)
        orig_fp = s.floor_plan.clone()
        r = self._apply_rotation_k(3, s)

        expected = torch.rot90(orig_fp, 3, (0, 1))
        self.assertTrue(torch.equal(r.floor_plan, expected))

    def test_rotation_270_antenna_coords(self):
        s = make_sample(H=4, W=6, x_ant=2.0, y_ant=1.0)
        r = self._apply_rotation_k(3, s)

        # k=3: new_x = old_H - y_ant - 1, new_y = x_ant
        self.assertEqual(r.x_ant, 4 - 1.0 - 1)  # 2.0
        self.assertEqual(r.y_ant, 2.0)

    def test_rotation_270_azimuth(self):
        s = make_sample(azimuth=100.0)
        r = self._apply_rotation_k(3, s)
        self.assertAlmostEqual(r.azimuth, (100.0 + 270) % 360)  # 10.0

    # ------------------------------------------------------------------
    # Spatial consistency: the antenna pixel value in input_img and
    # output_img should follow the antenna after rotation.
    # ------------------------------------------------------------------

    def test_antenna_pixel_follows_rotation(self):
        """
        Place a unique marker at the antenna position in both input and
        output, rotate, and verify the marker moved to the new antenna
        position.
        """
        H, W = 8, 8
        x_ant, y_ant = 5, 3  # col=5, row=3
        MARKER = 999.0

        for k in [1, 2, 3]:
            with self.subTest(k=k):
                s = make_sample(H=H, W=W, x_ant=float(x_ant), y_ant=float(y_ant))

                # Plant marker at antenna location
                s.input_img[:, y_ant, x_ant] = MARKER
                s.output_img[y_ant, x_ant] = MARKER

                r = self._apply_rotation_k(k, s)

                new_x = int(r.x_ant)
                new_y = int(r.y_ant)

                # The marker should now live at the new antenna position
                for ch in range(3):
                    self.assertAlmostEqual(
                        r.input_img[ch, new_y, new_x].item(),
                        MARKER,
                        msg=f"k={k}: input_img channel {ch} marker not at new antenna pos",
                    )
                self.assertAlmostEqual(
                    r.output_img[new_y, new_x].item(),
                    MARKER,
                    msg=f"k={k}: output_img marker not at new antenna pos",
                )

    # ------------------------------------------------------------------
    # Full-cycle: rotating 4 times by the same k should restore original
    # ------------------------------------------------------------------

    def test_four_rotations_restore_original(self):
        """Rotating by the same k four times must return the original sample."""
        for k in [1, 2, 3]:
            with self.subTest(k=k):
                s = make_sample(H=6, W=10, x_ant=4.0, y_ant=2.0, azimuth=30.0)
                orig_input = s.input_img.clone()
                orig_output = s.output_img.clone()
                orig_mask = s.mask.clone()
                orig_fp = s.floor_plan.clone()
                orig_x, orig_y = s.x_ant, s.y_ant
                orig_az = s.azimuth

                for _ in range(4):
                    s = self._apply_rotation_k(k, s)

                self.assertTrue(torch.equal(s.input_img, orig_input))
                self.assertTrue(torch.equal(s.output_img, orig_output))
                self.assertTrue(torch.equal(s.mask, orig_mask))
                self.assertTrue(torch.equal(s.floor_plan, orig_fp))
                self.assertAlmostEqual(s.x_ant, orig_x)
                self.assertAlmostEqual(s.y_ant, orig_y)
                self.assertAlmostEqual(s.azimuth % 360, orig_az % 360)

    # ------------------------------------------------------------------
    # Probability gate
    # ------------------------------------------------------------------

    def test_p_zero_never_augments(self):
        aug = CardinalRotationAugmentation(p=0.0)
        s = make_sample()
        orig_input = s.input_img.clone()

        for _ in range(50):
            s_copy = make_sample()
            result = aug(s_copy)
            self.assertTrue(torch.equal(result.input_img, orig_input))

    def test_p_one_always_augments(self):
        """With p=1.0 and 3 possible rotations (90/180/270), every call must
        produce a rotated result (never identity)."""
        aug = CardinalRotationAugmentation(p=1.0)
        identical_count = 0
        N = 50

        for _ in range(N):
            s = make_sample(H=4, W=6)
            orig_input = s.input_img.clone()
            result = aug(s)
            if torch.equal(result.input_img, orig_input):
                identical_count += 1

        # With a non-square input, no rotation (90/180/270) can produce
        # the identical tensor, so identical_count should be 0.
        self.assertEqual(identical_count, 0)

    # ------------------------------------------------------------------
    # Edge cases: None / missing fields
    # ------------------------------------------------------------------

    def test_none_output_img(self):
        """Augmentation must not crash when output_img is None (kaggle test data)."""
        for k in [1, 2, 3]:
            with self.subTest(k=k):
                s = make_sample(with_output=False)
                r = self._apply_rotation_k(k, s)
                self.assertIsNone(r.output_img)

    def test_none_floor_plan(self):
        """Augmentation must not crash when floor_plan is None."""
        for k in [1, 2, 3]:
            with self.subTest(k=k):
                s = make_sample(with_floor_plan=False)
                r = self._apply_rotation_k(k, s)
                self.assertIsNone(r.floor_plan)

    # ------------------------------------------------------------------
    # Square image: verify dimensions don't change
    # ------------------------------------------------------------------

    def test_square_image_preserves_dimensions(self):
        for k in [1, 2, 3]:
            with self.subTest(k=k):
                s = make_sample(H=8, W=8)
                r = self._apply_rotation_k(k, s)
                self.assertEqual(r.H, 8)
                self.assertEqual(r.W, 8)
                self.assertEqual(r.input_img.shape, (3, 8, 8))

    # ------------------------------------------------------------------
    # Mask zero-pixel marker tracks correctly
    # ------------------------------------------------------------------

    def test_mask_zero_pixel_tracks_rotation(self):
        """The single zero in the mask at (0,0) should move to the expected
        position after each rotation."""
        H, W = 6, 10

        expected_zero_pos = {
            # (row, col) of the zero pixel after rotation
            1: (W - 1, 0),       # k=1: (0,0) -> row=W-0-1, col=0
            2: (H - 1, W - 1),   # k=2: (0,0) -> row=H-1, col=W-1
            3: (0, H - 1),       # k=3: (0,0) -> row=0, col=H-1
        }

        for k, (exp_row, exp_col) in expected_zero_pos.items():
            with self.subTest(k=k):
                s = make_sample(H=H, W=W)
                r = self._apply_rotation_k(k, s)

                self.assertEqual(
                    r.mask[exp_row, exp_col].item(),
                    0.0,
                    msg=f"k={k}: expected mask zero at ({exp_row},{exp_col})",
                )
                # And total number of zeros should still be 1
                self.assertEqual((r.mask == 0).sum().item(), 1)


class TestBaseAugmentation(unittest.TestCase):
    """Test the BaseAugmentation interface contract."""

    def test_not_implemented(self):
        base = BaseAugmentation()
        with self.assertRaises(NotImplementedError):
            base(make_sample())


class TestCardinalRotationStatistical(unittest.TestCase):
    """Statistical tests to verify uniform distribution of rotation angles."""

    def test_uniform_distribution_of_k(self):
        """Over many calls, each k (1, 2, 3) should be chosen roughly equally."""
        aug = CardinalRotationAugmentation(p=1.0)
        H, W = 4, 6  # non-square so we can distinguish k by output shape

        shape_counts = {}
        N = 600

        for _ in range(N):
            s = make_sample(H=H, W=W)
            r = aug(s)
            key = (r.H, r.W)
            shape_counts[key] = shape_counts.get(key, 0) + 1

        # k=1 and k=3 produce (W, H) = (6, 4), k=2 produces (H, W) = (4, 6)
        # So we expect ~2/3 of samples to have shape (6, 4) and ~1/3 to have (4, 6)
        count_swapped = shape_counts.get((W, H), 0)  # k=1 or k=3
        count_same = shape_counts.get((H, W), 0)  # k=2

        # With N=600, expected: swapped=400, same=200. Allow generous margin.
        self.assertGreater(count_swapped, 250, "k=1/k=3 combined should appear often")
        self.assertGreater(count_same, 100, "k=2 should appear reasonably often")
        self.assertEqual(count_swapped + count_same, N)


if __name__ == "__main__":
    unittest.main(verbosity=2)
