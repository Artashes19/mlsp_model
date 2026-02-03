"""
Tests for WallInsertionAugmentation (synthetic wall insertion).

Covers:
- Probability gate (p=0, p=1)
- Transmittance monotonicity (walls only add)
- Output pathloss monotonicity
- Transmittance range constraints
- Wall density constraints
- Metadata preservation (antenna coords, dimensions, mask, non-transmittance channels)
- Masked-only updates (walls only added where mask == 1)
- Edge cases (None output, empty mask)
"""
import unittest

import torch

from src.utils.indoor.augmentations import WallInsertionAugmentation
from src.utils.indoor.types import RadarSample

from .conftest import make_sample, clone_sample


class TestWallInsertionAugmentation(unittest.TestCase):
    """Tests for WallInsertionAugmentation (synthetic wall insertion)."""

    def test_probability_zero_never_augments(self):
        """With p=0.0, the augmentation should never be applied."""
        aug = WallInsertionAugmentation(p=0.0)
        s = make_sample(H=8, W=8)
        orig_transmittance = s.input_img[1].clone()
        orig_output = s.output_img.clone()

        for _ in range(20):
            s_copy = clone_sample(s)
            result = aug(s_copy)
            self.assertTrue(torch.equal(result.input_img[1], orig_transmittance))
            self.assertTrue(torch.equal(result.output_img, orig_output))

    def test_probability_one_always_augments(self):
        """With p=1.0, the augmentation should always modify the sample."""
        aug = WallInsertionAugmentation(p=1.0, transmittance_range=(2, 18))
        modified_count = 0
        N = 20

        for _ in range(N):
            s = make_sample(H=8, W=8)
            orig_transmittance = s.input_img[1].clone()
            result = aug(s)

            # Check if transmittance was modified (walls added)
            if not torch.equal(result.input_img[1], orig_transmittance):
                modified_count += 1

        # Most runs should have added walls
        self.assertGreater(modified_count, N // 2)

    def test_transmittance_increases(self):
        """Wall insertion should only increase transmittance values."""
        aug = WallInsertionAugmentation(p=1.0, transmittance_range=(5, 15))
        s = make_sample(H=10, W=10)
        orig_transmittance = s.input_img[1].clone()

        for _ in range(10):
            s_copy = clone_sample(s)
            result = aug(s_copy)

            # New transmittance should be >= original everywhere
            diff = result.input_img[1] - orig_transmittance
            self.assertTrue((diff >= -1e-5).all(), "Transmittance should not decrease")

    def test_output_pathloss_increases(self):
        """Adding walls should increase pathloss (or keep it the same)."""
        aug = WallInsertionAugmentation(p=1.0, transmittance_range=(5, 15))
        s = make_sample(H=10, W=10)
        orig_output = s.output_img.clone()

        for _ in range(10):
            s_copy = clone_sample(s)
            result = aug(s_copy)

            # New pathloss should be >= original everywhere
            diff = result.output_img - orig_output
            self.assertTrue((diff >= -1e-5).all(), "Pathloss should not decrease")

    def test_transmittance_range_respected(self):
        """Inserted walls should have transmittance values within the expected range."""
        min_val, max_val = 10, 20
        aug = WallInsertionAugmentation(
            p=1.0, transmittance_range=(min_val, max_val), max_wall_density=0.5
        )

        for _ in range(10):
            s = make_sample(H=8, W=8)
            orig_transmittance = s.input_img[1].clone()
            result = aug(s)

            # Find added walls (pixels where transmittance increased)
            added_walls = result.input_img[1] - orig_transmittance
            new_wall_pixels = added_walls > 0

            if new_wall_pixels.any():
                new_values = added_walls[new_wall_pixels]
                self.assertTrue(
                    (new_values >= min_val - 0.1).all(),
                    f"Some wall values below min: {new_values.min()}",
                )
                self.assertTrue(
                    (new_values < max_val + 0.1).all(),
                    f"Some wall values above max: {new_values.max()}",
                )

    def test_none_output_img_handled(self):
        """Augmentation should handle samples with None output_img gracefully."""
        aug = WallInsertionAugmentation(p=1.0)
        s = make_sample(H=8, W=8, with_output=False)

        # Should not crash
        result = aug(s)
        self.assertIsNone(result.output_img)

    def test_empty_sample_handled(self):
        """Augmentation should handle samples with all-zero mask."""
        aug = WallInsertionAugmentation(p=1.0)
        s = make_sample(H=8, W=8)
        s.mask = torch.zeros_like(s.mask)  # All invalid

        # Should not crash and should return unchanged
        orig_transmittance = s.input_img[1].clone()
        result = aug(s)
        self.assertTrue(torch.equal(result.input_img[1], orig_transmittance))

    def test_wall_density_constraint(self):
        """Number of wall pixels should respect max_wall_density."""
        density = 0.2
        aug = WallInsertionAugmentation(
            p=1.0, transmittance_range=(10, 20), max_wall_density=density
        )

        for _ in range(5):
            s = make_sample(H=12, W=12)
            orig_transmittance = s.input_img[1].clone()
            result = aug(s)

            # Count pixels where walls were added
            added_walls = result.input_img[1] - orig_transmittance
            wall_pixels = (added_walls > 0).sum().item()
            total_pixels = (s.mask == 1).sum().item()

            # Wall pixels should not exceed density * total_pixels
            max_allowed = int(total_pixels * density) + 10  # +10 for tolerance
            self.assertLessEqual(
                wall_pixels,
                max_allowed,
                f"Wall pixels {wall_pixels} exceeds max {max_allowed}",
            )

    def test_antenna_coords_unchanged(self):
        """Wall insertion should not modify antenna coordinates."""
        aug = WallInsertionAugmentation(p=1.0)
        s = make_sample(H=8, W=8, x_ant=4.0, y_ant=3.0)
        orig_x, orig_y = s.x_ant, s.y_ant

        result = aug(s)
        self.assertEqual(result.x_ant, orig_x)
        self.assertEqual(result.y_ant, orig_y)

    def test_dimensions_unchanged(self):
        """Wall insertion should not change sample dimensions."""
        aug = WallInsertionAugmentation(p=1.0)
        s = make_sample(H=10, W=12)
        orig_H, orig_W = s.H, s.W

        result = aug(s)
        self.assertEqual(result.H, orig_H)
        self.assertEqual(result.W, orig_W)
        self.assertEqual(result.input_img.shape, (3, orig_H, orig_W))

    def test_mask_unchanged(self):
        """Wall insertion should not modify the mask."""
        aug = WallInsertionAugmentation(p=1.0)
        s = make_sample(H=8, W=8)
        orig_mask = s.mask.clone()

        result = aug(s)
        self.assertTrue(torch.equal(result.mask, orig_mask))

    def test_reflectance_and_distance_unchanged(self):
        """Wall insertion should only modify transmittance and output, not reflectance or distance."""
        aug = WallInsertionAugmentation(p=1.0)
        s = make_sample(H=8, W=8)
        orig_reflectance = s.input_img[0].clone()
        orig_distance = s.input_img[2].clone()

        result = aug(s)
        self.assertTrue(torch.equal(result.input_img[0], orig_reflectance))
        self.assertTrue(torch.equal(result.input_img[2], orig_distance))

    def test_walls_only_added_in_masked_region(self):
        """Walls should only be added where mask == 1, not in invalid regions."""
        aug = WallInsertionAugmentation(p=1.0, transmittance_range=(5, 15))

        for _ in range(10):
            s = make_sample(H=10, W=10)
            # Create a more restrictive mask (only center region is valid)
            s.mask = torch.zeros(10, 10, dtype=torch.float32)
            s.mask[2:8, 2:8] = 1.0  # Only center 6x6 is valid

            orig_transmittance = s.input_img[1].clone()
            result = aug(s)

            # Check that transmittance outside mask is unchanged
            outside_mask = s.mask == 0
            transmittance_diff_outside = result.input_img[1][outside_mask] - orig_transmittance[outside_mask]
            self.assertTrue(
                (transmittance_diff_outside.abs() < 1e-5).all(),
                f"Transmittance changed outside mask: {transmittance_diff_outside.abs().max()}",
            )

    def test_output_only_updated_in_masked_region(self):
        """Output pathloss should only be updated where mask == 1."""
        aug = WallInsertionAugmentation(p=1.0, transmittance_range=(5, 15))

        for _ in range(10):
            s = make_sample(H=10, W=10)
            # Create a more restrictive mask
            s.mask = torch.zeros(10, 10, dtype=torch.float32)
            s.mask[2:8, 2:8] = 1.0

            orig_output = s.output_img.clone()
            result = aug(s)

            # Check that output outside mask is unchanged
            outside_mask = s.mask == 0
            output_diff_outside = result.output_img[outside_mask] - orig_output[outside_mask]
            self.assertTrue(
                (output_diff_outside.abs() < 1e-5).all(),
                f"Output changed outside mask: {output_diff_outside.abs().max()}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
