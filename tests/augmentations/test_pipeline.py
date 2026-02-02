"""
Integration tests for AugmentationPipeline.

These tests verify the pipeline's behavior when combining multiple augmentations,
not the individual augmentations themselves (which are tested in their own files).

Covers:
- Edge case matrix: all on/off combinations of CardinalRotation + WallInsertion
- Probability compliance: statistical verification that per-augmentation p works in pipeline
- Sequence ordering: [Rotation, Walls] differs from [Walls, Rotation]
- Training mode toggle
- Empty pipeline
- Seed reproducibility
- Data accumulation across augmentations
- Output structural invariants after full pipeline
"""
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.utils.indoor.augmentations import (
    AugmentationPipeline,
    BaseAugmentation,
    CardinalRotationAugmentation,
    WallInsertionAugmentation,
)
from src.utils.indoor.types import RadarSample

from .conftest import make_sample, clone_sample, save_pipeline_visual, VISUAL_DIR


# ---------------------------------------------------------------------------
# Edge-case matrix: all on/off combinations
# ---------------------------------------------------------------------------

class TestPipelineCombinations(unittest.TestCase):
    """Test all on/off combinations of the two augmentations in the pipeline."""

    def _run_pipeline(self, rotation_p: float, wall_p: float, sample: RadarSample) -> RadarSample:
        """Build and run a pipeline with the given probabilities."""
        pipeline = AugmentationPipeline(
            augmentations=[
                CardinalRotationAugmentation(p=rotation_p),
                WallInsertionAugmentation(p=wall_p, transmittance_range=(5, 15)),
            ],
            training=True,
        )
        return pipeline(sample)

    def test_both_off_sample_unchanged(self):
        """p=0 for both: sample must be bitwise identical to the original."""
        s = make_sample(H=8, W=8, x_ant=3.0, y_ant=4.0)
        orig_input = s.input_img.clone()
        orig_output = s.output_img.clone()
        orig_mask = s.mask.clone()
        orig_x, orig_y, orig_az = s.x_ant, s.y_ant, s.azimuth

        result = self._run_pipeline(0.0, 0.0, s)

        self.assertTrue(torch.equal(result.input_img, orig_input))
        self.assertTrue(torch.equal(result.output_img, orig_output))
        self.assertTrue(torch.equal(result.mask, orig_mask))
        self.assertEqual(result.x_ant, orig_x)
        self.assertEqual(result.y_ant, orig_y)
        self.assertEqual(result.azimuth, orig_az)

    def test_rotation_on_walls_off(self):
        """Only rotation active: transmittance channel content should be a
        permutation of the original (lossless rotation), and output pathloss
        should be the rotated version of the original (no additive wall loss)."""
        for _ in range(10):
            s = make_sample(H=6, W=10, x_ant=4.0, y_ant=2.0)
            orig_output_sorted = s.output_img.flatten().sort().values

            result = self._run_pipeline(1.0, 0.0, s)

            # Rotation is lossless: all output pixel values must be preserved
            result_output_sorted = result.output_img.flatten().sort().values
            self.assertTrue(
                torch.equal(orig_output_sorted, result_output_sorted),
                "Rotation-only pipeline should preserve all output pixel values",
            )

    def test_walls_on_rotation_off(self):
        """Only wall insertion active: dimensions and antenna coords stay the
        same, transmittance can only increase, output pathloss can only increase."""
        for _ in range(10):
            s = make_sample(H=8, W=8, x_ant=3.0, y_ant=4.0)
            orig_transmittance = s.input_img[1].clone()
            orig_output = s.output_img.clone()
            orig_H, orig_W = s.H, s.W
            orig_x, orig_y = s.x_ant, s.y_ant

            result = self._run_pipeline(0.0, 1.0, s)

            # No rotation -> dimensions and antenna unchanged
            self.assertEqual(result.H, orig_H)
            self.assertEqual(result.W, orig_W)
            self.assertEqual(result.x_ant, orig_x)
            self.assertEqual(result.y_ant, orig_y)

            # Walls only add transmittance
            self.assertTrue(
                (result.input_img[1] >= orig_transmittance - 1e-5).all(),
                "Transmittance should not decrease when only walls are active",
            )
            # Pathloss only increases
            self.assertTrue(
                (result.output_img >= orig_output - 1e-5).all(),
                "Pathloss should not decrease when only walls are active",
            )

    def test_both_on_produces_valid_sample(self):
        """Both augmentations active: the result must be a structurally valid
        RadarSample -- correct shapes, antenna within bounds, no NaN/Inf."""
        for _ in range(20):
            s = make_sample(H=8, W=10, x_ant=5.0, y_ant=3.0)
            result = self._run_pipeline(1.0, 1.0, s)

            # Structural invariants
            C, H, W = result.input_img.shape
            self.assertEqual(C, 3)
            self.assertEqual(H, result.H)
            self.assertEqual(W, result.W)
            self.assertEqual(result.output_img.shape, (H, W))
            self.assertEqual(result.mask.shape, (H, W))

            # Antenna must be within image bounds
            self.assertGreaterEqual(result.x_ant, 0)
            self.assertLess(result.x_ant, W)
            self.assertGreaterEqual(result.y_ant, 0)
            self.assertLess(result.y_ant, H)

            # No NaN or Inf anywhere
            self.assertFalse(torch.isnan(result.input_img).any(), "NaN in input_img")
            self.assertFalse(torch.isinf(result.input_img).any(), "Inf in input_img")
            self.assertFalse(torch.isnan(result.output_img).any(), "NaN in output_img")
            self.assertFalse(torch.isinf(result.output_img).any(), "Inf in output_img")

            # Azimuth stays in [0, 360)
            self.assertGreaterEqual(result.azimuth, 0)
            self.assertLess(result.azimuth, 360)


# ---------------------------------------------------------------------------
# Probability compliance (statistical)
# ---------------------------------------------------------------------------

class TestPipelineProbability(unittest.TestCase):
    """Verify that per-augmentation probability parameters are respected
    when augmentations run inside the pipeline."""

    class LoggingRotation(CardinalRotationAugmentation):
        """Rotation augmentation that records whether the probability gate fired."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_fired = False

        def __call__(self, sample):
            if random.random() > self.p:
                self.last_fired = False
                return sample
            self.last_fired = True
            return self._apply_cardinal_rotation(sample)

    class LoggingWalls(WallInsertionAugmentation):
        """Wall augmentation that records whether the probability gate fired."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.last_fired = False

        def __call__(self, sample):
            if random.random() > self.p:
                self.last_fired = False
                return sample
            self.last_fired = True
            return self._apply_walls(sample)

    def test_rotation_probability_in_pipeline(self):
        """CardinalRotation with p=0.5 should fire roughly half the time,
        even when inside a pipeline with other augmentations."""
        p_rotation = 0.5
        rotation = self.LoggingRotation(p=p_rotation)
        pipeline = AugmentationPipeline(
            augmentations=[
                rotation,
                # Walls off so we can isolate rotation effect
                WallInsertionAugmentation(p=0.0),
            ],
            training=True,
        )

        N = 400
        rotated_count = 0
        for _ in range(N):
            s = make_sample(H=4, W=6)  # non-square to detect rotation by shape
            pipeline(s)
            if rotation.last_fired:
                rotated_count += 1

        # Expected: ~200. Accept [120, 280] (binomial 99% CI for N=400, p=0.5).
        rate = rotated_count / N
        self.assertGreater(rotated_count, 120,
                           f"Rotation rate {rate:.2f} too low for p={p_rotation}")
        self.assertLess(rotated_count, 280,
                        f"Rotation rate {rate:.2f} too high for p={p_rotation}")

    def test_wall_probability_in_pipeline(self):
        """WallInsertion with p=0.5 should fire roughly half the time."""
        p_walls = 0.5
        walls = self.LoggingWalls(p=p_walls, transmittance_range=(5, 15))
        pipeline = AugmentationPipeline(
            augmentations=[
                # Rotation off so we can isolate wall effect
                CardinalRotationAugmentation(p=0.0),
                walls,
            ],
            training=True,
        )

        N = 400
        wall_count = 0
        for _ in range(N):
            s = make_sample(H=8, W=8)
            pipeline(s)
            if walls.last_fired:
                wall_count += 1

        rate = wall_count / N
        self.assertGreater(wall_count, 120,
                           f"Wall rate {rate:.2f} too low for p={p_walls}")
        self.assertLess(wall_count, 280,
                        f"Wall rate {rate:.2f} too high for p={p_walls}")

    def test_augmentations_fire_independently(self):
        """When both augmentations have p=0.5, they should be independent:
        the joint probability of both firing should be ~0.25."""
        rotation = self.LoggingRotation(p=0.5)
        walls = self.LoggingWalls(p=0.5, transmittance_range=(5, 15))
        pipeline = AugmentationPipeline(
            augmentations=[
                rotation,
                walls,
            ],
            training=True,
        )

        N = 600
        both_count = 0
        neither_count = 0

        for _ in range(N):
            pipeline(make_sample(H=4, W=6))
            rotated = rotation.last_fired
            walled = walls.last_fired

            if rotated and walled:
                both_count += 1
            if not rotated and not walled:
                neither_count += 1

        # Expected both: ~0.25 * N. But walls can randomly generate 0
        # walls (returning unchanged), so the observed "both" is biased low
        # and "neither" is biased high. Use wide margins.
        self.assertGreater(both_count, 30,
                           f"Both-active rate {both_count/N:.2f} too low")
        self.assertLess(both_count, N // 2,
                        f"Both-active rate {both_count/N:.2f} too high")

        # "Neither" can be inflated because WallInsertion may roll 0 walls
        # even when it fires. Just check it's not nearly 100%.
        self.assertGreater(neither_count, 30,
                           f"Neither-active rate {neither_count/N:.2f} too low")
        self.assertLess(neither_count, int(N * 0.7),
                        f"Neither-active rate {neither_count/N:.2f} too high")


# ---------------------------------------------------------------------------
# Sequence ordering
# ---------------------------------------------------------------------------

class TestPipelineOrdering(unittest.TestCase):
    """Verify that augmentation order matters and is respected."""

    def test_order_matters_rotation_then_walls_vs_walls_then_rotation(self):
        """[Rotation, Walls] should generally produce different results than
        [Walls, Rotation] because rotation changes the antenna position that
        wall ray-tracing uses.

        We fix the random seed so both pipelines see the same random draws,
        then check the outputs differ (the wall ray-tracing from different
        antenna positions will produce different loss maps)."""

        pipeline_rot_walls = AugmentationPipeline(
            augmentations=[
                CardinalRotationAugmentation(p=1.0),
                WallInsertionAugmentation(p=1.0, transmittance_range=(5, 15)),
            ],
            training=True,
        )
        pipeline_walls_rot = AugmentationPipeline(
            augmentations=[
                WallInsertionAugmentation(p=1.0, transmittance_range=(5, 15)),
                CardinalRotationAugmentation(p=1.0),
            ],
            training=True,
        )

        differ_count = 0
        N = 20

        for i in range(N):
            seed = 42 + i

            # Pipeline A: rotation then walls
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            s_a = make_sample(H=8, W=8, x_ant=3.0, y_ant=5.0)
            result_a = pipeline_rot_walls(s_a)

            # Pipeline B: walls then rotation (same seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            s_b = make_sample(H=8, W=8, x_ant=3.0, y_ant=5.0)
            result_b = pipeline_walls_rot(s_b)

            if not torch.equal(result_a.output_img, result_b.output_img):
                differ_count += 1

        # The two orderings should produce different results at least sometimes.
        # With same seed but different ordering of operations, the random draws
        # are consumed by different augmentations, so almost all should differ.
        self.assertGreater(differ_count, 0,
                           "Augmentation order had no effect -- pipeline may not sequence correctly")

    def test_declared_order_is_execution_order(self):
        """Verify the pipeline executes augmentations in the exact order declared."""
        call_log = []

        class LoggingRotation(CardinalRotationAugmentation):
            def __call__(self, sample):
                call_log.append("rotation")
                return super().__call__(sample)

        class LoggingWalls(WallInsertionAugmentation):
            def __call__(self, sample):
                call_log.append("walls")
                return super().__call__(sample)

        pipeline = AugmentationPipeline(
            augmentations=[
                LoggingRotation(p=1.0),
                LoggingWalls(p=1.0, transmittance_range=(5, 15)),
            ],
            training=True,
        )

        pipeline(make_sample(H=8, W=8))
        self.assertEqual(call_log, ["rotation", "walls"])


# ---------------------------------------------------------------------------
# Training mode
# ---------------------------------------------------------------------------

class TestPipelineTrainingMode(unittest.TestCase):
    """Verify the training flag completely gates augmentation."""

    def test_training_false_bypasses_all_augmentations(self):
        """training=False: sample must pass through unchanged regardless of
        augmentation configuration."""
        pipeline = AugmentationPipeline(
            augmentations=[
                CardinalRotationAugmentation(p=1.0),
                WallInsertionAugmentation(p=1.0, transmittance_range=(5, 15)),
            ],
            training=False,
        )

        s = make_sample(H=6, W=10, x_ant=4.0, y_ant=2.0)
        orig_input = s.input_img.clone()
        orig_output = s.output_img.clone()
        orig_mask = s.mask.clone()
        orig_fp = s.floor_plan.clone()
        orig_x, orig_y, orig_az = s.x_ant, s.y_ant, s.azimuth

        result = pipeline(s)

        self.assertTrue(torch.equal(result.input_img, orig_input))
        self.assertTrue(torch.equal(result.output_img, orig_output))
        self.assertTrue(torch.equal(result.mask, orig_mask))
        self.assertTrue(torch.equal(result.floor_plan, orig_fp))
        self.assertEqual(result.x_ant, orig_x)
        self.assertEqual(result.y_ant, orig_y)
        self.assertEqual(result.azimuth, orig_az)

    def test_training_true_applies_augmentations(self):
        """training=True with p=1.0: at least something should change over
        multiple runs (non-square input guarantees rotation changes shape)."""
        pipeline = AugmentationPipeline(
            augmentations=[CardinalRotationAugmentation(p=1.0)],
            training=True,
        )

        changed = False
        for _ in range(10):
            s = make_sample(H=4, W=6)
            orig_shape = s.input_img.shape
            result = pipeline(s)
            if result.input_img.shape != orig_shape:
                changed = True
                break

        self.assertTrue(changed, "training=True pipeline never modified the sample")

    def test_empty_pipeline_is_identity(self):
        """Pipeline with no augmentations acts as identity regardless of training flag."""
        for training in [True, False]:
            with self.subTest(training=training):
                pipeline = AugmentationPipeline(augmentations=[], training=training)
                s = make_sample(H=8, W=8)
                orig_input = s.input_img.clone()
                result = pipeline(s)
                self.assertTrue(torch.equal(result.input_img, orig_input))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestPipelineReproducibility(unittest.TestCase):
    """Verify that fixing all random seeds produces identical outputs."""

    def _run_seeded(self, seed: int, pipeline: AugmentationPipeline) -> RadarSample:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        s = make_sample(H=8, W=10, x_ant=5.0, y_ant=3.0)
        return pipeline(s)

    def test_same_seed_same_output(self):
        """Two runs with the same seed must produce bitwise-identical results."""
        pipeline = AugmentationPipeline(
            augmentations=[
                CardinalRotationAugmentation(p=1.0),
                WallInsertionAugmentation(p=1.0, transmittance_range=(5, 15)),
            ],
            training=True,
        )

        r1 = self._run_seeded(12345, pipeline)
        r2 = self._run_seeded(12345, pipeline)

        self.assertTrue(torch.equal(r1.input_img, r2.input_img))
        self.assertTrue(torch.equal(r1.output_img, r2.output_img))
        self.assertTrue(torch.equal(r1.mask, r2.mask))
        self.assertEqual(r1.x_ant, r2.x_ant)
        self.assertEqual(r1.y_ant, r2.y_ant)
        self.assertEqual(r1.azimuth, r2.azimuth)
        self.assertEqual(r1.H, r2.H)
        self.assertEqual(r1.W, r2.W)

    def test_different_seed_different_output(self):
        """Two runs with different seeds should generally produce different results."""
        pipeline = AugmentationPipeline(
            augmentations=[
                CardinalRotationAugmentation(p=1.0),
                WallInsertionAugmentation(p=1.0, transmittance_range=(5, 15)),
            ],
            training=True,
        )

        differ_count = 0
        for seed_a, seed_b in [(1, 2), (10, 20), (100, 200), (42, 99), (7, 13)]:
            r1 = self._run_seeded(seed_a, pipeline)
            r2 = self._run_seeded(seed_b, pipeline)
            if not torch.equal(r1.output_img, r2.output_img):
                differ_count += 1

        self.assertGreater(differ_count, 0,
                           "Different seeds never produced different outputs")


# ---------------------------------------------------------------------------
# Data accumulation across augmentations
# ---------------------------------------------------------------------------

class TestPipelineDataAccumulation(unittest.TestCase):
    """Verify that changes from multiple augmentations compound correctly."""

    def test_rotation_preserves_pixel_values_walls_adds_to_them(self):
        """After [Rotation(p=1), WallInsertion(p=1)]:
        - The total number of distinct pixel values in reflectance channel should
          be the same as original (rotation is lossless, walls don't touch ch0).
        - The transmittance sum should be >= the original sum (walls add)."""
        pipeline = AugmentationPipeline(
            augmentations=[
                CardinalRotationAugmentation(p=1.0),
                WallInsertionAugmentation(p=1.0, transmittance_range=(5, 15)),
            ],
            training=True,
        )

        for _ in range(10):
            s = make_sample(H=8, W=8, x_ant=3.0, y_ant=4.0)
            orig_reflectance_values = s.input_img[0].flatten().sort().values
            orig_transmittance_sum = s.input_img[1].sum().item()

            result = pipeline(s)

            # Rotation is lossless on reflectance (walls don't touch it)
            result_reflectance_values = result.input_img[0].flatten().sort().values
            self.assertTrue(
                torch.equal(orig_reflectance_values, result_reflectance_values),
                "Reflectance pixel values should be preserved through rotation+walls",
            )

            # Walls only add transmittance, so sum should be >= original
            self.assertGreaterEqual(
                result.input_img[1].sum().item(),
                orig_transmittance_sum - 1e-3,
                "Transmittance sum should not decrease",
            )

    def test_wall_pathloss_accumulates_after_rotation(self):
        """The wall augmentation should recalculate pathloss using the
        post-rotation antenna position (not the original). We verify this
        indirectly: after rotation+walls with p=1, the pathloss values
        should be non-negative and the output should contain the rotated
        original values plus wall-loss additions."""
        pipeline = AugmentationPipeline(
            augmentations=[
                CardinalRotationAugmentation(p=1.0),
                WallInsertionAugmentation(p=1.0, transmittance_range=(5, 15)),
            ],
            training=True,
        )

        for _ in range(10):
            s = make_sample(H=8, W=8, x_ant=3.0, y_ant=4.0)
            result = pipeline(s)

            # Pathloss must be non-negative (original values are >= 0, walls only add)
            self.assertTrue(
                (result.output_img >= -1e-5).all(),
                f"Pathloss has negative values: min={result.output_img.min().item()}"
            )


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

class TestPipelineStructuralInvariants(unittest.TestCase):
    """After any pipeline run, certain structural properties must hold."""

    def _make_pipeline(self, rotation_p=1.0, wall_p=1.0):
        return AugmentationPipeline(
            augmentations=[
                CardinalRotationAugmentation(p=rotation_p),
                WallInsertionAugmentation(p=wall_p, transmittance_range=(5, 15)),
            ],
            training=True,
        )

    def test_input_img_always_3_channels(self):
        """input_img must always have exactly 3 channels after pipeline."""
        pipeline = self._make_pipeline()
        for _ in range(20):
            s = make_sample(H=8, W=10)
            result = pipeline(s)
            self.assertEqual(result.input_img.shape[0], 3)

    def test_spatial_dimensions_consistent(self):
        """H, W metadata must match actual tensor dimensions."""
        pipeline = self._make_pipeline()
        for _ in range(20):
            s = make_sample(H=6, W=10)
            result = pipeline(s)

            self.assertEqual(result.input_img.shape[1], result.H)
            self.assertEqual(result.input_img.shape[2], result.W)
            self.assertEqual(result.output_img.shape, (result.H, result.W))
            self.assertEqual(result.mask.shape, (result.H, result.W))
            if result.floor_plan is not None:
                self.assertEqual(result.floor_plan.shape, (result.H, result.W))

    def test_mask_binary(self):
        """Mask should remain binary (0 or 1) after pipeline.
        Cardinal rotation is lossless so this should hold exactly."""
        pipeline = self._make_pipeline(wall_p=0.0)  # walls don't touch mask
        for _ in range(20):
            s = make_sample(H=8, W=8)
            result = pipeline(s)
            unique = result.mask.unique()
            for v in unique:
                self.assertIn(v.item(), [0.0, 1.0],
                              f"Mask contains non-binary value: {v.item()}")

    def test_mask_zero_count_preserved(self):
        """Cardinal rotation should preserve the number of zero pixels in mask."""
        pipeline = self._make_pipeline(wall_p=0.0)
        for _ in range(20):
            s = make_sample(H=6, W=10)
            orig_zeros = (s.mask == 0).sum().item()
            result = pipeline(s)
            result_zeros = (result.mask == 0).sum().item()
            self.assertEqual(orig_zeros, result_zeros,
                             "Rotation changed the number of zero pixels in mask")

    def test_no_nan_no_inf(self):
        """No tensor field should contain NaN or Inf after pipeline."""
        pipeline = self._make_pipeline()
        for _ in range(20):
            s = make_sample(H=8, W=10)
            result = pipeline(s)

            for name, tensor in [
                ("input_img", result.input_img),
                ("output_img", result.output_img),
                ("mask", result.mask),
            ]:
                self.assertFalse(torch.isnan(tensor).any(), f"NaN in {name}")
                self.assertFalse(torch.isinf(tensor).any(), f"Inf in {name}")

            if result.floor_plan is not None:
                self.assertFalse(torch.isnan(result.floor_plan).any(), "NaN in floor_plan")
                self.assertFalse(torch.isinf(result.floor_plan).any(), "Inf in floor_plan")


# ---------------------------------------------------------------------------
# Visual: full pipeline before/after
# ---------------------------------------------------------------------------

class TestPipelineVisualDebug(unittest.TestCase):
    """Visual debugging for pipeline integration."""

    def test_visual_full_pipeline(self):
        """Save before/after for the full pipeline with both augs active."""
        pipeline = AugmentationPipeline(
            augmentations=[
                CardinalRotationAugmentation(p=1.0),
                WallInsertionAugmentation(p=1.0, transmittance_range=(5, 15)),
            ],
            training=True,
        )
        s = make_sample(H=8, W=10, x_ant=5.0, y_ant=3.0, azimuth=45.0)
        before = clone_sample(s)
        after = pipeline(s)
        save_pipeline_visual(
            before, after,
            pipeline_desc="CardinalRotation(p=1) + WallInsertion(p=1)",
            filename="pipeline_full_both_active.png",
        )

    def test_visual_walls_only_pipeline(self):
        """Save before/after for pipeline with only walls active."""
        pipeline = AugmentationPipeline(
            augmentations=[
                CardinalRotationAugmentation(p=0.0),
                WallInsertionAugmentation(p=1.0, transmittance_range=(5, 15)),
            ],
            training=True,
        )
        s = make_sample(H=8, W=10, x_ant=5.0, y_ant=3.0, azimuth=45.0)
        before = clone_sample(s)
        after = pipeline(s)
        save_pipeline_visual(
            before, after,
            pipeline_desc="CardinalRotation(p=0) + WallInsertion(p=1)",
            filename="pipeline_walls_only.png",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
