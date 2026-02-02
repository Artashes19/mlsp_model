"""
Visual debugging tests that save before/after images for each augmentation.
Images are saved to tests/augmentation_visuals/.

These tests always pass (they only produce visual output).
Run them to visually inspect augmentation behavior.
"""
import unittest
from unittest.mock import patch

from src.utils.indoor.augmentations import (
    AugmentationPipeline,
    CardinalRotationAugmentation,
    WallInsertionAugmentation,
)
from src.utils.indoor.types import RadarSample

from .conftest import (
    make_sample,
    clone_sample,
    save_rotation_visual,
    save_cycle_visual,
    save_pipeline_visual,
)


class TestVisualDebug(unittest.TestCase):
    """
    Visual debugging tests that save before/after images for each augmentation.
    Images are saved to tests/augmentation_visuals/.
    """

    def _apply_rotation_k(self, k: int, sample: RadarSample) -> RadarSample:
        aug = CardinalRotationAugmentation(p=1.0)
        with patch("src.utils.indoor.augmentations.random.choice", return_value=k):
            return aug(sample)

    # ------------------------------------------------------------------
    # One comprehensive image per rotation angle
    # ------------------------------------------------------------------

    def test_visual_rotation_90(self):
        """Save before/after visual for 90-degree CCW rotation."""
        s = make_sample(H=6, W=10, x_ant=7.0, y_ant=2.0, azimuth=45.0)
        before = clone_sample(s)
        after = self._apply_rotation_k(1, s)
        save_rotation_visual(before, after, k=1, filename="rotation_k1_90ccw.png")

    def test_visual_rotation_180(self):
        """Save before/after visual for 180-degree rotation."""
        s = make_sample(H=6, W=10, x_ant=7.0, y_ant=2.0, azimuth=45.0)
        before = clone_sample(s)
        after = self._apply_rotation_k(2, s)
        save_rotation_visual(before, after, k=2, filename="rotation_k2_180.png")

    def test_visual_rotation_270(self):
        """Save before/after visual for 270-degree CCW (= 90 CW) rotation."""
        s = make_sample(H=6, W=10, x_ant=7.0, y_ant=2.0, azimuth=45.0)
        before = clone_sample(s)
        after = self._apply_rotation_k(3, s)
        save_rotation_visual(before, after, k=3, filename="rotation_k3_270ccw.png")

    # ------------------------------------------------------------------
    # Antenna tracking -- marker planted at antenna position
    # ------------------------------------------------------------------

    def test_visual_antenna_tracking(self):
        """
        Plant a bright 999-marker at the antenna pixel, rotate, and
        visually verify the marker moved to the new antenna position.
        """
        H, W = 8, 12
        x_ant, y_ant = 9, 3
        MARKER = 999.0

        for k in [1, 2, 3]:
            with self.subTest(k=k):
                s = make_sample(H=H, W=W, x_ant=float(x_ant), y_ant=float(y_ant))
                s.input_img[:, y_ant, x_ant] = MARKER
                s.output_img[y_ant, x_ant] = MARKER
                before = clone_sample(s)
                after = self._apply_rotation_k(k, s)
                save_rotation_visual(
                    before, after, k=k,
                    filename=f"antenna_tracking_k{k}.png",
                )

    # ------------------------------------------------------------------
    # Rotation cycle: 4 x same rotation = identity
    # ------------------------------------------------------------------

    def test_visual_rotation_cycle(self):
        """
        Show the full 4-rotation cycle for each k, verifying visually
        that the last state matches the original.
        """
        for k in [1, 2, 3]:
            with self.subTest(k=k):
                s = make_sample(H=6, W=10, x_ant=4.0, y_ant=2.0, azimuth=30.0)
                states = [clone_sample(s)]
                for _ in range(4):
                    s = self._apply_rotation_k(k, s)
                    states.append(clone_sample(s))
                save_cycle_visual(states, k=k, filename=f"cycle_k{k}.png")

    # ------------------------------------------------------------------
    # Pipeline visual -- rotation through the pipeline
    # ------------------------------------------------------------------

    def test_visual_pipeline(self):
        """
        Show before/after of a sample going through AugmentationPipeline
        with a CardinalRotationAugmentation (random k).
        """
        pipeline = AugmentationPipeline(
            augmentations=[CardinalRotationAugmentation(p=1.0)],
            training=True,
        )
        s = make_sample(H=6, W=10, x_ant=7.0, y_ant=2.0, azimuth=45.0)
        before = clone_sample(s)
        after = pipeline(s)
        save_pipeline_visual(
            before, after,
            pipeline_desc="CardinalRotation(p=1.0)",
            filename="pipeline_cardinal_rotation.png",
        )

    # ------------------------------------------------------------------
    # Wall insertion visuals
    # ------------------------------------------------------------------

    def test_visual_wall_insertion(self):
        """Save before/after visual for wall insertion."""
        aug = WallInsertionAugmentation(
            p=1.0, transmittance_range=(5, 20), max_wall_density=0.3
        )
        s = make_sample(H=10, W=12, x_ant=5.0, y_ant=4.0, azimuth=0.0)
        before = clone_sample(s)
        after = aug(s)

        save_pipeline_visual(
            before,
            after,
            pipeline_desc="WallInsertion(p=1.0, transmittance_range=(5,20))",
            filename="wall_insertion_visual.png",
        )

    def test_visual_wall_insertion_multiple(self):
        """Save multiple examples of wall insertion."""
        aug = WallInsertionAugmentation(
            p=1.0, transmittance_range=(10, 25), max_wall_density=0.4
        )

        for i in range(3):
            s = make_sample(H=8, W=10, x_ant=4.0, y_ant=3.0)
            before = clone_sample(s)
            after = aug(s)

            save_pipeline_visual(
                before,
                after,
                pipeline_desc=f"WallInsertion example {i+1}",
                filename=f"wall_insertion_example_{i+1}.png",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
