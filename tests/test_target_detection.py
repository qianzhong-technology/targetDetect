import unittest

import cv2
import numpy as np

from common.target_lock import TargetLockHysteresis
from vision.target_center import detect_target_center
from vision.target_frame import detect_target_corners, order_points


class TargetGeometryTests(unittest.TestCase):
    def test_order_points_is_clockwise_from_top_left(self):
        points = np.asarray(
            [
                [470.0, 395.0],
                [170.0, 395.0],
                [470.0, 70.0],
                [170.0, 70.0],
            ],
            dtype=np.float32
        )
        ordered = order_points(points)
        np.testing.assert_allclose(
            ordered,
            np.asarray(
                [
                    [170.0, 70.0],
                    [470.0, 70.0],
                    [470.0, 395.0],
                    [170.0, 395.0],
                ],
                dtype=np.float32
            )
        )

    def test_outer_frame_detected_with_complex_background(self):
        rng = np.random.default_rng(4)
        frame = cv2.addWeighted(
            np.full((480, 640, 3), 220, dtype=np.uint8),
            0.35,
            cv2.GaussianBlur(
                rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
                (9, 9),
                0
            ),
            0.65,
            0
        )
        cv2.rectangle(frame, (30, 40), (145, 170), (20, 20, 20), -1)
        cv2.rectangle(frame, (170, 70), (470, 395), (0, 0, 0), 8)
        for radius in (20, 40, 60, 80, 100):
            cv2.circle(frame, (320, 232), radius, (0, 0, 255), 3)

        corners, _ = detect_target_corners(frame)
        self.assertIsNotNone(corners)
        center = np.mean(corners, axis=0)
        np.testing.assert_allclose(center, np.asarray([320.0, 232.0]), atol=10.0)
        area = abs(cv2.contourArea(corners.astype(np.float32).reshape(-1, 1, 2)))
        self.assertGreater(area, 70000)

    def test_ring_center_fit_is_high_confidence(self):
        rectified = np.full((297, 210, 3), 255, dtype=np.uint8)
        for radius in (20, 40, 60, 80, 100):
            cv2.circle(rectified, (105, 148), radius, (0, 0, 255), 2)

        center, rms, _, _, fit_status, _ = detect_target_center(rectified)
        self.assertIsNotNone(center)
        self.assertLess(rms, 5.0)
        self.assertEqual(fit_status, "OK")

    def test_random_red_shapes_do_not_create_a_target_lock(self):
        rng = np.random.default_rng(12)
        rectified = np.full((297, 210, 3), 255, dtype=np.uint8)
        for _ in range(60):
            x = int(rng.integers(15, 195))
            y = int(rng.integers(15, 282))
            radius = int(rng.integers(4, 15))
            cv2.circle(
                rectified,
                (x, y),
                radius,
                (0, 0, 255),
                -1
            )

        center, rms, _, _, fit_status, _ = detect_target_center(rectified)
        self.assertTrue(
            center is None
            or fit_status != "OK"
            or rms is None
            or rms > 5.0
        )

    def test_subpixel_frame_refinement_improves_corner_location(self):
        frame = np.full((480, 640, 3), 245, dtype=np.uint8)
        cv2.rectangle(frame, (170, 70), (470, 395), (0, 0, 0), 8)
        corners, _ = detect_target_corners(frame)
        self.assertIsNotNone(corners)
        expected = np.asarray(
            [
                [170.0, 70.0],
                [470.0, 70.0],
                [470.0, 395.0],
                [170.0, 395.0],
            ],
            dtype=np.float32
        )
        np.testing.assert_allclose(corners, expected, atol=8.0)


class TargetLockTests(unittest.TestCase):
    def test_large_jump_is_not_immediately_accepted_as_a_new_lock(self):
        lock = TargetLockHysteresis(
            lock_confirm_frames=2,
            loss_confirm_frames=3,
            max_point_jump_px=40.0,
            point_smooth_alpha=0.75
        )
        self.assertFalse(lock.update(True, (320.0, 240.0))[0])
        self.assertTrue(lock.update(True, (320.0, 240.0))[0])
        self.assertTrue(lock.update(True, (450.0, 100.0))[0])
        self.assertEqual(lock.update(False, None)[0], True)
        self.assertFalse(lock.update(False, None)[0])


if __name__ == "__main__":
    unittest.main()
