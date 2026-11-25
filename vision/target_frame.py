"""靶纸框检测、角点跟踪与透视几何。"""

import math
from typing import Optional

import cv2
import numpy as np

from common.config import (
    TARGET_ACCEL_USE_CPP_LK,
    TARGET_ACCEL_USE_CPP_PERSPECTIVE,
    CORNER_TRACK_MAX_FEATURES,
    CORNER_TRACK_MAX_FRAMES,
    CORNER_TRACK_MIN_FEATURES,
    CORNER_DETECT_INTERVAL,
    CORNER_TRACK_REFRESH_FEATURES,
    CORNER_TRACK_HOMOGRAPHY_INTERVAL,
    CORNER_TRACK_AFFINE_MAX_MEDIAN_RESIDUAL,
    CORNER_TRACK_AFFINE_MAX_RESIDUAL,
    CORNER_TRACK_MAX_OFFSCREEN_FRAMES,
    CORNER_LOST_DETECT_INTERVAL,
    CORNER_PREDICT_DETECT_INTERVAL,
    CORNER_TRACK_PREDICTION_VELOCITY_DECAY,
    CORNER_TRACK_CENTER_ANCHOR_MAX_CORRECTION_PX,
    CORNER_TRACK_MOTION_THRESHOLD_PX,
    CORNER_TRACK_PREDICTION_TOLERANCE_PX,
    CORNER_TRACK_SMOOTH_ALPHA,
    CORNER_TRACK_VELOCITY_ALPHA,
    CORNER_TRACK_MAX_SHIFT,
    CORNER_TRACK_AREA_MIN_RATIO,
    CORNER_TRACK_AREA_MAX_RATIO,
    CORNER_TRACK_REJECTED_FRAMES,
    CORNER_TRACK_SCALE,
    FRAME_CLOSE_KERNEL,
    FRAME_ASPECT_TOLERANCE,
    FRAME_ASPECT_WIDE_TOLERANCE,
    FRAME_DETECT_SCALE,
    FRAME_MIN_AREA_RATIO,
    FRAME_HSV_HIGH,
    FRAME_HSV_LOW,
    RECT_HEIGHT,
    RECT_WIDTH,
    TRACKING_ROI_MARGIN,
    TRACKING_ROI_MIN_HEIGHT,
    TRACKING_ROI_MIN_WIDTH,
)
from vision.native_backend import get_native_backend

_NATIVE_BACKEND = get_native_backend()

LK_FLOW_OPTIONS = {
    "winSize": (11, 11),
    "maxLevel": 2,
    "criteria": (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        15,
        0.03
    )
}


def order_points(pts):
    """
    将四个角点排序为：
    左上、右上、右下、左下

    返回 shape=(4,2) 的 float32 数组。
    """

    pts = np.array(pts, dtype=np.float32)

    center = np.mean(pts, axis=0)

    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0]
    )

    # 按绕中心的角度排序
    ordered = pts[np.argsort(angles)]

    # 找到最接近左上角的点作为起点
    start_index = np.argmin(
        ordered[:, 0] + ordered[:, 1]
    )

    ordered = np.roll(
        ordered,
        -start_index,
        axis=0
    )

    return ordered.astype(np.float32)


def _matches_target_aspect(corners):
    edges = np.linalg.norm(
        np.roll(corners, -1, axis=0) - corners,
        axis=1
    )
    average_width = (edges[0] + edges[2]) / 2.0
    average_height = (edges[1] + edges[3]) / 2.0
    if average_width <= 0 or average_height <= 0:
        return False

    measured_aspect = max(average_width, average_height) / min(
        average_width,
        average_height
    )
    expected_aspect = max(
        RECT_WIDTH / RECT_HEIGHT,
        RECT_HEIGHT / RECT_WIDTH
    )
    aspect_error = measured_aspect / expected_aspect - 1.0
    return (
        -FRAME_ASPECT_TOLERANCE
        <= aspect_error
        <= FRAME_ASPECT_WIDE_TOLERANCE
    )


def detect_target_corners(frame, reference_corners=None):
    """
    从原始相机图像中寻找黑色矩形边框。

    返回：
        corners: 左上、右上、右下、左下
        black_mask: 黑色区域二值图

    如果未找到可靠的四边形，返回 None。
    """

    analysis_size = (
        max(1, int(frame.shape[1] * FRAME_DETECT_SCALE)),
        max(1, int(frame.shape[0] * FRAME_DETECT_SCALE))
    )
    analysis_frame = cv2.resize(
        frame,
        analysis_size,
        interpolation=cv2.INTER_AREA
    )

    hsv = cv2.cvtColor(
        analysis_frame,
        cv2.COLOR_BGR2HSV
    )
    hsv_mask = cv2.inRange(
        hsv,
        FRAME_HSV_LOW,
        FRAME_HSV_HIGH
    )

    image_area = analysis_frame.shape[0] * analysis_frame.shape[1]
    image_height, image_width = analysis_frame.shape[:2]

    def find_candidates(mask):
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            FRAME_CLOSE_KERNEL,
            iterations=2
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        # 独立靶纸优先选择外框；靶纸与背景相连时使用内部框候选。
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)

            if (
                x <= 1
                or y <= 1
                or x + width >= image_width - 1
                or y + height >= image_height - 1
            ):
                continue

            if cv2.contourArea(contour) < image_area * FRAME_MIN_AREA_RATIO:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            for epsilon_ratio in [0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(
                    contour,
                    epsilon_ratio * perimeter,
                    True
                )

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quad = order_points(approx.reshape(4, 2))
                    quad_area = abs(
                        cv2.contourArea(
                            quad.astype(np.float32).reshape(-1, 1, 2)
                        )
                    )

                    if (
                        quad_area >= image_area * FRAME_MIN_AREA_RATIO
                        and _matches_target_aspect(quad)
                    ):
                        candidates.append((quad_area, quad))
                    break

        return candidates, mask

    candidates, black_mask = find_candidates(hsv_mask)

    if not candidates:
        gray = cv2.cvtColor(
            analysis_frame,
            cv2.COLOR_BGR2GRAY
        )
        otsu_threshold, _ = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        _, otsu_mask = cv2.threshold(
            gray,
            otsu_threshold,
            255,
            cv2.THRESH_BINARY_INV
        )
        candidates, black_mask = find_candidates(otsu_mask)

    if not candidates:
        return None, black_mask

    # 优先选择与已知靶纸框几何最一致的候选。
    use_reference_selection = False
    if reference_corners is not None:
        reference = np.asarray(
            reference_corners,
            dtype=np.float32
        )
        reference_area = abs(cv2.contourArea(
            reference.reshape(-1, 1, 2)
        ))
        if reference_area > 0:
            stable_candidates = []
            for _, quad in candidates:
                quad_full = quad.copy()
                quad_full[:, 0] *= frame.shape[1] / analysis_size[0]
                quad_full[:, 1] *= frame.shape[0] / analysis_size[1]
                candidate_area_full = abs(cv2.contourArea(
                    quad_full.reshape(-1, 1, 2)
                ))
                area_ratio = candidate_area_full / reference_area
                center_shift = np.linalg.norm(
                    np.mean(quad_full, axis=0)
                    - np.mean(reference, axis=0)
                )
                if (
                    0.65 <= area_ratio <= 1.55
                    and center_shift < max(
                        100.0,
                        math.sqrt(reference_area) * 0.45
                    )
                ):
                    corner_shift = float(np.max(np.linalg.norm(
                        quad_full - reference,
                        axis=1
                    )))
                    combined_score = (
                        center_shift
                        + corner_shift * 0.25
                        + abs(area_ratio - 1.0) * math.sqrt(reference_area)
                    )
                    stable_candidates.append((combined_score, quad))
            if stable_candidates:
                candidates = stable_candidates
                use_reference_selection = True

    if use_reference_selection:
        candidates.sort(key=lambda item: item[0])
    else:
        # 选择面积最大的候选矩形
        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

    corners = candidates[0][1].copy()
    corners[:, 0] *= frame.shape[1] / analysis_size[0]
    corners[:, 1] *= frame.shape[0] / analysis_size[1]

    return corners, black_mask

class TargetCornerTracker:
    def __init__(self):
        self.previous_gray: Optional[np.ndarray] = None
        self.previous_points: Optional[np.ndarray] = None
        self.corners: Optional[np.ndarray] = None
        self.predicted_corners: Optional[np.ndarray] = None
        self.corner_velocity: Optional[np.ndarray] = None
        self.predicted_frames = 0
        self.offscreen_predicted_frames = 0
        self.track_width: Optional[int] = None
        self.track_height: Optional[int] = None
        self.pending_corners: Optional[np.ndarray] = None
        self.pending_hits = 0
        self.frames_since_homography = 0
        self.clear()

    @property
    def is_active(self):
        if self.is_predicted:
            return (
                self.previous_gray is not None
                and self.predicted_corners is not None
                and self.corner_velocity is not None
            )
        return (
            self.previous_gray is not None
            and self.previous_points is not None
            and len(self.previous_points) >= CORNER_TRACK_MIN_FEATURES
        )

    @property
    def is_predicted(self):
        return self.predicted_frames > 0

    def needs_redetection(self, frame_index):
        if self.is_predicted:
            return (
                self.predicted_frames == 1
                or self.predicted_frames % CORNER_PREDICT_DETECT_INTERVAL == 0
            )
        if not self.is_active:
            return frame_index % CORNER_LOST_DETECT_INTERVAL == 0
        return self.frames_since_detection >= CORNER_DETECT_INTERVAL

    def clear(self):
        self.previous_gray = None
        self.previous_points = None
        self.corners = None
        self.predicted_corners = None
        self.corner_velocity = None
        self.predicted_frames = 0
        self.offscreen_predicted_frames = 0
        self.track_width = None
        self.track_height = None
        self.pending_corners = None
        self.pending_hits = 0
        self.frames_since_homography = 0
        self.frames_since_detection = 0
        self.failures = 0
        self.frames_since_homography = 0

    @staticmethod
    def _find_features(gray, corners):
        mask = np.zeros(gray.shape, dtype=np.uint8)
        polygon = np.round(corners).astype(np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=CORNER_TRACK_MAX_FEATURES,
            qualityLevel=0.01,
            minDistance=8,
            mask=mask,
            blockSize=5
        )
        return points

    @staticmethod
    def _prepare_gray(frame):
        width = max(1, int(round(frame.shape[1] * CORNER_TRACK_SCALE)))
        height = max(1, int(round(frame.shape[0] * CORNER_TRACK_SCALE)))
        small_frame = cv2.resize(
            frame,
            (width, height),
            interpolation=cv2.INTER_AREA
        )
        if small_frame.ndim == 3:
            return cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        return small_frame

    def _corners_intersect_frame(self, corners):
        if self.track_width is None or self.track_height is None:
            return True
        min_x = float(np.min(corners[:, 0]))
        max_x = float(np.max(corners[:, 0]))
        min_y = float(np.min(corners[:, 1]))
        max_y = float(np.max(corners[:, 1]))
        return not (
            max_x < 0.0
            or min_x > self.track_width - 1
            or max_y < 0.0
            or min_y > self.track_height - 1
        )

    def initialize(self, frame, corners):
        gray = self._prepare_gray(frame)
        candidate_corners = (
            corners.astype(np.float32) * CORNER_TRACK_SCALE
        )
        measured_velocity = None
        direct_motion = False

        if self.corners is not None:
            old_area = abs(cv2.contourArea(
                self.corners.reshape(-1, 1, 2)
            ))
            candidate_area = abs(cv2.contourArea(
                candidate_corners.reshape(-1, 1, 2)
            ))
            area_ratio = (
                candidate_area / old_area if old_area > 0 else 1.0
            )
            max_shift = CORNER_TRACK_MAX_SHIFT
            corner_shift = np.max(np.linalg.norm(
                candidate_corners - self.corners,
                axis=1
            ))
            velocity_prediction_error = None
            if self.corner_velocity is not None:
                predicted_corners = (
                    self.corners + self.corner_velocity
                )
                velocity_prediction_error = np.max(np.linalg.norm(
                    candidate_corners - predicted_corners,
                    axis=1
                ))

            motion_consistent = (
                velocity_prediction_error is not None
                and velocity_prediction_error
                <= max_shift * CORNER_TRACK_SCALE
            )
            geometry_consistent = (
                CORNER_TRACK_AREA_MIN_RATIO
                <= area_ratio
                <= CORNER_TRACK_AREA_MAX_RATIO
            )
            if (
                not geometry_consistent
                or (
                    corner_shift > max_shift * CORNER_TRACK_SCALE
                    and not motion_consistent
                )
            ):
                if self.pending_corners is not None:
                    pending_shift = np.max(np.linalg.norm(
                        candidate_corners - self.pending_corners,
                        axis=1
                    ))
                    if pending_shift <= max_shift * CORNER_TRACK_SCALE:
                        self.pending_hits += 1
                    else:
                        self.pending_corners = candidate_corners.copy()
                        self.pending_hits = 1
                else:
                    self.pending_corners = candidate_corners.copy()
                    self.pending_hits = 1

                if self.pending_hits >= CORNER_TRACK_REJECTED_FRAMES:
                    if self.pending_corners is not None:
                        candidate_corners = self.pending_corners.copy()
                    self.pending_corners = None
                    self.pending_hits = 0
                else:
                    return None
            else:
                self.pending_corners = None
                self.pending_hits = 0

            measured_velocity = (
                candidate_corners - self.corners
            ).astype(np.float32)
            direct_motion = (
                np.max(np.linalg.norm(measured_velocity, axis=1))
                / CORNER_TRACK_SCALE
                > CORNER_TRACK_MOTION_THRESHOLD_PX
                and motion_consistent
            )
            if not direct_motion:
                candidate_corners = (
                    CORNER_TRACK_SMOOTH_ALPHA * candidate_corners
                    + (1.0 - CORNER_TRACK_SMOOTH_ALPHA) * self.corners
                ).astype(np.float32)
            else:
                candidate_corners = candidate_corners.astype(np.float32)

        next_points = self._find_features(
            gray,
            candidate_corners
        )
        if next_points is None or len(next_points) < CORNER_TRACK_MIN_FEATURES:
            return None

        self.corners = candidate_corners
        self.predicted_corners = self.corners.copy()
        if measured_velocity is not None:
            self.corner_velocity = measured_velocity
        else:
            self.corner_velocity = np.zeros((4, 2), dtype=np.float32)
        self.predicted_frames = 0
        self.offscreen_predicted_frames = 0
        self.track_width = gray.shape[1]
        self.track_height = gray.shape[0]
        self.previous_gray = gray
        self.previous_points = next_points
        self.frames_since_detection = 0
        self.failures = 0
        return self.corners.copy() / CORNER_TRACK_SCALE

    def _tracking_failed(self, gray=None):
        self.failures += 1
        if self.predicted_corners is None:
            return None
        if self.corner_velocity is not None:
            self.corner_velocity = (
                self.corner_velocity
                * CORNER_TRACK_PREDICTION_VELOCITY_DECAY
            ).astype(np.float32)
            self.predicted_corners = (
                self.predicted_corners + self.corner_velocity
            ).astype(np.float32)
        self.predicted_frames += 1
        if self._corners_intersect_frame(self.predicted_corners):
            self.offscreen_predicted_frames = 0
        else:
            self.offscreen_predicted_frames += 1
        if (
            self.offscreen_predicted_frames
            > CORNER_TRACK_MAX_OFFSCREEN_FRAMES
        ):
            self.clear()
            return None
        self.corners = self.predicted_corners.copy()
        if gray is not None:
            refreshed_points = self._find_features(
                gray,
                self.corners
            )
            if (
                refreshed_points is not None
                and len(refreshed_points) >= 4
            ):
                self.previous_gray = gray
                self.previous_points = refreshed_points.astype(
                    np.float32,
                    copy=False
                )
        return self.predicted_corners.copy() / CORNER_TRACK_SCALE

    def _update_prediction_from_visible_features(self, gray):
        if (
            self.previous_gray is None
            or self.previous_points is None
            or self.corners is None
            or len(self.previous_points) < 4
        ):
            return False

        next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.previous_points,
            np.empty_like(self.previous_points),
            **LK_FLOW_OPTIONS
        )
        if next_points is None or forward_status is None:
            return False

        current = next_points.reshape(-1, 2)
        previous = self.previous_points.reshape(-1, 2)
        valid = (
            (forward_status.reshape(-1) != 0)
            & np.isfinite(current).all(axis=1)
            & (current[:, 0] >= 0.0)
            & (current[:, 0] < gray.shape[1])
            & (current[:, 1] >= 0.0)
            & (current[:, 1] < gray.shape[0])
        )
        previous_valid = previous[valid]
        current_valid = current[valid]
        if len(current_valid) < 4:
            return False

        displacement = current_valid - previous_valid
        median_delta = np.median(displacement, axis=0)
        residual = np.linalg.norm(
            displacement - median_delta,
            axis=1
        )
        median_residual = float(np.median(residual))
        mad_residual = float(np.median(
            np.abs(residual - median_residual)
        ))
        inliers = residual <= max(
            0.75,
            median_residual + 3.0 * 1.4826 * mad_residual
        )
        if np.count_nonzero(inliers) < 4:
            return False

        delta = np.median(
            displacement[inliers],
            axis=0
        ).astype(np.float32)
        predicted_corners = self.corners + delta
        if not np.isfinite(predicted_corners).all():
            return False
        if not self._corners_intersect_frame(predicted_corners):
            return False

        visible_inliers = current_valid[inliers]
        inside_count = sum(
            cv2.pointPolygonTest(
                predicted_corners,
                (float(point[0]), float(point[1])),
                False
            ) >= 0.0
            for point in visible_inliers
        )
        if inside_count < max(2, len(visible_inliers) // 4):
            return False

        if self.corner_velocity is None:
            velocity = delta.copy()
        else:
            velocity = (
                CORNER_TRACK_VELOCITY_ALPHA * delta
                + (1.0 - CORNER_TRACK_VELOCITY_ALPHA)
                * self.corner_velocity
            ).astype(np.float32)

        corrected_corners = predicted_corners.astype(np.float32)
        self.corners = corrected_corners
        self.predicted_corners = corrected_corners.copy()
        self.corner_velocity = velocity
        self.predicted_frames += 1
        self.offscreen_predicted_frames = (
            0
            if self._corners_intersect_frame(self.predicted_corners)
            else self.offscreen_predicted_frames + 1
        )
        self.previous_gray = gray

        visible_points = visible_inliers.reshape(-1, 1, 2)
        if len(visible_points) < CORNER_TRACK_REFRESH_FEATURES:
            refreshed_points = self._find_features(
                gray,
                self.corners
            )
            if (
                refreshed_points is not None
                and len(refreshed_points) >= 4
            ):
                visible_points = refreshed_points
        self.previous_points = visible_points.astype(
            np.float32,
            copy=False
        )
        return True

    def _apply_prediction_center_anchor(self, center_anchor):
        if (
            center_anchor is None
            or self.predicted_corners is None
        ):
            return False

        full_corners = (
            self.predicted_corners.astype(np.float32)
            / CORNER_TRACK_SCALE
        )
        if not cv2.isContourConvex(
            full_corners.reshape(-1, 1, 2)
        ):
            return False

        H = make_homography(full_corners)
        current_center = target_frame_center_to_camera(H)
        if current_center is None:
            return False
        correction = (
            np.asarray(center_anchor, dtype=np.float32)
            - np.asarray(current_center, dtype=np.float32)
        )
        correction_size = float(np.linalg.norm(correction))
        if (
            correction_size < 0.5
            or correction_size
            > CORNER_TRACK_CENTER_ANCHOR_MAX_CORRECTION_PX
        ):
            return False

        correction_scaled = (
            correction * CORNER_TRACK_SCALE
        ).astype(np.float32)
        corrected_corners = (
            self.predicted_corners + correction_scaled
        )
        if not self._corners_intersect_frame(corrected_corners):
            return False

        self.corners = corrected_corners
        self.predicted_corners = corrected_corners.copy()
        self.offscreen_predicted_frames = 0
        if self.previous_points is not None:
            self.previous_points = (
                self.previous_points.astype(np.float32)
                + correction_scaled
            )
        if (
            self.corner_velocity is not None
            and correction_size > 12.0
        ):
            self.corner_velocity = (
                self.corner_velocity * 0.7
            ).astype(np.float32)
        return True

    def update(self, frame, center_anchor=None):
        gray = self._prepare_gray(frame)
        if self.is_predicted:
            if self._update_prediction_from_visible_features(gray):
                self._apply_prediction_center_anchor(center_anchor)
                current_corners = self.corners
                if current_corners is None:
                    return None
                return current_corners.copy() / CORNER_TRACK_SCALE
            predicted = self._tracking_failed(gray)
            if predicted is not None:
                self._apply_prediction_center_anchor(center_anchor)
                current_corners = self.corners
                if current_corners is not None:
                    predicted = (
                        current_corners.copy() / CORNER_TRACK_SCALE
                    )
            return predicted

        previous_gray = self.previous_gray
        previous_points = self.previous_points
        corners = self.corners
        if (
            previous_gray is None
            or previous_points is None
            or corners is None
            or len(previous_points) < CORNER_TRACK_MIN_FEATURES
        ):
            return None

        if self.frames_since_detection >= CORNER_TRACK_MAX_FRAMES:
            return self._tracking_failed(gray)

        current_corners = None
        tracked_points = None
        use_native_lk = (
            TARGET_ACCEL_USE_CPP_LK
            and _NATIVE_BACKEND.is_available
        )

        if use_native_lk:
            native_result = _NATIVE_BACKEND.lk_track_corners(
                previous_gray=previous_gray,
                gray=gray,
                previous_points=previous_points,
                corners=corners,
                min_features=CORNER_TRACK_MIN_FEATURES,
                min_inlier_ratio=0.75,
                support_min_ratio=0.01,
                area_min_ratio=CORNER_TRACK_AREA_MIN_RATIO,
                area_max_ratio=CORNER_TRACK_AREA_MAX_RATIO
            )
            if native_result is None:
                return self._tracking_failed(gray)

            current_corners = native_result["corners"]
            tracked_points = native_result["tracked_points"]
            max_corner_shift = native_result["max_shift"]

            diagonal = max(
                np.linalg.norm(corners[0] - corners[2]),
                np.linalg.norm(corners[1] - corners[3])
            )
            if max_corner_shift > max(
                CORNER_TRACK_MAX_SHIFT * CORNER_TRACK_SCALE,
                diagonal * 0.35
            ):
                return self._tracking_failed(gray)
        else:
            next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                gray,
                previous_points,
                np.empty_like(previous_points),
                **LK_FLOW_OPTIONS
            )

            if next_points is None or forward_status is None:
                return self._tracking_failed(gray)

            current = next_points.reshape(-1, 2)
            forward_ok = forward_status.reshape(-1) != 0
            valid = (
                forward_ok
                & np.isfinite(current).all(axis=1)
            )

            previous_valid = previous_points.reshape(-1, 2)[valid]
            current_valid = current[valid]
            if len(previous_valid) < CORNER_TRACK_MIN_FEATURES:
                return self._tracking_failed(gray)

            use_full_homography = (
                self.frames_since_homography
                >= CORNER_TRACK_HOMOGRAPHY_INTERVAL
                or self.frames_since_homography == 0
            )
            affine_accepted = False
            if not use_full_homography:
                affine, affine_mask = cv2.estimateAffinePartial2D(
                    previous_valid,
                    current_valid,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=1.5
                )
                if affine is not None and affine_mask is not None:
                    affine_inliers = affine_mask.reshape(-1) != 0
                    if np.count_nonzero(affine_inliers) >= (
                        CORNER_TRACK_MIN_FEATURES
                    ):
                        affine_points = cv2.transform(
                            previous_valid[affine_inliers].reshape(
                                -1,
                                1,
                                2
                            ),
                            affine
                        ).reshape(-1, 2)
                        residual = np.linalg.norm(
                            affine_points
                            - current_valid[affine_inliers],
                            axis=1
                        )
                        affine_accepted = (
                            np.median(residual)
                            <= CORNER_TRACK_AFFINE_MAX_MEDIAN_RESIDUAL
                            and np.max(residual)
                            <= CORNER_TRACK_AFFINE_MAX_RESIDUAL
                            and np.count_nonzero(affine_inliers)
                            / len(previous_valid)
                            >= 0.75
                        )
                        if affine_accepted:
                            homography = np.eye(
                                3,
                                dtype=np.float32
                            )
                            homography[:2] = affine
                            inlier_mask = affine_mask
            if use_full_homography or not affine_accepted:
                homography, inlier_mask = cv2.findHomography(
                    previous_valid,
                    current_valid,
                    cv2.RANSAC,
                    3.0
                )
                if homography is not None:
                    self.frames_since_homography = 0
            if homography is None or inlier_mask is None:
                return self._tracking_failed(gray)

            inliers = inlier_mask.reshape(-1) != 0
            inlier_count = int(np.count_nonzero(inliers))
            if (
                inlier_count < CORNER_TRACK_MIN_FEATURES
                or inlier_count / len(previous_valid) < 0.75
            ):
                return self._tracking_failed(gray)

            previous_corners = corners.reshape(-1, 1, 2)
            previous_area = abs(cv2.contourArea(previous_corners))
            inlier_hull = cv2.convexHull(
                previous_valid[inliers].astype(np.float32).reshape(-1, 1, 2)
            )
            support_ratio = (
                cv2.contourArea(inlier_hull) / previous_area
                if previous_area > 0
                else 0.0
            )
            if support_ratio < 0.01:
                return self._tracking_failed(gray)

            current_corners = cv2.perspectiveTransform(
                previous_corners,
                homography
            ).reshape(-1, 2)
            if (
                not np.isfinite(current_corners).all()
                or not cv2.isContourConvex(
                    current_corners.astype(np.float32).reshape(-1, 1, 2)
                )
            ):
                return self._tracking_failed(gray)

            current_area = abs(cv2.contourArea(
                current_corners.astype(np.float32).reshape(-1, 1, 2)
            ))
            area_ratio = current_area / previous_area if previous_area > 0 else 0.0
            if not (
                CORNER_TRACK_AREA_MIN_RATIO
                <= area_ratio
                <= CORNER_TRACK_AREA_MAX_RATIO
            ):
                return self._tracking_failed(gray)

            diagonal = max(
                np.linalg.norm(corners[0] - corners[2]),
                np.linalg.norm(corners[1] - corners[3])
            )
            corner_shift = np.linalg.norm(
                current_corners - corners,
                axis=1
            )
            if np.max(corner_shift) > max(
                CORNER_TRACK_MAX_SHIFT * CORNER_TRACK_SCALE,
                diagonal * 0.35
            ):
                return self._tracking_failed(gray)

            tracked_points = current_valid[inliers].reshape(-1, 1, 2)

        current_delta = current_corners - corners
        near_frame_edge = (
            (
                np.min(corners[:, 0]) <= 0.0
                or np.max(corners[:, 0]) >= gray.shape[1] - 1
                or np.min(corners[:, 1]) <= 0.0
                or np.max(corners[:, 1]) >= gray.shape[0] - 1
            )
            or (
                np.min(current_corners[:, 0]) <= 0.0
                or np.max(current_corners[:, 0]) >= gray.shape[1] - 1
                or np.min(current_corners[:, 1]) <= 0.0
                or np.max(current_corners[:, 1]) >= gray.shape[0] - 1
            )
        )
        if (
            near_frame_edge
            and self.corner_velocity is not None
            and (
                np.max(np.linalg.norm(
                    self.corner_velocity,
                    axis=1
                ))
                / CORNER_TRACK_SCALE
                >= CORNER_TRACK_MOTION_THRESHOLD_PX
            )
        ):
            predicted_corners = corners + self.corner_velocity
            prediction_error = (
                np.max(np.linalg.norm(
                    current_corners - predicted_corners,
                    axis=1
                ))
                / CORNER_TRACK_SCALE
            )
            if prediction_error > CORNER_TRACK_PREDICTION_TOLERANCE_PX:
                return self._tracking_failed(gray)

        if self.corner_velocity is None:
            self.corner_velocity = current_delta
        else:
            self.corner_velocity = (
                CORNER_TRACK_VELOCITY_ALPHA * current_delta
                + (1.0 - CORNER_TRACK_VELOCITY_ALPHA)
                * self.corner_velocity
            )

        max_motion = (
            np.max(np.linalg.norm(current_delta, axis=1))
            / CORNER_TRACK_SCALE
        )
        if max_motion <= CORNER_TRACK_MOTION_THRESHOLD_PX:
            current_corners = (
                CORNER_TRACK_SMOOTH_ALPHA * current_corners
                + (1.0 - CORNER_TRACK_SMOOTH_ALPHA) * corners
            ).astype(np.float32)

        if len(tracked_points) < CORNER_TRACK_REFRESH_FEATURES:
            refreshed_points = self._find_features(gray, current_corners)
            if (
                refreshed_points is not None
                and len(refreshed_points) >= CORNER_TRACK_MIN_FEATURES
            ):
                tracked_points = refreshed_points

        self.previous_gray = gray
        self.previous_points = tracked_points.astype(
            np.float32,
            copy=False
        )
        accepted_corners = current_corners.astype(np.float32)
        self.corners = accepted_corners
        self.predicted_corners = accepted_corners.copy()
        self.predicted_frames = 0
        self.offscreen_predicted_frames = 0
        self.frames_since_detection += 1
        self.frames_since_homography += 1
        self.failures = 0
        return accepted_corners.copy() / CORNER_TRACK_SCALE


def get_tracking_roi(corners, center, frame_width, frame_height):
    """根据靶纸框和靶心创建带边距的跟踪 ROI。"""
    min_x = float(np.min(corners[:, 0]))
    max_x = float(np.max(corners[:, 0]))
    min_y = float(np.min(corners[:, 1]))
    max_y = float(np.max(corners[:, 1]))
    center_x, center_y = center

    roi_width = min(
        frame_width,
        max(
            TRACKING_ROI_MIN_WIDTH,
            int(math.ceil(
                2.0 * max(center_x - min_x, max_x - center_x)
                * (1.0 + TRACKING_ROI_MARGIN)
            ))
        )
    )
    roi_height = min(
        frame_height,
        max(
            TRACKING_ROI_MIN_HEIGHT,
            int(math.ceil(
                2.0 * max(center_y - min_y, max_y - center_y)
                * (1.0 + TRACKING_ROI_MARGIN)
            ))
        )
    )

    x1 = int(round(center_x - roi_width / 2.0))
    y1 = int(round(center_y - roi_height / 2.0))
    x1 = max(0, min(frame_width - roi_width, x1))
    y1 = max(0, min(frame_height - roi_height, y1))

    return x1, y1, x1 + roi_width, y1 + roi_height


def make_homography(corners):
    """
    将原图中的四个角点映射到 A4 正视图。

    corners 顺序：
        左上、右上、右下、左下
    """

    if (
        TARGET_ACCEL_USE_CPP_PERSPECTIVE
        and _NATIVE_BACKEND.is_available
    ):
        homography = _NATIVE_BACKEND.make_homography(
            corners,
            RECT_WIDTH,
            RECT_HEIGHT
        )
        if homography is not None:
            return homography

    top_edge = np.linalg.norm(corners[1] - corners[0])
    right_edge = np.linalg.norm(corners[2] - corners[1])

    if top_edge > right_edge:
        dst_points = np.array([
            [0, RECT_HEIGHT - 1],
            [0, 0],
            [RECT_WIDTH - 1, 0],
            [RECT_WIDTH - 1, RECT_HEIGHT - 1]
        ], dtype=np.float32)
    else:
        dst_points = np.array([
            [0, 0],
            [RECT_WIDTH - 1, 0],
            [RECT_WIDTH - 1, RECT_HEIGHT - 1],
            [0, RECT_HEIGHT - 1]
        ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(
        corners.astype(np.float32),
        dst_points
    )

    return H


def rectify_image(
    frame,
    H
):
    if (
        TARGET_ACCEL_USE_CPP_PERSPECTIVE
        and _NATIVE_BACKEND.is_available
    ):
        rectified = _NATIVE_BACKEND.warp_perspective(
            frame,
            H,
            RECT_WIDTH,
            RECT_HEIGHT
        )
        if rectified is not None:
            return rectified

    rectified = cv2.warpPerspective(
        frame,
        H,
        (RECT_WIDTH, RECT_HEIGHT),
        flags=cv2.INTER_LINEAR
    )

    return rectified


def rectified_to_camera(center, H):

    if center is None:
        return None

    if (
        TARGET_ACCEL_USE_CPP_PERSPECTIVE
        and _NATIVE_BACKEND.is_available
    ):
        original = _NATIVE_BACKEND.inverse_project_point(
            center,
            H
        )
        if original is not None:
            return original

    # 求逆矩阵
    H_inv = np.linalg.inv(H)

    point = np.array(
        [[[center[0], center[1]]]],
        dtype=np.float32
    )

    original_point = cv2.perspectiveTransform(
        point,
        H_inv
    )

    x = float(
        original_point[0, 0, 0]
    )

    y = float(
        original_point[0, 0, 1]
    )

    return x, y


def target_frame_center_to_camera(H):
    """将靶纸标准校正图的几何中心反投影到相机图像。"""
    return rectified_to_camera(
        (
            (RECT_WIDTH - 1) / 2.0,
            (RECT_HEIGHT - 1) / 2.0
        ),
        H
    )
