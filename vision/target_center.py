"""红色圆环提取与同心圆靶心拟合。"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from common.config import (
    TARGET_ACCEL_USE_CPP_RING_FIT,
    CENTER_MAX_OFFSET_PX,
    CENTER_PRIOR_WEIGHT,
    CHECK_RING_RADIUS,
    CENTER_DETECT_SCALE,
    FAR_CENTER_DETECT_SCALE,
    FAR_RED_EXCESS_THRESHOLD,
    FAR_WEAK_RED_EXCESS_THRESHOLD,
    FIT_RMS_WARNING_THRESHOLD,
    MAX_POINTS_PER_RING,
    MAX_RADIUS_ERROR,
    RECT_HEIGHT,
    RECT_WIDTH,
    RAW_CENTER_CLUSTER_PX,
    RAW_CENTER_DETECT_SCALE,
    RAW_CENTER_MIN_CANDIDATES,
    RAW_CENTER_RED_THRESHOLD,
    RED_CLOSE_KERNEL,
    RED_HSV_HIGH1,
    RED_HSV_HIGH2,
    RED_HSV_LOW1,
    RED_HSV_LOW2,
    RED_LOW_LIGHT_MAX_VALUE,
    RED_LOW_LIGHT_MIN_SATURATION,
    RED_LOW_LIGHT_MIN_VALUE,
    RING_RADII_PX,
    RING_SEARCH_TOLERANCE,
)
from vision.native_backend import get_native_backend

_NATIVE_BACKEND = get_native_backend()

CANDIDATE_SCORE_MAX_POINTS = 16


def detect_raw_target_center(frame):
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return None

    analysis_width = max(1, int(round(width * RAW_CENTER_DETECT_SCALE)))
    analysis_height = max(1, int(round(height * RAW_CENTER_DETECT_SCALE)))
    analysis = cv2.resize(
        frame,
        (analysis_width, analysis_height),
        interpolation=cv2.INTER_AREA
    )
    blue = analysis[:, :, 0]
    green = analysis[:, :, 1]
    red = analysis[:, :, 2]
    red_excess = cv2.subtract(
        red,
        cv2.max(green, blue)
    )
    _, mask = cv2.threshold(
        red_excess,
        RAW_CENTER_RED_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        RED_CLOSE_KERNEL,
        iterations=1
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
    scale_x = width / float(analysis_width)
    scale_y = height / float(analysis_height)
    candidates = []
    for contour in contours:
        if len(contour) < 8:
            continue
        (center_x, center_y), (axis_a, axis_b), _ = cv2.fitEllipse(
            contour
        )
        min_axis = min(axis_a, axis_b)
        if min_axis < 4.0:
            continue
        axis_ratio = max(axis_a, axis_b) / min_axis
        radius = (axis_a + axis_b) / 4.0
        if radius < 6.0 or axis_ratio > 2.2:
            continue
        candidates.append((
            np.asarray(
                [
                    center_x * scale_x,
                    center_y * scale_y
                ],
                dtype=np.float32
            ),
            radius * (scale_x + scale_y) * 0.5
        ))

    if len(candidates) < RAW_CENTER_MIN_CANDIDATES:
        return None

    best_cluster = []
    best_score = None
    for center, radius in candidates:
        cluster = [
            candidate
            for candidate in candidates
            if np.linalg.norm(
                candidate[0] - center
            ) <= RAW_CENTER_CLUSTER_PX
        ]
        if len(cluster) < RAW_CENTER_MIN_CANDIDATES:
            continue
        radius_bins = {
            int(round(candidate[1] / 10.0))
            for candidate in cluster
        }
        score = (
            len(radius_bins),
            len(cluster)
        )
        if (
            len(radius_bins) >= 2
            and (
                best_score is None
                or score > best_score
            )
        ):
            best_score = score
            best_cluster = cluster

    if not best_cluster:
        return None
    center = np.median(
        np.asarray(
            [candidate[0] for candidate in best_cluster],
            dtype=np.float32
        ),
        axis=0
    )
    return (
        float(center[0]),
        float(center[1])
    )


def extract_red_mask(rectified, red_excess_threshold=8):
    blue = rectified[:, :, 0]
    green = rectified[:, :, 1]
    red = rectified[:, :, 2]
    red_reference = cv2.max(green, blue)
    red_excess = cv2.subtract(red, red_reference)
    _, base_mask = cv2.threshold(
        red_excess,
        red_excess_threshold,
        255,
        cv2.THRESH_BINARY
    )
    base_mask = cv2.morphologyEx(
        base_mask,
        cv2.MORPH_CLOSE,
        RED_CLOSE_KERNEL,
        iterations=1
    )
    scene_value = max(cv2.mean(rectified)[:3])
    return base_mask, (
        scene_value,
        RED_LOW_LIGHT_MIN_VALUE,
        0
    )


def extract_dim_red_mask(rectified, base_mask):
    hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)
    dim_red_low1 = np.array(
        [
            RED_HSV_LOW1[0],
            RED_LOW_LIGHT_MIN_SATURATION,
            RED_LOW_LIGHT_MIN_VALUE
        ],
        dtype=np.uint8
    )
    dim_red_low2 = np.array(
        [
            RED_HSV_LOW2[0],
            RED_LOW_LIGHT_MIN_SATURATION,
            RED_LOW_LIGHT_MIN_VALUE
        ],
        dtype=np.uint8
    )
    dim_red_high1 = np.array(
        [RED_HSV_HIGH1[0], 255, RED_LOW_LIGHT_MAX_VALUE],
        dtype=np.uint8
    )
    dim_red_high2 = np.array(
        [RED_HSV_HIGH2[0], 255, RED_LOW_LIGHT_MAX_VALUE],
        dtype=np.uint8
    )
    dim_red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, dim_red_low1, dim_red_high1),
        cv2.inRange(hsv, dim_red_low2, dim_red_high2)
    )
    low_light_added = cv2.bitwise_and(
        dim_red_mask,
        cv2.bitwise_not(base_mask)
    )
    candidate_count = cv2.countNonZero(low_light_added)
    if candidate_count == 0:
        return base_mask, 0

    expanded_mask = cv2.morphologyEx(
        cv2.bitwise_or(base_mask, dim_red_mask),
        cv2.MORPH_CLOSE,
        RED_CLOSE_KERNEL,
        iterations=1
    )
    return expanded_mask, candidate_count


def collect_ring_points(
    mask,
    center,
    pixel_coordinates=None,
    max_points=MAX_POINTS_PER_RING,
    limit_points=True,
    expected_radii=RING_RADII_PX,
    search_tolerance: float = RING_SEARCH_TOLERANCE
):
    """
    根据当前靶心估计值，将红色像素分配给五个圆环。

    center:
        当前估计的靶心 (x, y)

    返回圆环点组及失败原因。
    """

    if pixel_coordinates is None:
        ys, xs = np.nonzero(mask)
    else:
        xs, ys = pixel_coordinates

    if len(xs) == 0:
        return None, "empty red mask"

    cx, cy = center

    xs_float = np.asarray(xs, dtype=np.float32)
    ys_float = np.asarray(ys, dtype=np.float32)
    dx = xs_float - cx
    dy = ys_float - cy

    squared_distances = dx * dx + dy * dy

    groups = []

    for ring_index, expected_radius in enumerate(expected_radii):
        # 直接比较半径平方，避免逐像素计算平方根。
        inner_radius = expected_radius - search_tolerance
        outer_radius = expected_radius + search_tolerance

        valid = (
            (squared_distances >= inner_radius * inner_radius)
            & (squared_distances <= outer_radius * outer_radius)
        )

        points = np.column_stack((
            xs[valid],
            ys[valid]
        )).astype(np.float64)

        if len(points) < 30:
            return None, "ring {}: {} px".format(
                ring_index + 1,
                len(points)
            )

        # 避免单个圆环的点数过多
        if limit_points and len(points) > max_points:

            indices = np.linspace(
                0,
                len(points) - 1,
                max_points,
                dtype=np.int32
            )

            points = points[indices]

        groups.append(points)

    return groups, None


def joint_circle_fit(groups, scale_x=1.0, scale_y=1.0):
    """
    多个圆共用一个圆心，但每个圆有独立半径。

    展开圆方程后，以每个环的半径常数项作为独立未知量，联合最小二乘拟合。
    """

    ring_count = len(groups)
    if ring_count < 3:
        return None

    reference_x = RECT_WIDTH * 0.5 * scale_x - 0.5
    reference_y = RECT_HEIGHT * 0.5 * scale_y - 0.5
    matrix_size = 2 + ring_count
    normal = np.zeros((matrix_size, matrix_size), dtype=np.float64)
    values = np.zeros(matrix_size, dtype=np.float64)

    for ring_index, points in enumerate(groups):
        x = points[:, 0] - reference_x
        y = points[:, 1] - reference_y
        twice_x = 2.0 * x
        twice_y = 2.0 * y
        squared_distance = x * x + y * y
        ring_column = 2 + ring_index

        normal[0, 0] += np.dot(twice_x, twice_x)
        normal[0, 1] += np.dot(twice_x, twice_y)
        normal[1, 1] += np.dot(twice_y, twice_y)
        normal[0, ring_column] = -np.sum(twice_x)
        normal[1, ring_column] = -np.sum(twice_y)
        normal[ring_column, ring_column] = len(points)

        values[0] += np.dot(twice_x, squared_distance)
        values[1] += np.dot(twice_y, squared_distance)
        values[ring_column] = -np.sum(squared_distance)

    normal[1, 0] = normal[0, 1]

    for ring_index in range(ring_count):
        ring_column = 2 + ring_index
        normal[ring_column, 0] = normal[0, ring_column]
        normal[ring_column, 1] = normal[1, ring_column]

    try:
        solution = np.linalg.solve(normal, values)
    except np.linalg.LinAlgError:
        return None

    cx = reference_x + solution[0]
    cy = reference_y + solution[1]

    radii = []

    for ring_index in range(ring_count):
        q = solution[2 + ring_index]
        radius_squared = (
            solution[0] * solution[0]
            + solution[1] * solution[1]
            - q
        )

        if radius_squared <= 0:
            return None

        radii.append(math.sqrt(radius_squared))

    return (
        float(cx),
        float(cy),
        radii
    )


def calculate_fit_rms(groups, center, radii, scale=1.0):
    cx, cy = center
    squared_error_sum = 0.0
    point_count = 0

    for points, radius in zip(groups, radii):
        dx = points[:, 0] - cx
        dy = points[:, 1] - cy
        error = np.sqrt(dx * dx + dy * dy) - radius
        squared_error_sum += float(np.dot(error, error))
        point_count += error.size

    if point_count == 0:
        return 999.0

    rms = math.sqrt(squared_error_sum / point_count)
    if rms <= FIT_RMS_WARNING_THRESHOLD * scale:
        return rms

    ring_scores = []
    for points, radius in zip(groups, radii):
        dx = points[:, 0] - cx
        dy = points[:, 1] - cy
        error = np.sqrt(dx * dx + dy * dy) - radius
        ring_scores.append(float(np.median(error * error)))

    if not ring_scores:
        return rms
    return math.sqrt(sum(ring_scores) / len(ring_scores))


def check_angular_coverage(groups, center):

    cx, cy = center

    for points in groups:

        dx = points[:, 0] - cx
        dy = points[:, 1] - cy

        angles = np.arctan2(
            dy,
            dx
        )

        # 将圆周分成 12 个角度区间
        bins = np.floor(
            (angles + np.pi) / (2.0 * np.pi) * 12
        ).astype(np.int32)

        bins = np.clip(
            bins,
            0,
            11
        )

        occupied = np.count_nonzero(
            np.bincount(bins, minlength=12)
        )

        # 衰减或部分遮挡的圆环仍需保留足够宽的角度覆盖。
        if occupied < 6:
            return False

    return True


def _candidate_has_angular_coverage(points, center):
    """在十二个扇区中检查是否至少覆盖六个，不逐点调用 NumPy 三角函数。"""
    cx, cy = center
    occupied_mask = 0
    occupied_count = 0
    sqrt_three = 1.7320508075688772

    for point in points:
        dx = float(point[0]) - cx
        dy = float(point[1]) - cy
        abs_x = abs(dx)
        abs_y = abs(dy)

        if abs_x == 0.0:
            sector = 9 if dy > 0.0 else 3
        elif abs_y == 0.0:
            sector = 6 if dx > 0.0 else 11
        elif dx > 0.0 and dy > 0.0:
            if abs_y * sqrt_three < abs_x:
                sector = 6
            elif abs_y < abs_x * sqrt_three:
                sector = 7
            else:
                sector = 8
        elif dx < 0.0 and dy > 0.0:
            if abs_y * sqrt_three <= abs_x:
                sector = 11
            elif abs_y <= abs_x * sqrt_three:
                sector = 10
            else:
                sector = 9
        elif dx < 0.0:
            if abs_y * sqrt_three < abs_x:
                sector = 0
            elif abs_y < abs_x * sqrt_three:
                sector = 1
            else:
                sector = 2
        elif abs_y * sqrt_three <= abs_x:
            sector = 5
        elif abs_y <= abs_x * sqrt_three:
            sector = 4
        else:
            sector = 3

        sector_bit = 1 << sector
        if not occupied_mask & sector_bit:
            occupied_mask |= sector_bit
            occupied_count += 1
            if occupied_count >= 6:
                return True

    return False


def _candidate_fit_rms(points, center, radius):
    cx, cy = center
    squared_error_sum = 0.0
    for point in points:
        dx = float(point[0]) - cx
        dy = float(point[1]) - cy
        error = math.sqrt(dx * dx + dy * dy) - radius
        squared_error_sum += error * error
    return math.sqrt(squared_error_sum / len(points))


def _fit_target_center_pixels(mask, scale_x=1.0, scale_y=1.0) -> Tuple[
    Optional[Tuple[float, float]],
    Optional[float],
    Optional[List[float]],
    str
]:
    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        return None, None, None, "empty red mask"

    pixel_coordinates = (
        xs.astype(np.float32),
        ys.astype(np.float32)
    )

    scale = (scale_x + scale_y) / 2.0
    paper_center = (
        RECT_WIDTH * 0.5 * scale_x - 0.5,
        RECT_HEIGHT * 0.5 * scale_y - 0.5
    )
    expected_radii = [radius * scale for radius in RING_RADII_PX]
    red_median = (
        float(np.median(xs)),
        float(np.median(ys))
    )

    candidates = []
    failure_reason = "circle fit failed"
    initial_centers = (paper_center, red_median)
    if red_median == paper_center:
        initial_centers = (paper_center,)

    for initial_center in initial_centers:
        groups, failure = collect_ring_points(
            mask,
            initial_center,
            pixel_coordinates,
            expected_radii=expected_radii,
            search_tolerance=RING_SEARCH_TOLERANCE * scale
        )

        if groups is None:
            failure_reason = (
                failure if failure is not None else "circle fit failed"
            )
            continue

        ring_centers = []
        radii = []

        for points in groups:
            ellipse = cv2.fitEllipse(
                points.astype(np.float32).reshape(-1, 1, 2)
            )
            (ring_cx, ring_cy), (axis_a, axis_b), _ = ellipse
            ring_centers.append((ring_cx, ring_cy))
            radii.append((axis_a + axis_b) / 4.0)

        center_array = np.median(
            np.asarray(ring_centers, dtype=np.float64),
            axis=0
        )
        center = (float(center_array[0]), float(center_array[1]))

        ring_center_spread = max(
            np.linalg.norm(
                np.asarray(ring_center, dtype=np.float64)
                - center_array
            )
            for ring_center in ring_centers
        )
        if ring_center_spread > max(
            8.0,
            RING_SEARCH_TOLERANCE
        ):
            failure_reason = "ring centers inconsistent"
            continue

        if not check_angular_coverage(groups, center):
            failure_reason = "poor ring coverage"
            continue

        rms = calculate_fit_rms(groups, center, radii, scale)

        if CHECK_RING_RADIUS:
            invalid_radius = None
            for ring_index, (fitted_radius, expected_radius) in enumerate(zip(
                radii,
                expected_radii
            )):
                if abs(fitted_radius - expected_radius) > MAX_RADIUS_ERROR * scale:
                    invalid_radius = "ring {} radius {:.1f}".format(
                        ring_index + 1,
                        fitted_radius
                    )
                    break

            if invalid_radius is not None:
                failure_reason = invalid_radius
                continue

        if not (
            0 <= center[0] < mask.shape[1]
            and 0 <= center[1] < mask.shape[0]
        ):
            failure_reason = "center out of bounds"
            continue

        center_full = (
            (center[0] + 0.5) / scale_x - 0.5,
            (center[1] + 0.5) / scale_y - 0.5
        )
        distance_from_paper_center = math.hypot(
            center_full[0] - (RECT_WIDTH - 1) / 2.0,
            center_full[1] - (RECT_HEIGHT - 1) / 2.0
        )
        if distance_from_paper_center > CENTER_MAX_OFFSET_PX:
            failure_reason = "center too far from paper center"
            continue
        rms_full = rms / scale
        score = rms_full + CENTER_PRIOR_WEIGHT * distance_from_paper_center
        candidates.append((
            score,
            center_full,
            rms_full,
            [radius / scale for radius in radii]
        ))

    if not candidates:
        return None, None, None, failure_reason

    _, center, rms, radii = min(
        candidates,
        key=lambda item: item[0]
    )

    fit_status = "OK"
    if rms > FIT_RMS_WARNING_THRESHOLD:
        fit_status = "low confidence: RMS {:.1f}px > {:.1f}px".format(
            rms,
            FIT_RMS_WARNING_THRESHOLD
        )

    return center, rms, radii, fit_status


def fit_target_center(mask, scale_x=1.0, scale_y=1.0) -> Tuple[
    Optional[Tuple[float, float]],
    Optional[float],
    Optional[List[float]],
    str
]:
    """优先根据轮廓拟合已知同心圆环，再根据需要进入像素兜底。"""
    if (
        TARGET_ACCEL_USE_CPP_RING_FIT
        and _NATIVE_BACKEND.is_available
    ):
        native_result = _NATIVE_BACKEND.fit_target_center_mask(
            mask=mask,
            scale_x=scale_x,
            scale_y=scale_y,
            expected_radii_px=RING_RADII_PX,
            search_tolerance=RING_SEARCH_TOLERANCE,
            max_radius_error=MAX_RADIUS_ERROR,
            check_ring_radius=CHECK_RING_RADIUS,
            center_prior_weight=CENTER_PRIOR_WEIGHT,
            fit_rms_warning_threshold=FIT_RMS_WARNING_THRESHOLD,
            rect_width=RECT_WIDTH,
            rect_height=RECT_HEIGHT
        )
        if native_result is not None:
            center, rms, radii, fit_status = native_result
            if (
                center is not None
                and fit_status == "OK"
                and math.hypot(
                    center[0] - (RECT_WIDTH - 1) / 2.0,
                    center[1] - (RECT_HEIGHT - 1) / 2.0
                ) <= CENTER_MAX_OFFSET_PX
            ):
                return center, rms, radii, fit_status

    scale = (scale_x + scale_y) / 2.0
    contours, _ = cv2.findContours(
        mask.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
    expected_radii = [radius * scale for radius in RING_RADII_PX]
    candidates = [[] for _ in RING_RADII_PX]

    for contour in contours:
        if len(contour) < 5:
            continue

        _, _, box_width, box_height = cv2.boundingRect(contour)
        box_radius = (box_width + box_height) * 0.25
        if min(
            abs(box_radius - expected_radius)
            for expected_radius in expected_radii
        ) > MAX_RADIUS_ERROR * scale + 1.5:
            continue

        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (axis_a, axis_b), _ = ellipse
        radius = (axis_a + axis_b) / 4.0
        ring_index = min(
            range(len(RING_RADII_PX)),
            key=lambda index: abs(radius - expected_radii[index])
        )
        expected_radius = expected_radii[ring_index]

        if CHECK_RING_RADIUS and abs(radius - expected_radius) > MAX_RADIUS_ERROR * scale:
            continue
        if not (0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0]):
            continue

        points = contour.reshape(-1, 2)
        # 候选评分只需宽覆盖度；最终联合拟合保留完整轮廓点。
        sample_step = max(
            1,
            (len(points) + CANDIDATE_SCORE_MAX_POINTS - 1)
            // CANDIDATE_SCORE_MAX_POINTS
        )
        scoring_points = points[::sample_step]
        center = (float(cx), float(cy))
        has_coverage = _candidate_has_angular_coverage(
            scoring_points,
            center
        )
        if not has_coverage:
            continue

        rms = _candidate_fit_rms(scoring_points, center, radius)
        center_prior = math.hypot(
            (cx + 0.5) / scale_x - 0.5 - (RECT_WIDTH - 1) / 2.0,
            (cy + 0.5) / scale_y - 0.5 - (RECT_HEIGHT - 1) / 2.0
        )
        score = (
            rms / scale
            + CENTER_PRIOR_WEIGHT * center_prior
            + 0.05 * abs(radius / scale - RING_RADII_PX[ring_index])
        )
        candidates[ring_index].append(
            (score, center, radius, points)
        )

    selected = [
        (ring_index, min(ring_candidates, key=lambda item: item[0]))
        for ring_index, ring_candidates in enumerate(candidates)
        if ring_candidates
    ]
    if len(selected) < 3:
        return _fit_target_center_pixels(mask, scale_x, scale_y)

    center_array = np.median(
        np.asarray([item[1][1] for item in selected], dtype=np.float64),
        axis=0
    )
    center_local = (float(center_array[0]), float(center_array[1]))
    center_spread = max(
        np.linalg.norm(item[1][1] - center_array)
        for item in selected
    ) / scale
    if center_spread > max(8.0, RING_SEARCH_TOLERANCE):
        return _fit_target_center_pixels(mask, scale_x, scale_y)

    groups = [item[1][3] for item in selected]
    joint_fit = joint_circle_fit(groups, scale_x, scale_y)
    if joint_fit is None:
        return None, None, None, "joint circle fit failed"
    center_local = joint_fit[:2]
    fitted_radii = joint_fit[2]
    expected_selected_radii = [
        expected_radii[item[0]]
        for item in selected
    ]
    if CHECK_RING_RADIUS and any(
        abs(fitted - expected) > MAX_RADIUS_ERROR * scale
        for fitted, expected in zip(fitted_radii, expected_selected_radii)
    ):
        return _fit_target_center_pixels(mask, scale_x, scale_y)

    center = (
        (center_local[0] + 0.5) / scale_x - 0.5,
        (center_local[1] + 0.5) / scale_y - 0.5
    )
    if math.hypot(
        center[0] - (RECT_WIDTH - 1) / 2.0,
        center[1] - (RECT_HEIGHT - 1) / 2.0
    ) > CENTER_MAX_OFFSET_PX:
        return _fit_target_center_pixels(mask, scale_x, scale_y)
    radii = [radius / scale for radius in fitted_radii]
    rms = calculate_fit_rms(
        groups,
        center_local,
        fitted_radii,
        scale
    ) / scale

    fit_status = "OK"
    if rms > FIT_RMS_WARNING_THRESHOLD:
        fit_status = "low confidence: RMS {:.1f}px > {:.1f}px".format(
            rms,
            FIT_RMS_WARNING_THRESHOLD
        )

    return center, rms, radii, fit_status


def detect_target_center(rectified, distant=False):
    """
    计算靶心；远距离标准掩膜失败时增加低反差圆环重试。
    """

    process_width = max(1, int(round(rectified.shape[1] * CENTER_DETECT_SCALE)))
    process_height = max(1, int(round(rectified.shape[0] * CENTER_DETECT_SCALE)))
    process_image = cv2.resize(
        rectified,
        (process_width, process_height),
        interpolation=cv2.INTER_AREA
    )
    scale_x = process_width / float(rectified.shape[1])
    scale_y = process_height / float(rectified.shape[0])

    base_mask, red_low_light_info = extract_red_mask(
        process_image
    )

    center, rms, radii, fit_status = fit_target_center(
        base_mask,
        scale_x,
        scale_y
    )
    selected_mask = base_mask
    low_light_attempted = False
    low_light_used = False

    low_light_candidate_count = 0
    if (
        center is None
        or rms is None
        or rms > FIT_RMS_WARNING_THRESHOLD
    ):
        expanded_mask, low_light_candidate_count = extract_dim_red_mask(
            process_image,
            base_mask
        )
        if low_light_candidate_count > 0:
            low_light_attempted = True
            expanded_center, expanded_rms, expanded_radii, expanded_status = (
                fit_target_center(
                    expanded_mask,
                    scale_x,
                    scale_y
                )
            )
            if (
                expanded_center is not None
                and expanded_rms is not None
                and (
                    center is None
                    or rms is None
                    or expanded_rms < rms
                )
            ):
                center = expanded_center
                rms = expanded_rms
                radii = expanded_radii
                fit_status = expanded_status
                selected_mask = expanded_mask
                low_light_used = True

    if (
        distant
        and (
            center is None
            or rms is None
            or rms > FIT_RMS_WARNING_THRESHOLD
        )
    ):
        scaled_images = {
            CENTER_DETECT_SCALE: (
                process_image,
                scale_x,
                scale_y
            )
        }
        retry_options = (
            (CENTER_DETECT_SCALE, FAR_RED_EXCESS_THRESHOLD),
            (FAR_CENTER_DETECT_SCALE, FAR_RED_EXCESS_THRESHOLD),
            (FAR_CENTER_DETECT_SCALE, FAR_WEAK_RED_EXCESS_THRESHOLD)
        )

        for retry_scale, red_threshold in retry_options:
            if retry_scale not in scaled_images:
                retry_width = max(
                    1,
                    int(round(rectified.shape[1] * retry_scale))
                )
                retry_height = max(
                    1,
                    int(round(rectified.shape[0] * retry_scale))
                )
                retry_image = cv2.resize(
                    rectified,
                    (retry_width, retry_height),
                    interpolation=cv2.INTER_AREA
                )
                scaled_images[retry_scale] = (
                    retry_image,
                    retry_width / float(rectified.shape[1]),
                    retry_height / float(rectified.shape[0])
                )

            retry_image, retry_scale_x, retry_scale_y = (
                scaled_images[retry_scale]
            )
            retry_mask, _ = extract_red_mask(
                retry_image,
                red_excess_threshold=red_threshold
            )
            retry_center, retry_rms, retry_radii, retry_status = (
                fit_target_center(
                    retry_mask,
                    retry_scale_x,
                    retry_scale_y
                )
            )
            if (
                retry_center is not None
                and retry_rms is not None
                and (
                    center is None
                    or rms is None
                    or retry_rms < rms
                )
            ):
                center = retry_center
                rms = retry_rms
                radii = retry_radii
                fit_status = retry_status
                selected_mask = retry_mask

            if fit_status == "OK":
                break

    red_low_light_info = (
        red_low_light_info[0],
        red_low_light_info[1],
        low_light_candidate_count,
        low_light_attempted,
        low_light_used
    )
    return center, rms, radii, selected_mask, fit_status, red_low_light_info
