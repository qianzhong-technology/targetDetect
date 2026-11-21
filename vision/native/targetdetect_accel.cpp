#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define TD_HAVE_NEON 1
#else
#define TD_HAVE_NEON 0
#endif

namespace {

constexpr int kCandidateScoreMaxPoints = 16;

struct RingCandidate {
    double score = std::numeric_limits<double>::max();
    cv::Point2f center;
    double radius = 0.0;
    int ring_index = -1;
    std::vector<cv::Point2f> points;
};

bool IsFinitePoint(const cv::Point2f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
}

double CandidateFitRmsScalar(
    const std::vector<cv::Point2f>& points,
    const cv::Point2f center,
    const double radius
) {
    if (points.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    double squared_error_sum = 0.0;
    for (const auto& point : points) {
        const double dx = static_cast<double>(point.x) - center.x;
        const double dy = static_cast<double>(point.y) - center.y;
        const double error = std::sqrt(dx * dx + dy * dy) - radius;
        squared_error_sum += error * error;
    }
    return std::sqrt(squared_error_sum / static_cast<double>(points.size()));
}

double CandidateFitRms(
    const std::vector<cv::Point2f>& points,
    const cv::Point2f center,
    const double radius
) {
    if (points.empty()) {
        return std::numeric_limits<double>::infinity();
    }

#if TD_HAVE_NEON
    const float32x4_t center_x = vdupq_n_f32(center.x);
    const float32x4_t center_y = vdupq_n_f32(center.y);
    const float32x4_t radius_v = vdupq_n_f32(static_cast<float>(radius));
    const float32x4_t zero_v = vdupq_n_f32(0.0f);
    float32x4_t sum_v = vdupq_n_f32(0.0f);

    int index = 0;
    const int size = static_cast<int>(points.size());
    for (; index + 4 <= size; index += 4) {
        const float* raw = reinterpret_cast<const float*>(&points[index]);
        const float32x4x2_t xy = vld2q_f32(raw);
        const float32x4_t dx = vsubq_f32(xy.val[0], center_x);
        const float32x4_t dy = vsubq_f32(xy.val[1], center_y);
        const float32x4_t distance = vsqrtq_f32(
            vaddq_f32(vmulq_f32(dx, dx), vmulq_f32(dy, dy))
        );
        const float32x4_t error = vsubq_f32(distance, radius_v);
        sum_v = vmlaq_f32(sum_v, error, error);
    }

    float squared_error_sum = vaddvq_f32(sum_v);
    if (size - index > 0) {
        std::vector<cv::Point2f> tail(
            points.begin() + index,
            points.end()
        );
        squared_error_sum += static_cast<float>(
            std::pow(CandidateFitRmsScalar(tail, center, radius), 2.0)
        ) * static_cast<float>(tail.size());
    }
    if (squared_error_sum < 0.0f) {
        squared_error_sum = 0.0f;
    }
    return std::sqrt(
        static_cast<double>(squared_error_sum) / static_cast<double>(points.size())
    );
#else
    return CandidateFitRmsScalar(points, center, radius);
#endif
}

bool HasAngularCoverage(
    const std::vector<cv::Point2f>& points,
    const cv::Point2f center
) {
    int occupied_mask = 0;
    int occupied_count = 0;
    constexpr double sqrt_three = 1.7320508075688772;

    for (const auto& point : points) {
        const double dx = static_cast<double>(point.x) - center.x;
        const double dy = static_cast<double>(point.y) - center.y;
        const double abs_x = std::abs(dx);
        const double abs_y = std::abs(dy);

        int sector = 0;
        if (abs_x == 0.0) {
            sector = dy > 0.0 ? 9 : 3;
        } else if (abs_y == 0.0) {
            sector = dx > 0.0 ? 6 : 11;
        } else if (dx > 0.0 && dy > 0.0) {
            if (abs_y * sqrt_three < abs_x) {
                sector = 6;
            } else if (abs_y < abs_x * sqrt_three) {
                sector = 7;
            } else {
                sector = 8;
            }
        } else if (dx < 0.0 && dy > 0.0) {
            if (abs_y * sqrt_three <= abs_x) {
                sector = 11;
            } else if (abs_y <= abs_x * sqrt_three) {
                sector = 10;
            } else {
                sector = 9;
            }
        } else if (dx < 0.0) {
            if (abs_y * sqrt_three < abs_x) {
                sector = 0;
            } else if (abs_y < abs_x * sqrt_three) {
                sector = 1;
            } else {
                sector = 2;
            }
        } else if (abs_y * sqrt_three <= abs_x) {
            sector = 5;
        } else if (abs_y <= abs_x * sqrt_three) {
            sector = 4;
        } else {
            sector = 3;
        }

        const int sector_bit = (1 << sector);
        if ((occupied_mask & sector_bit) == 0) {
            occupied_mask |= sector_bit;
            occupied_count += 1;
            if (occupied_count >= 6) {
                return true;
            }
        }
    }

    return false;
}

double Median(std::vector<double>* values) {
    if (values == nullptr || values->empty()) {
        return 0.0;
    }
    std::sort(values->begin(), values->end());
    const std::size_t mid = values->size() / 2;
    if ((values->size() & 1U) == 0U) {
        return ((*values)[mid - 1] + (*values)[mid]) * 0.5;
    }
    return (*values)[mid];
}

bool JointCircleFit(
    const std::vector<std::vector<cv::Point2f>>& groups,
    const double scale_x,
    const double scale_y,
    const int rect_width,
    const int rect_height,
    cv::Point2d* out_center,
    std::vector<double>* out_radii
) {
    if (groups.size() < 3 || out_center == nullptr || out_radii == nullptr) {
        return false;
    }

    const int ring_count = static_cast<int>(groups.size());
    const double reference_x = (static_cast<double>(rect_width) - 0.5) * scale_x - 0.5;
    const double reference_y = (static_cast<double>(rect_height) - 0.5) * scale_y - 0.5;
    const int matrix_size = 2 + ring_count;

    cv::Mat normal = cv::Mat::zeros(matrix_size, matrix_size, CV_64F);
    cv::Mat values = cv::Mat::zeros(matrix_size, 1, CV_64F);

    for (int ring = 0; ring < ring_count; ++ring) {
        const auto& points = groups[ring];
        if (points.empty()) {
            return false;
        }
        const int column = 2 + ring;

        double sum_twice_x_square = 0.0;
        double sum_twice_x_twice_y = 0.0;
        double sum_twice_y_square = 0.0;
        double sum_twice_x = 0.0;
        double sum_twice_y = 0.0;
        double sum_twice_x_sd = 0.0;
        double sum_twice_y_sd = 0.0;
        double sum_sd = 0.0;

        for (const auto& point : points) {
            const double x = static_cast<double>(point.x) - reference_x;
            const double y = static_cast<double>(point.y) - reference_y;
            const double tx = 2.0 * x;
            const double ty = 2.0 * y;
            const double sd = x * x + y * y;
            sum_twice_x_square += tx * tx;
            sum_twice_x_twice_y += tx * ty;
            sum_twice_y_square += ty * ty;
            sum_twice_x += tx;
            sum_twice_y += ty;
            sum_twice_x_sd += tx * sd;
            sum_twice_y_sd += ty * sd;
            sum_sd += sd;
        }

        normal.at<double>(0, 0) += sum_twice_x_square;
        normal.at<double>(0, 1) += sum_twice_x_twice_y;
        normal.at<double>(1, 1) += sum_twice_y_square;
        normal.at<double>(0, column) = -sum_twice_x;
        normal.at<double>(1, column) = -sum_twice_y;
        normal.at<double>(column, column) = static_cast<double>(points.size());

        values.at<double>(0, 0) += sum_twice_x_sd;
        values.at<double>(1, 0) += sum_twice_y_sd;
        values.at<double>(column, 0) = -sum_sd;
    }

    normal.at<double>(1, 0) = normal.at<double>(0, 1);
    for (int ring = 0; ring < ring_count; ++ring) {
        const int column = 2 + ring;
        normal.at<double>(column, 0) = normal.at<double>(0, column);
        normal.at<double>(column, 1) = normal.at<double>(1, column);
    }

    cv::Mat solution;
    if (!cv::solve(normal, values, solution, cv::DECOMP_LU)) {
        return false;
    }

    const double local_x = solution.at<double>(0, 0);
    const double local_y = solution.at<double>(1, 0);
    out_center->x = reference_x + local_x;
    out_center->y = reference_y + local_y;

    out_radii->clear();
    out_radii->reserve(ring_count);
    for (int ring = 0; ring < ring_count; ++ring) {
        const double q = solution.at<double>(2 + ring, 0);
        const double radius_squared = local_x * local_x + local_y * local_y - q;
        if (radius_squared <= 0.0) {
            return false;
        }
        out_radii->push_back(std::sqrt(radius_squared));
    }
    return true;
}

double CalculateFitRms(
    const std::vector<std::vector<cv::Point2f>>& groups,
    const cv::Point2d center,
    const std::vector<double>& radii
) {
    if (groups.empty() || groups.size() != radii.size()) {
        return std::numeric_limits<double>::infinity();
    }

    double squared_error_sum = 0.0;
    std::size_t point_count = 0;
    for (std::size_t ring = 0; ring < groups.size(); ++ring) {
        for (const auto& point : groups[ring]) {
            const double dx = static_cast<double>(point.x) - center.x;
            const double dy = static_cast<double>(point.y) - center.y;
            const double error = std::sqrt(dx * dx + dy * dy) - radii[ring];
            squared_error_sum += error * error;
            point_count += 1;
        }
    }
    if (point_count == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(squared_error_sum / static_cast<double>(point_count));
}

}  // namespace

extern "C" {

int32_t td_make_homography(
    const float* corners_xy,
    int32_t rect_width,
    int32_t rect_height,
    double* out_h9
) {
    if (
        corners_xy == nullptr
        || out_h9 == nullptr
        || rect_width <= 0
        || rect_height <= 0
    ) {
        return -1;
    }

    std::vector<cv::Point2f> corners(4);
    for (int i = 0; i < 4; ++i) {
        corners[i] = cv::Point2f(corners_xy[i * 2], corners_xy[i * 2 + 1]);
    }

    const double top_edge = cv::norm(corners[1] - corners[0]);
    const double right_edge = cv::norm(corners[2] - corners[1]);

    std::vector<cv::Point2f> dst_points;
    dst_points.reserve(4);
    if (top_edge > right_edge) {
        dst_points.emplace_back(0.0f, static_cast<float>(rect_height - 1));
        dst_points.emplace_back(0.0f, 0.0f);
        dst_points.emplace_back(static_cast<float>(rect_width - 1), 0.0f);
        dst_points.emplace_back(
            static_cast<float>(rect_width - 1),
            static_cast<float>(rect_height - 1)
        );
    } else {
        dst_points.emplace_back(0.0f, 0.0f);
        dst_points.emplace_back(static_cast<float>(rect_width - 1), 0.0f);
        dst_points.emplace_back(
            static_cast<float>(rect_width - 1),
            static_cast<float>(rect_height - 1)
        );
        dst_points.emplace_back(0.0f, static_cast<float>(rect_height - 1));
    }

    const cv::Mat h = cv::getPerspectiveTransform(corners, dst_points);
    if (h.rows != 3 || h.cols != 3) {
        return -2;
    }

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            out_h9[row * 3 + col] = h.at<double>(row, col);
        }
    }
    return 0;
}

int32_t td_warp_perspective_bgr(
    const uint8_t* frame_data,
    int32_t rows,
    int32_t cols,
    int32_t channels,
    int32_t stride_bytes,
    const double* h9,
    int32_t out_width,
    int32_t out_height,
    uint8_t* out_data,
    int32_t out_stride_bytes
) {
    if (
        frame_data == nullptr
        || h9 == nullptr
        || out_data == nullptr
        || rows <= 0
        || cols <= 0
        || channels != 3
        || out_width <= 0
        || out_height <= 0
    ) {
        return -1;
    }

    cv::Mat input(
        rows,
        cols,
        CV_8UC3,
        const_cast<uint8_t*>(frame_data),
        static_cast<std::size_t>(stride_bytes)
    );
    cv::Mat output(
        out_height,
        out_width,
        CV_8UC3,
        out_data,
        static_cast<std::size_t>(out_stride_bytes)
    );
    cv::Mat h(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            h.at<double>(row, col) = h9[row * 3 + col];
        }
    }
    cv::warpPerspective(
        input,
        output,
        h,
        cv::Size(out_width, out_height),
        cv::INTER_LINEAR
    );
    return 0;
}

int32_t td_inverse_project_point(
    const double* h9,
    float x,
    float y,
    float* out_xy
) {
    if (h9 == nullptr || out_xy == nullptr) {
        return -1;
    }

    cv::Mat h(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            h.at<double>(row, col) = h9[row * 3 + col];
        }
    }

    cv::Mat h_inv;
    if (std::abs(cv::determinant(h)) <= 1e-9) {
        return -2;
    }
    cv::invert(h, h_inv, cv::DECOMP_SVD);
    std::vector<cv::Point2f> src = {cv::Point2f(x, y)};
    std::vector<cv::Point2f> dst;
    cv::perspectiveTransform(src, dst, h_inv);
    if (dst.empty() || !IsFinitePoint(dst[0])) {
        return -3;
    }
    out_xy[0] = dst[0].x;
    out_xy[1] = dst[0].y;
    return 0;
}

int32_t td_lk_track_corners(
    const uint8_t* previous_gray_data,
    int32_t previous_rows,
    int32_t previous_cols,
    int32_t previous_stride,
    const uint8_t* gray_data,
    int32_t rows,
    int32_t cols,
    int32_t stride,
    const float* previous_points_xy,
    int32_t point_count,
    const float* previous_corners_xy,
    int32_t min_features,
    float min_inlier_ratio,
    float support_min_ratio,
    float area_min_ratio,
    float area_max_ratio,
    float* out_current_corners_xy,
    float* out_inlier_points_xy,
    int32_t out_inlier_capacity,
    int32_t* out_inlier_points_count,
    int32_t* out_inlier_count,
    float* out_support_ratio,
    float* out_area_ratio,
    float* out_max_shift
) {
    if (
        previous_gray_data == nullptr
        || gray_data == nullptr
        || previous_points_xy == nullptr
        || previous_corners_xy == nullptr
        || out_current_corners_xy == nullptr
        || out_inlier_points_xy == nullptr
        || out_inlier_points_count == nullptr
        || out_inlier_count == nullptr
        || out_support_ratio == nullptr
        || out_area_ratio == nullptr
        || out_max_shift == nullptr
        || previous_rows <= 0
        || previous_cols <= 0
        || rows <= 0
        || cols <= 0
        || point_count <= 0
        || out_inlier_capacity <= 0
        || min_features <= 0
    ) {
        return -1;
    }

    cv::Mat previous_gray(
        previous_rows,
        previous_cols,
        CV_8UC1,
        const_cast<uint8_t*>(previous_gray_data),
        static_cast<std::size_t>(previous_stride)
    );
    cv::Mat gray(
        rows,
        cols,
        CV_8UC1,
        const_cast<uint8_t*>(gray_data),
        static_cast<std::size_t>(stride)
    );

    std::vector<cv::Point2f> previous_points;
    previous_points.reserve(point_count);
    for (int i = 0; i < point_count; ++i) {
        previous_points.emplace_back(
            previous_points_xy[i * 2],
            previous_points_xy[i * 2 + 1]
        );
    }

    std::vector<cv::Point2f> previous_corners(4);
    for (int i = 0; i < 4; ++i) {
        previous_corners[i] = cv::Point2f(
            previous_corners_xy[i * 2],
            previous_corners_xy[i * 2 + 1]
        );
    }

    std::vector<cv::Point2f> next_points;
    std::vector<uint8_t> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
        previous_gray,
        gray,
        previous_points,
        next_points,
        status,
        err,
        cv::Size(11, 11),
        2,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 15, 0.03)
    );
    if (next_points.empty() || status.empty()) {
        return -2;
    }

    std::vector<cv::Point2f> previous_valid;
    std::vector<cv::Point2f> current_valid;
    previous_valid.reserve(point_count);
    current_valid.reserve(point_count);
    for (int i = 0; i < point_count; ++i) {
        if (status[i] == 0 || !IsFinitePoint(next_points[i])) {
            continue;
        }
        previous_valid.push_back(previous_points[i]);
        current_valid.push_back(next_points[i]);
    }
    if (static_cast<int>(previous_valid.size()) < min_features) {
        return -3;
    }

    cv::Mat inlier_mask;
    const cv::Mat homography = cv::findHomography(
        previous_valid,
        current_valid,
        cv::RANSAC,
        3.0,
        inlier_mask
    );
    if (homography.empty() || inlier_mask.empty()) {
        return -4;
    }

    int inlier_count = 0;
    const int inlier_mask_len = static_cast<int>(inlier_mask.total());
    for (int i = 0; i < inlier_mask_len; ++i) {
        if (inlier_mask.data[i] != 0U) {
            inlier_count += 1;
        }
    }
    const double inlier_ratio = static_cast<double>(inlier_count) /
                                static_cast<double>(previous_valid.size());
    if (inlier_count < min_features || inlier_ratio < min_inlier_ratio) {
        return -5;
    }

    const double previous_area = std::abs(cv::contourArea(previous_corners));
    if (previous_area <= 0.0) {
        return -6;
    }

    std::vector<cv::Point2f> inlier_previous_points;
    inlier_previous_points.reserve(inlier_count);
    std::vector<cv::Point2f> inlier_current_points;
    inlier_current_points.reserve(inlier_count);
    for (int i = 0; i < static_cast<int>(current_valid.size()); ++i) {
        if (inlier_mask.data[i] == 0U) {
            continue;
        }
        inlier_previous_points.push_back(previous_valid[i]);
        inlier_current_points.push_back(current_valid[i]);
    }
    if (inlier_previous_points.empty()) {
        return -7;
    }

    std::vector<cv::Point2f> inlier_hull;
    cv::convexHull(inlier_previous_points, inlier_hull);
    const double support_ratio = previous_area > 0.0
                                     ? std::abs(cv::contourArea(inlier_hull)) / previous_area
                                     : 0.0;
    if (support_ratio < support_min_ratio) {
        return -8;
    }

    std::vector<cv::Point2f> current_corners;
    cv::perspectiveTransform(previous_corners, current_corners, homography);
    if (current_corners.size() != 4) {
        return -9;
    }
    for (const auto& corner : current_corners) {
        if (!IsFinitePoint(corner)) {
            return -10;
        }
    }
    if (!cv::isContourConvex(current_corners)) {
        return -11;
    }

    const double current_area = std::abs(cv::contourArea(current_corners));
    const double area_ratio = current_area / previous_area;
    if (area_ratio < area_min_ratio || area_ratio > area_max_ratio) {
        return -12;
    }

    double max_shift = 0.0;
    for (int i = 0; i < 4; ++i) {
        const double shift = cv::norm(current_corners[i] - previous_corners[i]);
        if (shift > max_shift) {
            max_shift = shift;
        }
    }

    const int output_count = std::min(static_cast<int>(inlier_current_points.size()), out_inlier_capacity);
    for (int i = 0; i < output_count; ++i) {
        out_inlier_points_xy[i * 2] = inlier_current_points[i].x;
        out_inlier_points_xy[i * 2 + 1] = inlier_current_points[i].y;
    }
    for (int i = 0; i < 4; ++i) {
        out_current_corners_xy[i * 2] = current_corners[i].x;
        out_current_corners_xy[i * 2 + 1] = current_corners[i].y;
    }

    *out_inlier_points_count = output_count;
    *out_inlier_count = inlier_count;
    *out_support_ratio = static_cast<float>(support_ratio);
    *out_area_ratio = static_cast<float>(area_ratio);
    *out_max_shift = static_cast<float>(max_shift);
    return 0;
}

int32_t td_fit_target_center_mask(
    const uint8_t* mask_data,
    int32_t rows,
    int32_t cols,
    int32_t stride_bytes,
    float scale_x,
    float scale_y,
    const float* expected_radii_px,
    int32_t ring_count,
    float ring_search_tolerance,
    float max_radius_error,
    int32_t check_ring_radius,
    float center_prior_weight,
    float fit_rms_warning_threshold,
    int32_t rect_width,
    int32_t rect_height,
    float* out_center_xy,
    int32_t* out_status_code,
    float* out_rms,
    float* out_radii,
    int32_t* out_radii_count
) {
    if (
        mask_data == nullptr
        || expected_radii_px == nullptr
        || out_center_xy == nullptr
        || out_status_code == nullptr
        || out_rms == nullptr
        || out_radii == nullptr
        || out_radii_count == nullptr
        || rows <= 0
        || cols <= 0
        || ring_count < 3
        || rect_width <= 0
        || rect_height <= 0
    ) {
        return -10;
    }

    cv::Mat mask(
        rows,
        cols,
        CV_8UC1,
        const_cast<uint8_t*>(mask_data),
        static_cast<std::size_t>(stride_bytes)
    );
    if (cv::countNonZero(mask) == 0) {
        return -1;
    }

    const double scale = (static_cast<double>(scale_x) + static_cast<double>(scale_y)) * 0.5;
    std::vector<double> expected_radii_scaled(ring_count);
    for (int i = 0; i < ring_count; ++i) {
        expected_radii_scaled[i] = static_cast<double>(expected_radii_px[i]) * scale;
    }

    std::vector<std::vector<RingCandidate>> candidates(ring_count);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(
        mask.clone(),
        contours,
        cv::RETR_LIST,
        cv::CHAIN_APPROX_SIMPLE
    );

    for (const auto& contour : contours) {
        if (contour.size() < 5) {
            continue;
        }

        const cv::Rect box = cv::boundingRect(contour);
        const double box_radius = (box.width + box.height) * 0.25;
        double min_box_radius_delta = std::numeric_limits<double>::max();
        for (int ring = 0; ring < ring_count; ++ring) {
            min_box_radius_delta = std::min(
                min_box_radius_delta,
                std::abs(box_radius - expected_radii_scaled[ring])
            );
        }
        if (min_box_radius_delta > max_radius_error * scale + 1.5) {
            continue;
        }

        const cv::RotatedRect ellipse = cv::fitEllipse(contour);
        const double radius = (ellipse.size.width + ellipse.size.height) * 0.25;

        int ring_index = 0;
        double min_radius_delta = std::numeric_limits<double>::max();
        for (int ring = 0; ring < ring_count; ++ring) {
            const double delta = std::abs(radius - expected_radii_scaled[ring]);
            if (delta < min_radius_delta) {
                min_radius_delta = delta;
                ring_index = ring;
            }
        }

        if (
            check_ring_radius != 0
            && std::abs(radius - expected_radii_scaled[ring_index]) > max_radius_error * scale
        ) {
            continue;
        }

        const cv::Point2f center = ellipse.center;
        if (
            center.x < 0.0f
            || center.y < 0.0f
            || center.x >= static_cast<float>(cols)
            || center.y >= static_cast<float>(rows)
        ) {
            continue;
        }

        std::vector<cv::Point2f> points;
        points.reserve(contour.size());
        for (const auto& point : contour) {
            points.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
        }
        if (points.empty()) {
            continue;
        }

        const int sample_step = std::max(
            1,
            (static_cast<int>(points.size()) + kCandidateScoreMaxPoints - 1) / kCandidateScoreMaxPoints
        );
        std::vector<cv::Point2f> scoring_points;
        scoring_points.reserve((points.size() + sample_step - 1) / sample_step);
        for (int i = 0; i < static_cast<int>(points.size()); i += sample_step) {
            scoring_points.push_back(points[i]);
        }

        if (!HasAngularCoverage(scoring_points, center)) {
            continue;
        }

        const double rms = CandidateFitRms(scoring_points, center, radius);
        const double center_prior = std::hypot(
            (center.x + 0.5) / scale_x - 0.5 - (static_cast<double>(rect_width) - 1.0) * 0.5,
            (center.y + 0.5) / scale_y - 0.5 - (static_cast<double>(rect_height) - 1.0) * 0.5
        );
        const double score = rms / scale
            + center_prior_weight * center_prior
            + 0.05 * std::abs(radius / scale - expected_radii_px[ring_index]);

        RingCandidate candidate;
        candidate.score = score;
        candidate.center = center;
        candidate.radius = radius;
        candidate.ring_index = ring_index;
        candidate.points = std::move(points);
        candidates[ring_index].push_back(std::move(candidate));
    }

    std::vector<RingCandidate> selected;
    selected.reserve(ring_count);
    for (int ring = 0; ring < ring_count; ++ring) {
        if (candidates[ring].empty()) {
            continue;
        }
        auto best = std::min_element(
            candidates[ring].begin(),
            candidates[ring].end(),
            [](const RingCandidate& left, const RingCandidate& right) {
                return left.score < right.score;
            }
        );
        selected.push_back(*best);
    }
    if (selected.size() < 3) {
        return -2;
    }

    std::vector<double> selected_x;
    std::vector<double> selected_y;
    selected_x.reserve(selected.size());
    selected_y.reserve(selected.size());
    for (const auto& candidate : selected) {
        selected_x.push_back(candidate.center.x);
        selected_y.push_back(candidate.center.y);
    }

    const cv::Point2d center_array(Median(&selected_x), Median(&selected_y));
    double center_spread = 0.0;
    for (const auto& candidate : selected) {
        center_spread = std::max(
            center_spread,
            cv::norm(cv::Point2d(candidate.center) - center_array)
        );
    }
    center_spread /= scale;
    if (center_spread > std::max(8.0, static_cast<double>(ring_search_tolerance))) {
        return -3;
    }

    std::vector<std::vector<cv::Point2f>> groups;
    groups.reserve(selected.size());
    std::vector<double> expected_selected_radii;
    expected_selected_radii.reserve(selected.size());
    for (const auto& candidate : selected) {
        groups.push_back(candidate.points);
        expected_selected_radii.push_back(expected_radii_scaled[candidate.ring_index]);
    }

    cv::Point2d center_local;
    std::vector<double> fitted_radii;
    if (!JointCircleFit(
            groups,
            scale_x,
            scale_y,
            rect_width,
            rect_height,
            &center_local,
            &fitted_radii
        )) {
        return -4;
    }

    if (check_ring_radius != 0) {
        for (std::size_t ring = 0; ring < fitted_radii.size(); ++ring) {
            if (std::abs(fitted_radii[ring] - expected_selected_radii[ring]) > max_radius_error * scale) {
                return -5;
            }
        }
    }

    const cv::Point2d center_full(
        (center_local.x + 0.5) / scale_x - 0.5,
        (center_local.y + 0.5) / scale_y - 0.5
    );
    const double rms = CalculateFitRms(groups, center_local, fitted_radii) / scale;

    out_center_xy[0] = static_cast<float>(center_full.x);
    out_center_xy[1] = static_cast<float>(center_full.y);
    out_rms[0] = static_cast<float>(rms);
    *out_status_code = rms > fit_rms_warning_threshold ? 1 : 0;

    const int radii_count = std::min(static_cast<int>(fitted_radii.size()), ring_count);
    for (int ring = 0; ring < radii_count; ++ring) {
        out_radii[ring] = static_cast<float>(fitted_radii[ring] / scale);
    }
    *out_radii_count = radii_count;
    return 0;
}

int32_t td_npu_stage_supported() {
    return 0;
}

}  // extern "C"

