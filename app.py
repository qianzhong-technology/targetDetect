"""MaixCam 相机循环与应用流程编排。"""

import atexit
import signal
import time

import cv2
import numpy as np
from maix import app, camera, display, err, gpio, image, pinmap

from common.config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CORNER_TRACK_SCALE,
    DISPLAY_UPDATE_INTERVAL,
    FAR_TARGET_AREA_RATIO,
    GIMBAL_CENTER_X,
    GIMBAL_CENTER_Y,
    GIMBAL_DEADBAND_PX,
    GIMBAL_DERIVATIVE_FILTER_ALPHA,
    GIMBAL_ENABLED,
    GIMBAL_INITIAL_PAN_DEG,
    GIMBAL_INITIAL_TILT_DEG,
    GIMBAL_IMU_ATTITUDE_DEADBAND_DEG,
    GIMBAL_IMU_ATTITUDE_KD,
    GIMBAL_IMU_ATTITUDE_KP,
    GIMBAL_IMU_ATTITUDE_OUTPUT_LIMIT_DEG,
    GIMBAL_IMU_ATTITUDE_VELOCITY_DEADBAND_DPS,
    GIMBAL_IMU_CALIBRATION_SAMPLES,
    GIMBAL_IMU_ENABLED,
    GIMBAL_IMU_FORCE_CALIBRATE,
    GIMBAL_IMU_MAHONY_KI,
    GIMBAL_IMU_MAHONY_KP,
    GIMBAL_IMU_PAN_AXIS,
    GIMBAL_IMU_PAN_SIGN,
    GIMBAL_IMU_TILT_AXIS,
    GIMBAL_IMU_TILT_SIGN,
    GIMBAL_INTEGRAL_LIMIT_DEG,
    GIMBAL_MAX_DT,
    GIMBAL_OUTPUT_DEADBAND_DEG,
    GIMBAL_OUTPUT_LIMIT_DEG,
    GIMBAL_PAN_DIRECTION,
    GIMBAL_PAN_KD,
    GIMBAL_PAN_KI,
    GIMBAL_PAN_KP,
    GIMBAL_PAN_LIMIT_DEG,
    GIMBAL_PRINT_INTERVAL,
    GIMBAL_TILT_DIRECTION,
    GIMBAL_TILT_KD,
    GIMBAL_TILT_KI,
    GIMBAL_TILT_KP,
    GIMBAL_TILT_LIMIT_DEG,
    TARGET_LOCK_CONFIRM_FRAMES,
    TARGET_LOSS_CONFIRM_FRAMES,
    TIMING_PRINT_INTERVAL,
    UART_SEND_INTERVAL,
)

from control.camera_attitude import CameraAttitude
from common.console_output import print_status
from common.target_lock import TargetLockHysteresis
from control.gimbal_pid import TwoAxisGimbalPID
from vision.kalman import TargetKalman
from vision.lighting import update_fill_light
from vision.target_center import (
    detect_raw_target_center,
    detect_target_center,
)
from control.serial_sender import TargetSerialSender
from vision.target_frame import (
    TargetCornerTracker,
    detect_target_corners,
    get_tracking_roi,
    make_homography,
    rectify_image,
    rectified_to_camera,
    target_frame_center_to_camera,
)


def draw_realtime_status(
    vis,
    target_status,
    target_status_color,
    target_point,
    frame_status,
    frame_status_color,
    fill_light_on,
    fps,
    fit_rms,
    fit_status
):
    height, width = vis.shape[:2]

    cv2.rectangle(
        vis,
        (8, 8),
        (width - 9, 58),
        (25, 25, 25),
        -1
    )
    cv2.circle(vis, (24, 33), 7, target_status_color, -1)
    cv2.putText(
        vis,
        target_status,
        (38, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2
    )

    if target_point is not None:
        tx, ty = target_point
        cv2.putText(
            vis,
            "X {:.1f}  Y {:.1f}".format(tx, ty),
            (width - 210, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1
        )

    cv2.rectangle(
        vis,
        (8, height - 42),
        (width - 9, height - 9),
        (25, 25, 25),
        -1
    )

    cv2.circle(
        vis,
        (24, height - 25),
        5,
        frame_status_color,
        -1
    )
    cv2.putText(
        vis,
        frame_status,
        (36, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    light_text = "LIGHT ON" if fill_light_on else "LIGHT OFF"
    light_color = (0, 255, 255) if fill_light_on else (180, 180, 180)
    cv2.putText(
        vis,
        light_text,
        (width - 250, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        light_color,
        1
    )

    cv2.putText(
        vis,
        "FPS {:.1f}".format(fps),
        (width - 105, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    rms_text = (
        "RMS {:.2f}px".format(fit_rms)
        if fit_rms is not None
        else "RMS --"
    )
    rms_color = (
        (0, 255, 0)
        if fit_status == "OK"
        else (0, 165, 255)
        if fit_rms is not None
        else (180, 180, 180)
    )
    cv2.putText(
        vis,
        rms_text,
        (width - 360, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        rms_color,
        1
    )


def main():

    disp = display.Display()
    err.check_raise(
        pinmap.set_pin_function("B3", "GPIOB3"),
        "set fill light pin failed"
    )
    fill_light_gpio = gpio.GPIO("GPIOB3", gpio.Mode.OUT)
    fill_light_gpio.value(0)

    def stop_program(signum, frame):
        del signum, frame
        fill_light_gpio.value(0)
        serial_sender.close()
        raise SystemExit(0)

    atexit.register(fill_light_gpio.value, 0)
    serial_sender = TargetSerialSender()
    atexit.register(serial_sender.close)
    signal.signal(signal.SIGINT, stop_program)
    signal.signal(signal.SIGTERM, stop_program)

    fill_light_on = False

    # 使用 BGR888，方便直接交给 OpenCV
    cam = camera.Camera(
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        image.Format.FMT_BGR888,
        fps=CAMERA_FPS
    )

    cam.skip_frames(20)

    kalman = TargetKalman()
    gimbal_controller = None
    if GIMBAL_ENABLED:
        gimbal_controller = TwoAxisGimbalPID(
            frame_width=CAMERA_WIDTH,
            frame_height=CAMERA_HEIGHT,
            center_x=GIMBAL_CENTER_X,
            center_y=GIMBAL_CENTER_Y,
            pan_kp=GIMBAL_PAN_KP,
            pan_ki=GIMBAL_PAN_KI,
            pan_kd=GIMBAL_PAN_KD,
            tilt_kp=GIMBAL_TILT_KP,
            tilt_ki=GIMBAL_TILT_KI,
            tilt_kd=GIMBAL_TILT_KD,
            deadband_px=GIMBAL_DEADBAND_PX,
            output_deadband_deg=GIMBAL_OUTPUT_DEADBAND_DEG,
            integral_limit_deg=GIMBAL_INTEGRAL_LIMIT_DEG,
            output_limit_deg=GIMBAL_OUTPUT_LIMIT_DEG,
            derivative_filter_alpha=GIMBAL_DERIVATIVE_FILTER_ALPHA,
            max_dt=GIMBAL_MAX_DT,
            initial_pan_deg=GIMBAL_INITIAL_PAN_DEG,
            initial_tilt_deg=GIMBAL_INITIAL_TILT_DEG,
            pan_limit_deg=GIMBAL_PAN_LIMIT_DEG,
            tilt_limit_deg=GIMBAL_TILT_LIMIT_DEG,
            print_interval=GIMBAL_PRINT_INTERVAL,
            pan_direction=GIMBAL_PAN_DIRECTION,
            tilt_direction=GIMBAL_TILT_DIRECTION,
            attitude_kp=GIMBAL_IMU_ATTITUDE_KP,
            attitude_kd=GIMBAL_IMU_ATTITUDE_KD,
            attitude_deadband_deg=(
                GIMBAL_IMU_ATTITUDE_DEADBAND_DEG
            ),
            attitude_velocity_deadband_dps=(
                GIMBAL_IMU_ATTITUDE_VELOCITY_DEADBAND_DPS
            ),
            attitude_output_limit_deg=(
                GIMBAL_IMU_ATTITUDE_OUTPUT_LIMIT_DEG
            )
        )
    camera_attitude = None
    if GIMBAL_IMU_ENABLED and gimbal_controller is not None:
        camera_attitude = CameraAttitude(
            pan_axis=GIMBAL_IMU_PAN_AXIS,
            tilt_axis=GIMBAL_IMU_TILT_AXIS,
            pan_sign=GIMBAL_IMU_PAN_SIGN,
            tilt_sign=GIMBAL_IMU_TILT_SIGN,
            mahony_kp=GIMBAL_IMU_MAHONY_KP,
            mahony_ki=GIMBAL_IMU_MAHONY_KI,
            force_calibrate=GIMBAL_IMU_FORCE_CALIBRATE,
            calibration_samples=(
                GIMBAL_IMU_CALIBRATION_SAMPLES
            )
        )

    last_time = time.monotonic()
    last_timing_print = time.monotonic()
    last_uart_send = time.monotonic()

    frame_count = 0
    tracking_roi = (0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)
    corner_tracker = TargetCornerTracker()
    target_lock = TargetLockHysteresis(
        lock_confirm_frames=TARGET_LOCK_CONFIRM_FRAMES,
        loss_confirm_frames=TARGET_LOSS_CONFIRM_FRAMES
    )

    print_status(
        "APP",
        "STARTED",
        camera="640x480",
        target_tracking="ready"
    )

    while not app.need_exit():

        loop_start = time.monotonic()
        module_times = {
            "camera_io": 0.0,
            "lighting": 0.0,
            "target_frame": 0.0,
            "target_center": 0.0,
            "kalman": 0.0,
            "imu": 0.0,
            "gimbal": 0.0,
            "annotations": 0.0,
            "status_overlay": 0.0,
            "uart": 0.0,
            "display_convert": 0.0,
            "display_show": 0.0,
        }

        # ----------------------------------------------------
        # 读取图像
        # ----------------------------------------------------

        stage_start = time.monotonic()
        img = cam.read()

        # 不进行额外颜色转换和复制
        frame = image.image2cv(
            img,
            ensure_bgr=False,
            copy=False
        )
        module_times["camera_io"] += time.monotonic() - stage_start

        stage_start = time.monotonic()
        fill_light_on, _ = update_fill_light(
            frame,
            fill_light_gpio,
            fill_light_on
        )
        module_times["lighting"] += time.monotonic() - stage_start

        # ----------------------------------------------------
        # 检测黑色矩形边框
        # ----------------------------------------------------

        stage_start = time.monotonic()
        corners = None
        corners_tracked = False
        roi_x1, roi_y1, roi_x2, roi_y2 = tracking_roi
        detect_corners = corner_tracker.needs_redetection(frame_count)
        corner_mode = "DETECT" if detect_corners else "TRACK"
        prediction_anchor = (
            detect_raw_target_center(frame)
            if corner_tracker.is_predicted
            else None
        )
        detect_roi = corner_tracker.get_redetection_roi(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            center_anchor=prediction_anchor
        ) if corner_tracker.is_predicted else (
            roi_x1,
            roi_y1,
            roi_x2,
            roi_y2
        )
        detect_x1, detect_y1, detect_x2, detect_y2 = detect_roi

        if detect_corners:
            roi_frame = frame[
                detect_y1:detect_y2,
                detect_x1:detect_x2
            ]
            reference_corners = (
                corner_tracker.corners.copy() / CORNER_TRACK_SCALE
                if (
                    corner_tracker.corners is not None
                    and not corner_tracker.is_predicted
                )
                else None
            )
            if reference_corners is not None:
                reference_corners = reference_corners.copy()
                reference_corners[:, 0] -= detect_x1
                reference_corners[:, 1] -= detect_y1
            corners, _ = detect_target_corners(
                roi_frame,
                reference_corners
            )
            if corners is not None:
                corners[:, 0] += detect_x1
                corners[:, 1] += detect_y1

        if corners is not None:
            accepted_corners = corner_tracker.initialize(
                frame,
                corners,
                force=corner_tracker.is_predicted,
                center_anchor=prediction_anchor
            )
            if accepted_corners is not None:
                corners = accepted_corners
            elif corner_tracker.is_active:
                corners = corner_tracker.update(
                    frame,
                    center_anchor=prediction_anchor
                )
                corners_tracked = corners is not None
                corner_mode = "DETECT+TRACK"
                if corner_tracker.is_predicted:
                    corner_mode = "DETECT+PREDICT"
        else:
            if detect_corners and not corner_tracker.is_predicted:
                tracking_roi = (0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)
            if corner_tracker.is_active:
                corners = corner_tracker.update(
                    frame,
                    center_anchor=prediction_anchor
                )
                corners_tracked = corners is not None
                predicted = corner_tracker.is_predicted
                if detect_corners:
                    corner_mode = (
                        "DETECT+PREDICT"
                        if predicted
                        else "DETECT+TRACK"
                    )
                elif predicted:
                    corner_mode = "PREDICT"

        module_times["target_frame"] += time.monotonic() - stage_start
        if corners is None:
            corner_mode = "LOST"
            tracking_roi = (0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)

        center_camera = None
        center_rectified = None
        fit_rms = None
        fit_status = "frame not found"

        if corners is not None:

            # ------------------------------------------------
            # 透视校正
            # ------------------------------------------------

            stage_start = time.monotonic()
            H = make_homography(
                corners
            )

            if corner_tracker.is_predicted:
                module_times["target_frame"] += time.monotonic() - stage_start
                center_camera = target_frame_center_to_camera(H)
                fit_status = "frame center fallback"
            else:
                rectified = rectify_image(
                    frame,
                    H
                )
                module_times["target_frame"] += time.monotonic() - stage_start

                target_area_ratio = (
                    abs(cv2.contourArea(corners.astype(np.float32)))
                    / float(CAMERA_WIDTH * CAMERA_HEIGHT)
                )

                stage_start = time.monotonic()
                (
                    center_rectified,
                    fit_rms,
                    _,
                    _,
                    fit_status,
                    _
                ) = detect_target_center(
                    rectified,
                    distant=target_area_ratio <= FAR_TARGET_AREA_RATIO
                )
                module_times["target_center"] += (
                    time.monotonic() - stage_start
                )

                if center_rectified is not None:
                    stage_start = time.monotonic()
                    center_camera = rectified_to_camera(
                        center_rectified,
                        H
                    )
                    module_times["target_frame"] += (
                        time.monotonic() - stage_start
                    )
                else:
                    center_camera = target_frame_center_to_camera(H)
                    fit_status = "frame center fallback"

        if corners is not None:
            tracking_center = (
                center_camera
                if center_camera is not None and fit_status == "OK"
                else np.mean(corners, axis=0)
            )
            stage_start = time.monotonic()
            tracking_roi = get_tracking_roi(
                corners,
                tracking_center,
                CAMERA_WIDTH,
                CAMERA_HEIGHT
            )
            module_times["target_frame"] += time.monotonic() - stage_start

        # ----------------------------------------------------
        # 卡尔曼滤波
        # ----------------------------------------------------

        now = time.monotonic()

        dt = now - last_time
        last_time = now
        target_status = "TARGET NOT FOUND"
        target_status_color = (0, 0, 255)
        target_point = None

        measured_target_locked = (
            center_camera is not None
            and fit_status == "OK"
        )
        stable_target_locked, stable_target_point = target_lock.update(
            measured_target_locked,
            center_camera
            if measured_target_locked
            else None
        )
        serial_target_point = (
            stable_target_point
            if stable_target_locked
            else None
        )

        if stable_target_locked and stable_target_point is not None:
            if measured_target_locked:
                stage_start = time.monotonic()
                kalman.update(
                    stable_target_point,
                    dt
                )
                module_times["kalman"] += time.monotonic() - stage_start
            target_status = "TARGET LOCKED"
            target_status_color = (0, 255, 0)
            target_point = stable_target_point

        elif measured_target_locked:
            target_status = "TARGET ACQUIRING"
            target_status_color = (0, 165, 255)
            target_point = center_camera

        elif center_camera is not None:
            target_status = (
                "FRAME CENTER"
                if fit_status == "frame center fallback"
                else "LOW CONF"
            )
            target_status_color = (0, 165, 255)
            target_point = (center_camera[0], center_camera[1])

        elif corners is not None and kalman.initialized:
            target_status = "PREDICTING"
            target_status_color = (0, 165, 255)
            stage_start = time.monotonic()
            filtered_center = kalman.predict(dt)
            module_times["kalman"] += time.monotonic() - stage_start
            if filtered_center is not None:
                fx, fy = filtered_center
                target_status = "PREDICTING"
                target_status_color = (0, 165, 255)
                target_point = (fx, fy)
        else:
            stage_start = time.monotonic()
            kalman.reset()
            module_times["kalman"] += time.monotonic() - stage_start

        if gimbal_controller is not None:
            attitude_feedback = None
            if camera_attitude is not None:
                stage_start = time.monotonic()
                attitude_reading = camera_attitude.read(
                    now=stage_start
                )
                module_times["imu"] += time.monotonic() - stage_start
                if attitude_reading.get("valid", False):
                    attitude_feedback = attitude_reading

            stage_start = time.monotonic()
            if (
                target_status == "TARGET LOCKED"
                and target_point is not None
            ):
                gimbal_controller.update(
                    target_point,
                    dt,
                    attitude_feedback=attitude_feedback
                )
            else:
                gimbal_controller.mark_target_lost()
            module_times["gimbal"] += time.monotonic() - stage_start

        # ----------------------------------------------------
        # FPS 显示
        # ----------------------------------------------------

        frame_count += 1

        elapsed = time.monotonic() - loop_start
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        vis = frame
        display_updated = (
            (frame_count - 1) % DISPLAY_UPDATE_INTERVAL == 0
        )
        if display_updated:
            if corner_tracker.is_predicted:
                frame_status = "FRAME PREDICT"
                frame_color = (0, 165, 255)
            elif corners_tracked:
                frame_status = "FRAME TRACKED"
                frame_color = (0, 165, 255)
            elif corners is not None:
                frame_status = "FRAME LOCKED"
                frame_color = (0, 255, 0)
            else:
                frame_status = "FRAME LOST"
                frame_color = (0, 0, 255)

            stage_start = time.monotonic()
            if corners is not None:
                for point in corners:
                    cv2.circle(
                        vis,
                        (int(point[0]), int(point[1])),
                        5,
                        (255, 0, 0),
                        -1
                    )
                cv2.polylines(
                    vis,
                    [corners.astype(np.int32)],
                    True,
                    (255, 0, 0),
                    2
                )

            if target_point is not None:
                marker_radius = 7 if center_camera is not None else 4
                cv2.circle(
                    vis,
                    (int(target_point[0]), int(target_point[1])),
                    marker_radius,
                    target_status_color,
                    2 if center_camera is not None else -1
                )
            module_times["annotations"] += time.monotonic() - stage_start

            stage_start = time.monotonic()
            draw_realtime_status(
                vis,
                target_status,
                target_status_color,
                target_point,
                frame_status,
                frame_color,
                fill_light_on,
                fps,
                fit_rms,
                fit_status
            )
            module_times["status_overlay"] += time.monotonic() - stage_start

        now_uart = time.monotonic()
        if now_uart - last_uart_send >= UART_SEND_INTERVAL:
            stage_start = time.monotonic()
            serial_sender.send_frame(
                center_camera=serial_target_point
            )
            module_times["uart"] += time.monotonic() - stage_start
            last_uart_send = now_uart

        # ----------------------------------------------------
        # 显示图像
        # ----------------------------------------------------

        if display_updated:
            stage_start = time.monotonic()
            show_img = image.cv2image(
                vis,
                bgr=True,
                copy=False
            )
            module_times["display_convert"] += time.monotonic() - stage_start

            stage_start = time.monotonic()
            disp.show(show_img)
            module_times["display_show"] += time.monotonic() - stage_start

        total_seconds = time.monotonic() - loop_start
        if time.monotonic() - last_timing_print >= TIMING_PRINT_INTERVAL:
            module_times["app_and_render"] = max(
                0.0,
                total_seconds - sum(module_times.values())
            )
            timing_report = {
                "frame": frame_count,
                "corner_mode": corner_mode,
                "fit_status": fit_status,
                "fit_rms": (
                    round(float(fit_rms), 3)
                    if fit_rms is not None
                    else None
                ),
                "display_updated": display_updated,
                "corner_features": (
                    len(corner_tracker.previous_points)
                    if corner_tracker.previous_points is not None
                    else 0
                ),
                "modules_ms": {
                    name: round(elapsed * 1000.0, 3)
                    for name, elapsed in module_times.items()
                },
                "total_ms": round(total_seconds * 1000.0, 3),
                "fps": round(1.0 / total_seconds, 2) if total_seconds > 0 else 0.0,
            }
            print_status(
                "TIMING",
                "SNAPSHOT",
                frame=frame_count,
                corner_mode=corner_mode,
                fit_status=fit_status,
                fit_rms=fit_rms,
                display_updated=display_updated,
                corner_features=timing_report["corner_features"],
                camera_io_ms=timing_report["modules_ms"]["camera_io"],
                target_frame_ms=timing_report["modules_ms"]["target_frame"],
                target_center_ms=timing_report["modules_ms"]["target_center"],
                imu_ms=timing_report["modules_ms"]["imu"],
                gimbal_ms=timing_report["modules_ms"]["gimbal"],
                annotations_ms=timing_report["modules_ms"]["annotations"],
                status_overlay_ms=timing_report["modules_ms"]["status_overlay"],
                display_ms=timing_report["modules_ms"]["display_show"],
                total_ms=timing_report["total_ms"],
                fps=timing_report["fps"]
            )
            last_timing_print = time.monotonic()

    fill_light_gpio.value(0)
    serial_sender.close()
