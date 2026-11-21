"""二轴舵机云台的闭环 PID 控制与方向打印。"""

import math
import time
from typing import Optional, Tuple

from common.console_output import print_status


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class _AxisPid:
    """单轴位置式 PID 控制器。"""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_limit_deg: float,
        output_limit_deg: float,
        output_deadband_deg: float,
        derivative_filter_alpha: float,
        max_dt: float
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit_deg = integral_limit_deg
        self.output_limit_deg = output_limit_deg
        self.output_deadband_deg = output_deadband_deg
        self.derivative_filter_alpha = _clamp(
            derivative_filter_alpha,
            0.0,
            1.0
        )
        self.max_dt = max(0.001, max_dt)
        self.integral_term_deg = 0.0
        self.previous_error_px: Optional[float] = None
        self.filtered_derivative = 0.0

    def reset(self):
        self.integral_term_deg = 0.0
        self.previous_error_px = None
        self.filtered_derivative = 0.0

    def compute(
        self,
        error_px: float,
        dt: float,
        deadband_px: float,
        at_lower_limit: bool = False,
        at_upper_limit: bool = False
    ) -> float:
        if not math.isfinite(error_px):
            return 0.0

        if abs(error_px) <= deadband_px:
            self.integral_term_deg = 0.0
            self.previous_error_px = error_px
            self.filtered_derivative = 0.0
            return 0.0

        try:
            dt_value = float(dt)
        except (TypeError, ValueError):
            dt_value = 0.01
        if not math.isfinite(dt_value):
            dt_value = 0.01
        dt = _clamp(
            abs(dt_value),
            0.001,
            self.max_dt
        )

        proportional_term = self.kp * error_px

        if self.previous_error_px is None:
            raw_derivative = 0.0
        else:
            raw_derivative = (
                error_px - self.previous_error_px
            ) / dt

        self.filtered_derivative = (
            self.derivative_filter_alpha * raw_derivative
            + (1.0 - self.derivative_filter_alpha)
            * self.filtered_derivative
        )
        derivative_term = self.kd * self.filtered_derivative
        self.previous_error_px = error_px

        moving_toward_limit = (
            at_lower_limit and error_px < 0.0
        ) or (
            at_upper_limit and error_px > 0.0
        )
        if moving_toward_limit:
            self.integral_term_deg = 0.0
            candidate_integral = 0.0
            self.filtered_derivative = 0.0
        else:
            candidate_integral = _clamp(
                self.integral_term_deg
                + self.ki * error_px * dt,
                -self.integral_limit_deg,
                self.integral_limit_deg
            )

        raw_output = (
            proportional_term
            + candidate_integral
            + derivative_term
        )
        clamped_output = _clamp(
            raw_output,
            -self.output_limit_deg,
            self.output_limit_deg
        )

        # 饱和且误差仍指向同一方向时，不继续累积积分项。
        if (
            not moving_toward_limit
            and (
                abs(raw_output) <= self.output_limit_deg
                or raw_output * error_px <= 0.0
            )
        ):
            self.integral_term_deg = candidate_integral

        if abs(clamped_output) < self.output_deadband_deg:
            return 0.0

        return clamped_output


class TwoAxisGimbalPID:
    """云台水平、俯仰两轴的闭环 PID 控制器。"""

    def __init__(
        self,
        frame_width: float = 640.0,
        frame_height: float = 480.0,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        pan_kp: float = 0.035,
        pan_ki: float = 0.002,
        pan_kd: float = 0.012,
        tilt_kp: float = 0.035,
        tilt_ki: float = 0.002,
        tilt_kd: float = 0.012,
        deadband_px: float = 5.0,
        output_deadband_deg: float = 0.15,
        integral_limit_deg: float = 3.0,
        output_limit_deg: float = 5.0,
        derivative_filter_alpha: float = 0.30,
        max_dt: float = 0.25,
        initial_pan_deg: float = 90.0,
        initial_tilt_deg: float = 90.0,
        pan_limit_deg: Tuple[float, float] = (0.0, 180.0),
        tilt_limit_deg: Tuple[float, float] = (0.0, 180.0),
        print_interval: float = 0.25,
        pan_direction: int = 1,
        tilt_direction: int = 1,
        attitude_kp: float = 0.60,
        attitude_kd: float = 0.08,
        attitude_deadband_deg: float = 1.0,
        attitude_velocity_deadband_dps: float = 2.0,
        attitude_output_limit_deg: float = 5.0
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = (
            frame_width / 2.0
            if center_x is None
            else float(center_x)
        )
        self.center_y = (
            frame_height / 2.0
            if center_y is None
            else float(center_y)
        )
        self.deadband_px = max(0.0, deadband_px)
        self.print_interval = max(0.0, print_interval)
        self.pan_direction = 1 if pan_direction >= 0 else -1
        self.tilt_direction = 1 if tilt_direction >= 0 else -1
        self.pan_limit_deg = (
            float(pan_limit_deg[0]),
            float(pan_limit_deg[1])
        )
        self.tilt_limit_deg = (
            float(tilt_limit_deg[0]),
            float(tilt_limit_deg[1])
        )
        self.pan_angle_deg = _clamp(
            initial_pan_deg,
            *self.pan_limit_deg
        )
        self.tilt_angle_deg = _clamp(
            initial_tilt_deg,
            *self.tilt_limit_deg
        )
        self.pan_pid = _AxisPid(
            pan_kp,
            pan_ki,
            pan_kd,
            integral_limit_deg,
            output_limit_deg,
            output_deadband_deg,
            derivative_filter_alpha,
            max_dt
        )
        self.tilt_pid = _AxisPid(
            tilt_kp,
            tilt_ki,
            tilt_kd,
            integral_limit_deg,
            output_limit_deg,
            output_deadband_deg,
            derivative_filter_alpha,
            max_dt
        )
        self.attitude_kp = max(0.0, attitude_kp)
        self.attitude_kd = max(0.0, attitude_kd)
        self.attitude_deadband_deg = max(0.0, attitude_deadband_deg)
        self.attitude_velocity_deadband_dps = max(
            0.0,
            attitude_velocity_deadband_dps
        )
        self.attitude_output_limit_deg = max(
            0.0,
            attitude_output_limit_deg
        )
        self.initial_pan_command_deg = self.pan_angle_deg
        self.initial_tilt_command_deg = self.tilt_angle_deg
        self.last_print_time = 0.0
        self.target_available = False
        self.last_lost_print = False

    def reset(self):
        self.pan_pid.reset()
        self.tilt_pid.reset()
        self.target_available = False
        self.last_lost_print = False

    def mark_target_lost(self):
        """靶心丢失时复位积分项并保持最后姿态，只打印一次。"""
        self.pan_pid.reset()
        self.tilt_pid.reset()
        if not self.last_lost_print:
            print_status(
                "GIMBAL",
                "TARGET_LOST",
                pan_deg=self.pan_angle_deg,
                tilt_deg=self.tilt_angle_deg
            )
            self.last_lost_print = True
        self.target_available = False

    def _format_pan_delta(
        self,
        delta_deg: float,
        error_px: float,
        pan_output_deg: float,
        at_lower_limit: bool,
        at_upper_limit: bool
    ):
        if at_lower_limit and pan_output_deg < 0.0:
            return "hold, pan lower limit {:.2f}deg".format(
                self.pan_limit_deg[0]
            )
        if at_upper_limit and pan_output_deg > 0.0:
            return "hold, pan upper limit {:.2f}deg".format(
                self.pan_limit_deg[1]
            )
        if abs(delta_deg) < 0.05:
            if abs(error_px) <= self.deadband_px:
                return "hold, inside deadband"
            return "hold, inside output deadband"
        if delta_deg > 0:
            name = "move_right" if self.pan_direction > 0 else "move_left"
        else:
            name = "move_left" if self.pan_direction > 0 else "move_right"
        return "{} {:.2f}deg".format(name, abs(delta_deg))

    def _format_tilt_delta(
        self,
        delta_deg: float,
        error_px: float,
        tilt_output_deg: float,
        at_lower_limit: bool,
        at_upper_limit: bool
    ):
        if at_lower_limit and tilt_output_deg < 0.0:
            return "hold, tilt lower limit {:.2f}deg".format(
                self.tilt_limit_deg[0]
            )
        if at_upper_limit and tilt_output_deg > 0.0:
            return "hold, tilt upper limit {:.2f}deg".format(
                self.tilt_limit_deg[1]
            )
        if abs(delta_deg) < 0.05:
            if abs(error_px) <= self.deadband_px:
                return "hold, inside deadband"
            return "hold, inside output deadband"
        if delta_deg > 0:
            name = "move_down" if self.tilt_direction > 0 else "move_up"
        else:
            name = "move_up" if self.tilt_direction > 0 else "move_down"
        return "{} {:.2f}deg".format(name, abs(delta_deg))

    def _attitude_inner_output(
        self,
        target_relative_deg: float,
        actual_relative_deg: float,
        angular_velocity_dps: float,
        at_lower_limit: bool,
        at_upper_limit: bool
    ):
        target_relative_deg = float(target_relative_deg)
        actual_relative_deg = float(actual_relative_deg)
        angular_velocity_dps = float(angular_velocity_dps)
        if not (
            math.isfinite(target_relative_deg)
            and math.isfinite(actual_relative_deg)
            and math.isfinite(angular_velocity_dps)
        ):
            return 0.0, target_relative_deg - actual_relative_deg, "feedback_invalid"

        attitude_error_deg = target_relative_deg - actual_relative_deg
        if abs(attitude_error_deg) <= self.attitude_deadband_deg:
            raw_output = -(
                self.attitude_kd * angular_velocity_dps
            )
            if abs(angular_velocity_dps) <= self.attitude_velocity_deadband_dps:
                status = "positioned"
                raw_output = 0.0
            else:
                status = "damping"
        else:
            raw_output = (
                self.attitude_kp * attitude_error_deg
                - self.attitude_kd * angular_velocity_dps
            )
            status = (
                "moving"
                if abs(angular_velocity_dps)
                >= self.attitude_velocity_deadband_dps
                else "settling"
            )

        if at_lower_limit and raw_output < 0.0:
            raw_output = 0.0
            status = "lower_limit"
        if at_upper_limit and raw_output > 0.0:
            raw_output = 0.0
            status = "upper_limit"

        output_deg = _clamp(
            raw_output,
            -self.attitude_output_limit_deg,
            self.attitude_output_limit_deg
        )
        return output_deg, attitude_error_deg, status

    def update(
        self,
        target_camera,
        dt: float,
        now: Optional[float] = None,
        attitude_feedback: Optional[dict] = None
    ):
        """根据相机坐标计算并打印下一步云台转动。"""
        if target_camera is None:
            self.mark_target_lost()
            return None

        try:
            target_x = float(target_camera[0])
            target_y = float(target_camera[1])
        except (TypeError, ValueError, IndexError):
            self.mark_target_lost()
            return None

        if not math.isfinite(target_x) or not math.isfinite(target_y):
            self.mark_target_lost()
            return None

        pan_error_px = target_x - self.center_x
        tilt_error_px = target_y - self.center_y

        at_pan_lower_limit = (
            self.pan_angle_deg
            <= self.pan_limit_deg[0] + 0.001
        )
        at_pan_upper_limit = (
            self.pan_angle_deg
            >= self.pan_limit_deg[1] - 0.001
        )
        at_tilt_lower_limit = (
            self.tilt_angle_deg
            <= self.tilt_limit_deg[0] + 0.001
        )
        at_tilt_upper_limit = (
            self.tilt_angle_deg
            >= self.tilt_limit_deg[1] - 0.001
        )

        pan_output_deg = self.pan_pid.compute(
            pan_error_px * self.pan_direction,
            dt,
            self.deadband_px,
            at_pan_lower_limit,
            at_pan_upper_limit
        )
        tilt_output_deg = self.tilt_pid.compute(
            tilt_error_px * self.tilt_direction,
            dt,
            self.deadband_px,
            at_tilt_lower_limit,
            at_tilt_upper_limit
        )

        target_pan_angle = _clamp(
            self.pan_angle_deg + pan_output_deg,
            *self.pan_limit_deg
        )
        target_tilt_angle = _clamp(
            self.tilt_angle_deg + tilt_output_deg,
            *self.tilt_limit_deg
        )

        visual_pan_delta_deg = target_pan_angle - self.pan_angle_deg
        visual_tilt_delta_deg = target_tilt_angle - self.tilt_angle_deg

        pan_delta_deg = visual_pan_delta_deg
        tilt_delta_deg = visual_tilt_delta_deg
        pan_attitude_error_deg = None
        tilt_attitude_error_deg = None
        pan_actual_relative_deg = None
        tilt_actual_relative_deg = None
        pan_attitude_status = "unavailable"
        tilt_attitude_status = "unavailable"
        pan_inner_output_deg = 0.0
        tilt_inner_output_deg = 0.0

        if attitude_feedback is not None:
            try:
                pan_actual_relative_deg = float(
                    attitude_feedback["pan_relative_deg"]
                )
                tilt_actual_relative_deg = float(
                    attitude_feedback["tilt_relative_deg"]
                )
            except (KeyError, TypeError, ValueError):
                pan_actual_relative_deg = None
                tilt_actual_relative_deg = None

            pan_velocity_dps = 0.0
            tilt_velocity_dps = 0.0
            if pan_actual_relative_deg is not None:
                try:
                    pan_velocity_dps = float(
                        attitude_feedback.get("pan_velocity_dps", 0.0)
                    )
                    if not math.isfinite(pan_velocity_dps):
                        pan_velocity_dps = 0.0
                except (TypeError, ValueError):
                    pan_velocity_dps = 0.0

                target_pan_relative_deg = (
                    target_pan_angle - self.initial_pan_command_deg
                )
                (
                    pan_inner_output_deg,
                    pan_attitude_error_deg,
                    pan_attitude_status
                ) = self._attitude_inner_output(
                    target_pan_relative_deg,
                    pan_actual_relative_deg,
                    pan_velocity_dps,
                    at_pan_lower_limit,
                    at_pan_upper_limit
                )
                pan_delta_deg = pan_inner_output_deg

            if tilt_actual_relative_deg is not None:
                try:
                    tilt_velocity_dps = float(
                        attitude_feedback.get("tilt_velocity_dps", 0.0)
                    )
                    if not math.isfinite(tilt_velocity_dps):
                        tilt_velocity_dps = 0.0
                except (TypeError, ValueError):
                    tilt_velocity_dps = 0.0

                target_tilt_relative_deg = (
                    target_tilt_angle - self.initial_tilt_command_deg
                )
                (
                    tilt_inner_output_deg,
                    tilt_attitude_error_deg,
                    tilt_attitude_status
                ) = self._attitude_inner_output(
                    target_tilt_relative_deg,
                    tilt_actual_relative_deg,
                    tilt_velocity_dps,
                    at_tilt_lower_limit,
                    at_tilt_upper_limit
                )
                tilt_delta_deg = tilt_inner_output_deg

        self.pan_angle_deg = target_pan_angle
        self.tilt_angle_deg = target_tilt_angle
        self.target_available = True
        self.last_lost_print = False

        if now is None:
            now = time.monotonic()
        if now - self.last_print_time >= self.print_interval:
            if pan_attitude_error_deg is None:
                pan_attitude_status_value = "unavailable"
                pan_attitude_error_text = None
                pan_actual_text = None
            else:
                pan_attitude_status_value = pan_attitude_status
                pan_attitude_error_text = pan_attitude_error_deg
                pan_actual_text = pan_actual_relative_deg
            if tilt_attitude_error_deg is None:
                tilt_attitude_status_value = "unavailable"
                tilt_attitude_error_text = None
                tilt_actual_text = None
            else:
                tilt_attitude_status_value = tilt_attitude_status
                tilt_attitude_error_text = tilt_attitude_error_deg
                tilt_actual_text = tilt_actual_relative_deg

            print_status(
                "GIMBAL",
                "TRACKING",
                pan_error_px=pan_error_px,
                pan_action=self._format_pan_delta(
                    pan_delta_deg,
                    pan_error_px,
                    pan_delta_deg,
                    at_pan_lower_limit,
                    at_pan_upper_limit
                ),
                tilt_error_px=tilt_error_px,
                tilt_action=self._format_tilt_delta(
                    tilt_delta_deg,
                    tilt_error_px,
                    tilt_delta_deg,
                    at_tilt_lower_limit,
                    at_tilt_upper_limit
                ),
                pan_attitude_status=pan_attitude_status_value,
                pan_attitude_error_deg=pan_attitude_error_text,
                pan_actual_deg=pan_actual_text,
                tilt_attitude_status=tilt_attitude_status_value,
                tilt_attitude_error_deg=tilt_attitude_error_text,
                tilt_actual_deg=tilt_actual_text,
                pan_deg=self.pan_angle_deg,
                tilt_deg=self.tilt_angle_deg
            )
            self.last_print_time = now

        return {
            "pan_error_px": pan_error_px,
            "tilt_error_px": tilt_error_px,
            "pan_output_deg": pan_output_deg,
            "tilt_output_deg": tilt_output_deg,
            "pan_delta_deg": pan_delta_deg,
            "tilt_delta_deg": tilt_delta_deg,
            "visual_pan_delta_deg": visual_pan_delta_deg,
            "visual_tilt_delta_deg": visual_tilt_delta_deg,
            "pan_inner_output_deg": pan_inner_output_deg,
            "tilt_inner_output_deg": tilt_inner_output_deg,
            "pan_attitude_error_deg": pan_attitude_error_deg,
            "tilt_attitude_error_deg": tilt_attitude_error_deg,
            "pan_actual_relative_deg": pan_actual_relative_deg,
            "tilt_actual_relative_deg": tilt_actual_relative_deg,
            "pan_attitude_status": pan_attitude_status,
            "tilt_attitude_status": tilt_attitude_status,
            "pan_angle_deg": self.pan_angle_deg,
            "tilt_angle_deg": self.tilt_angle_deg,
        }
