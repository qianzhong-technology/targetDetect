"""MaixCAM-Pro 内置 QMI8658 姿态读取与欧拉角反馈。"""

import math
import time
import types
from typing import Any, Callable, Optional

from maix import ahrs
from maix.ext_dev import imu  # pyright: ignore[reportMissingImports]

from common.console_output import print_status


def _shortest_angle_delta(current_deg: float, reference_deg: float) -> float:
    return (
        (current_deg - reference_deg + 180.0) % 360.0
    ) - 180.0


def _get_callable(instance: Any, name: str) -> Optional[Callable[..., Any]]:
    method = getattr(instance, name, None)
    return method if callable(method) else None


class CameraAttitude:
    """读取板载 IMU, 并输出相对于首次姿态的水平、俯仰反馈。"""

    def __init__(
        self,
        pan_axis: str = "yaw",
        tilt_axis: str = "pitch",
        pan_sign: int = 1,
        tilt_sign: int = 1,
        mahony_kp: float = 2.0,
        mahony_ki: float = 0.01,
        force_calibrate: bool = False,
        calibration_samples: int = 10000
    ):
        self.pan_axis = pan_axis.lower()
        self.tilt_axis = tilt_axis.lower()
        self.pan_sign = 1 if pan_sign >= 0 else -1
        self.tilt_sign = 1 if tilt_sign >= 0 else -1
        self.mahony_kp = float(mahony_kp)
        self.mahony_ki = float(mahony_ki)
        self.force_calibrate = force_calibrate
        self.calibration_samples = max(1000, int(calibration_samples))
        self.sensor: Optional[Any] = None
        self.ahrs_filter: Optional[Any] = None
        self.available = False
        self.initialization_error = None
        self.status = "INITIALIZING"
        self.first_valid_read_printed = False
        self.first_read_error_printed = False
        self.last_read_time = time.monotonic()
        self.initial_pan_raw_deg: Optional[float] = None
        self.initial_tilt_raw_deg: Optional[float] = None
        self.pan_relative_deg = 0.0
        self.tilt_relative_deg = 0.0

        try:
            print_status(
                "IMU",
                "INITIALIZING",
                step="loading_maix_imu"
            )
            imu_factory = _get_callable(imu, "IMU")
            if imu_factory is None:
                raise AttributeError("IMU factory is unavailable")

            self.sensor = imu_factory(
                "default",
                mode=imu.Mode.DUAL,
                acc_scale=imu.AccScale.ACC_SCALE_2G,
                acc_odr=imu.AccOdr.ACC_ODR_1000,
                gyro_scale=imu.GyroScale.GYRO_SCALE_256DPS,
                gyro_odr=imu.GyroOdr.GYRO_ODR_8000
            )
            if self.sensor is None:
                raise RuntimeError("IMU initialization returned no sensor")
            mahony_factory = _get_callable(ahrs, "MahonyAHRS")
            if mahony_factory is None:
                raise AttributeError("MahonyAHRS is unavailable")
            self.ahrs_filter = mahony_factory(
                self.mahony_kp,
                self.mahony_ki
            )
            if self.ahrs_filter is None:
                raise RuntimeError("MahonyAHRS initialization failed")
            calibration_exists = False
            calibration_exists_method = _get_callable(
                self.sensor,
                "calib_gyro_exists"
            )
            if calibration_exists_method is not None:
                try:
                    calibration_exists = bool(
                        calibration_exists_method()
                    )
                except (
                    AttributeError,
                    RuntimeError,
                    OSError,
                    NotImplementedError
                ):
                    calibration_exists = False

            if self.force_calibrate or not calibration_exists:
                print_status(
                    "IMU",
                    "CALIBRATING",
                    keep_camera_stationary=True,
                    estimated_seconds=10
                )
                calibrate_method = _get_callable(
                    self.sensor,
                    "calib_gyro"
                )
                if calibrate_method is None:
                    raise AttributeError(
                        "gyro calibration method is unavailable"
                    )
                calibrate_method(self.calibration_samples)
                calibration_status = "calibrated"
            else:
                load_calibration_method = _get_callable(
                    self.sensor,
                    "load_calib_gyro"
                )
                if load_calibration_method is not None:
                    load_calibration_method()
                calibration_status = "loaded"

            self.available = True
            self.status = "READY"
            print_status(
                "IMU",
                "READY",
                sensor="QMI8658",
                ahrs="MahonyAHRS",
                calibration=calibration_status
            )
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            OSError,
            TypeError,
            ValueError,
            NotImplementedError
        ) as error:
            self.initialization_error = str(error)
            self.available = False
            self.status = "UNAVAILABLE"
            print_status(
                "IMU",
                "UNAVAILABLE",
                fallback="visual_pid",
                reason=error
            )

    def reset_reference(self):
        """以当前姿态重新建立水平、俯仰相对参考。"""
        self.initial_pan_raw_deg = None
        self.initial_tilt_raw_deg = None
        self.pan_relative_deg = 0.0
        self.tilt_relative_deg = 0.0

    def _axis_angle(self, angle, axis_name):
        axis_name = axis_name.lower()
        if axis_name == "pitch":
            return float(angle.x)
        if axis_name == "roll":
            return float(angle.y)
        if axis_name == "yaw":
            return float(angle.z)
        raise ValueError(
            "unsupported IMU attitude axis: {}".format(axis_name)
        )

    def _axis_velocity(self, gyro, axis_name):
        axis_name = axis_name.lower()
        if axis_name == "pitch":
            return float(gyro.x)
        if axis_name == "roll":
            return float(gyro.y)
        if axis_name == "yaw":
            return float(gyro.z)
        raise ValueError(
            "unsupported IMU velocity axis: {}".format(axis_name)
        )

    def _read_sensor(self, now):
        sensor = self.sensor
        if sensor is None:
            return None, "IMU not initialized"

        read_all = _get_callable(sensor, "read_all")
        if read_all is None:
            return None, "IMU read_all method unavailable"

        try:
            data = read_all(True, True)
        except (
            AttributeError,
            RuntimeError,
            OSError,
            TypeError,
            ValueError
        ) as error:
            return None, "IMU read failed: {}".format(error)

        try:
            dt = max(
                0.001,
                min(
                    0.25,
                    float(now - self.last_read_time)
                )
            )
        except (TypeError, ValueError):
            dt = 0.01
        self.last_read_time = now

        if data is None:
            return None, "IMU read returned no data"

        mag = getattr(data, "mag", None)
        if mag is None:
            mag = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)

        ahrs_filter = self.ahrs_filter
        if ahrs_filter is None:
            return None, "AHRS filter not initialized"
        get_angle = _get_callable(ahrs_filter, "get_angle")
        if get_angle is None:
            return None, "AHRS get_angle method unavailable"

        try:
            angle = get_angle(
                data.acc,
                data.gyro,
                mag,
                dt,
                radian=False
            )
            pan_raw_deg = self._axis_angle(
                angle,
                self.pan_axis
            )
            tilt_raw_deg = self._axis_angle(
                angle,
                self.tilt_axis
            )
            pan_velocity_dps = math.degrees(
                self._axis_velocity(data.gyro, self.pan_axis)
            ) * self.pan_sign
            tilt_velocity_dps = math.degrees(
                self._axis_velocity(data.gyro, self.tilt_axis)
            ) * self.tilt_sign
        except (AttributeError, TypeError, ValueError) as error:
            return None, "AHRS calculation failed: {}".format(error)

        if not all(
            math.isfinite(value)
            for value in (
                pan_raw_deg,
                tilt_raw_deg,
                pan_velocity_dps,
                tilt_velocity_dps
            )
        ):
            return None, "IMU returned non-finite attitude"

        initial_pan_raw_deg = self.initial_pan_raw_deg
        if initial_pan_raw_deg is None:
            initial_pan_raw_deg = pan_raw_deg
            self.initial_pan_raw_deg = initial_pan_raw_deg

        initial_tilt_raw_deg = self.initial_tilt_raw_deg
        if initial_tilt_raw_deg is None:
            initial_tilt_raw_deg = tilt_raw_deg
            self.initial_tilt_raw_deg = initial_tilt_raw_deg

        self.pan_relative_deg = (
            _shortest_angle_delta(
                pan_raw_deg,
                initial_pan_raw_deg
            ) * self.pan_sign
        )
        self.tilt_relative_deg = (
            _shortest_angle_delta(
                tilt_raw_deg,
                initial_tilt_raw_deg
            ) * self.tilt_sign
        )

        temperature = getattr(data, "temp", None)
        return {
            "valid": True,
            "pan_relative_deg": self.pan_relative_deg,
            "tilt_relative_deg": self.tilt_relative_deg,
            "pan_velocity_dps": pan_velocity_dps,
            "tilt_velocity_dps": tilt_velocity_dps,
            "pan_raw_deg": pan_raw_deg,
            "tilt_raw_deg": tilt_raw_deg,
            "pitch_deg": float(angle.x),
            "roll_deg": float(angle.y),
            "yaw_deg": float(angle.z),
            "temperature_c": (
                float(temperature)
                if temperature is not None
                else None
            ),
        }, None

    def read(self, now=None):
        if not self.available:
            return {
                "valid": False,
                "error": self.initialization_error,
            }
        if now is None:
            now = time.monotonic()
        feedback, error = self._read_sensor(now)
        if feedback is None:
            if not self.first_read_error_printed:
                print_status(
                    "IMU",
                    "READ_ERROR",
                    fallback="visual_pid",
                    reason=error
                )
                self.first_read_error_printed = True
            return {
                "valid": False,
                "error": error,
            }
        if not self.first_valid_read_printed:
            temperature = feedback.get("temperature_c")
            print_status(
                "IMU",
                "RUNNING",
                pitch_deg=feedback["pitch_deg"],
                roll_deg=feedback["roll_deg"],
                yaw_deg=feedback["yaw_deg"],
                pan_relative_deg=feedback["pan_relative_deg"],
                tilt_relative_deg=feedback["tilt_relative_deg"],
                temperature_c=temperature
            )
            self.status = "RUNNING"
            self.first_valid_read_printed = True
        return feedback
