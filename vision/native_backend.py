"""C++/NEON 加速后端桥接。"""

import ctypes
import glob
import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from common.config import (
    TARGET_ACCEL_BACKEND,
    TARGET_ACCEL_LIB_PATH,
)


def _resolve_library_path() -> str:
    env_path = os.getenv("TARGETDETECT_NATIVE_LIB")
    if env_path:
        return env_path
    repo_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(repo_root, TARGET_ACCEL_LIB_PATH)


class NativeVisionBackend:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False
        self._available = False
        self._backend_mode = TARGET_ACCEL_BACKEND.lower().strip()
        self._lib: Optional[ctypes.CDLL] = None
        self._loaded_dep_libs: List[ctypes.CDLL] = []
        self._failure_reason = ""

    @property
    def is_available(self) -> bool:
        self._ensure_initialized()
        return self._available

    @property
    def failure_reason(self) -> str:
        self._ensure_initialized()
        return self._failure_reason

    @property
    def backend_mode(self) -> str:
        return self._backend_mode

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return

            mode = self._backend_mode
            if mode == "python":
                self._failure_reason = "backend forced to python"
                self._initialized = True
                return

            lib_path = _resolve_library_path()
            if not os.path.exists(lib_path):
                self._failure_reason = "native library missing: {}".format(lib_path)
                self._initialized = True
                return

            preload_error = self._preload_dependency_libraries(lib_path)
            if preload_error is not None:
                self._failure_reason = preload_error
                self._initialized = True
                return

            try:
                lib = ctypes.CDLL(lib_path)
            except OSError as exc:
                self._failure_reason = "failed to load native library: {}".format(exc)
                self._initialized = True
                return

            self._configure_signatures(lib)
            if mode == "npu" and not bool(lib.td_npu_stage_supported()):
                self._failure_reason = "npu stage unavailable in native library"
                self._initialized = True
                return

            self._lib = lib
            self._available = True
            self._initialized = True

    def _preload_dependency_libraries(self, lib_path: str) -> Optional[str]:
        dep_dir_env = os.getenv("TARGETDETECT_NATIVE_DL_LIB_DIR")
        search_dirs = []
        if dep_dir_env:
            search_dirs.append(dep_dir_env)

        repo_root = os.path.dirname(os.path.dirname(__file__))
        search_dirs.extend([
            os.path.join(repo_root, "vision", "native", "dl_lib"),
            os.path.join(repo_root, "vision", "dl_lib"),
            os.path.join(os.path.dirname(lib_path), "dl_lib"),
        ])

        loaded_paths = set()
        self._loaded_dep_libs = []
        load_mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        for directory in search_dirs:
            if not os.path.isdir(directory):
                continue
            for dep_path in sorted(glob.glob(os.path.join(directory, "*.so*"))):
                if dep_path in loaded_paths:
                    continue
                try:
                    dep_lib = ctypes.CDLL(dep_path, mode=load_mode)
                except OSError as exc:
                    return "failed to load dependency {}: {}".format(dep_path, exc)
                self._loaded_dep_libs.append(dep_lib)
                loaded_paths.add(dep_path)
        return None

    @staticmethod
    def _configure_signatures(lib: ctypes.CDLL) -> None:
        float_ptr = ctypes.POINTER(ctypes.c_float)
        double_ptr = ctypes.POINTER(ctypes.c_double)
        uint8_ptr = ctypes.POINTER(ctypes.c_uint8)
        int_ptr = ctypes.POINTER(ctypes.c_int32)

        lib.td_make_homography.argtypes = [
            float_ptr,
            ctypes.c_int32,
            ctypes.c_int32,
            double_ptr,
        ]
        lib.td_make_homography.restype = ctypes.c_int32

        lib.td_warp_perspective_bgr.argtypes = [
            uint8_ptr,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            double_ptr,
            ctypes.c_int32,
            ctypes.c_int32,
            uint8_ptr,
            ctypes.c_int32,
        ]
        lib.td_warp_perspective_bgr.restype = ctypes.c_int32

        lib.td_inverse_project_point.argtypes = [
            double_ptr,
            ctypes.c_float,
            ctypes.c_float,
            float_ptr,
        ]
        lib.td_inverse_project_point.restype = ctypes.c_int32

        lib.td_lk_track_corners.argtypes = [
            uint8_ptr,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            uint8_ptr,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            float_ptr,
            ctypes.c_int32,
            float_ptr,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            float_ptr,
            float_ptr,
            ctypes.c_int32,
            int_ptr,
            int_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
        ]
        lib.td_lk_track_corners.restype = ctypes.c_int32

        lib.td_fit_target_center_mask.argtypes = [
            uint8_ptr,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            float_ptr,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_int32,
            float_ptr,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            float_ptr,
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.td_fit_target_center_mask.restype = ctypes.c_int32

        lib.td_npu_stage_supported.argtypes = []
        lib.td_npu_stage_supported.restype = ctypes.c_int32

    @staticmethod
    def _as_u8_image(array: np.ndarray) -> np.ndarray:
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        return np.ascontiguousarray(array)

    @staticmethod
    def _as_f32_array(array: np.ndarray) -> np.ndarray:
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        return np.ascontiguousarray(array)

    @staticmethod
    def _as_f64_array(array: np.ndarray) -> np.ndarray:
        if array.dtype != np.float64:
            array = array.astype(np.float64)
        return np.ascontiguousarray(array)

    def supports_npu_stage(self) -> bool:
        self._ensure_initialized()
        if not self._available or self._lib is None:
            return False
        return bool(self._lib.td_npu_stage_supported())

    def make_homography(
        self,
        corners: np.ndarray,
        rect_width: int,
        rect_height: int,
    ) -> Optional[np.ndarray]:
        self._ensure_initialized()
        if not self._available or self._lib is None:
            return None

        corners = self._as_f32_array(corners.reshape(4, 2))
        out_h = np.zeros(9, dtype=np.float64)
        rc = self._lib.td_make_homography(
            corners.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(rect_width),
            ctypes.c_int32(rect_height),
            out_h.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            return None
        return out_h.reshape(3, 3)

    def warp_perspective(
        self,
        frame: np.ndarray,
        homography: np.ndarray,
        rect_width: int,
        rect_height: int,
    ) -> Optional[np.ndarray]:
        self._ensure_initialized()
        if not self._available or self._lib is None:
            return None

        frame_u8 = self._as_u8_image(frame)
        if frame_u8.ndim != 3 or frame_u8.shape[2] != 3:
            return None

        h64 = self._as_f64_array(homography.reshape(3, 3))
        output = np.empty((rect_height, rect_width, 3), dtype=np.uint8)
        rc = self._lib.td_warp_perspective_bgr(
            frame_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(frame_u8.shape[0]),
            ctypes.c_int32(frame_u8.shape[1]),
            ctypes.c_int32(frame_u8.shape[2]),
            ctypes.c_int32(frame_u8.strides[0]),
            h64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int32(rect_width),
            ctypes.c_int32(rect_height),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(output.strides[0]),
        )
        if rc != 0:
            return None
        return output

    def inverse_project_point(
        self,
        point: Tuple[float, float],
        homography: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        self._ensure_initialized()
        if not self._available or self._lib is None:
            return None

        h64 = self._as_f64_array(homography.reshape(3, 3))
        out_xy = np.zeros(2, dtype=np.float32)
        rc = self._lib.td_inverse_project_point(
            h64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_float(float(point[0])),
            ctypes.c_float(float(point[1])),
            out_xy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            return None
        return (float(out_xy[0]), float(out_xy[1]))

    def lk_track_corners(
        self,
        previous_gray: np.ndarray,
        gray: np.ndarray,
        previous_points: np.ndarray,
        corners: np.ndarray,
        min_features: int,
        min_inlier_ratio: float,
        support_min_ratio: float,
        area_min_ratio: float,
        area_max_ratio: float,
    ) -> Optional[Dict[str, object]]:
        self._ensure_initialized()
        if not self._available or self._lib is None:
            return None

        previous_gray_u8 = self._as_u8_image(previous_gray)
        gray_u8 = self._as_u8_image(gray)
        points = self._as_f32_array(previous_points.reshape(-1, 2))
        corners = self._as_f32_array(corners.reshape(4, 2))

        out_corners = np.zeros((4, 2), dtype=np.float32)
        out_points = np.zeros_like(points)
        out_points_count = ctypes.c_int32(0)
        out_inlier_count = ctypes.c_int32(0)
        out_support_ratio = np.zeros(1, dtype=np.float32)
        out_area_ratio = np.zeros(1, dtype=np.float32)
        out_max_shift = np.zeros(1, dtype=np.float32)

        rc = self._lib.td_lk_track_corners(
            previous_gray_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(previous_gray_u8.shape[0]),
            ctypes.c_int32(previous_gray_u8.shape[1]),
            ctypes.c_int32(previous_gray_u8.strides[0]),
            gray_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(gray_u8.shape[0]),
            ctypes.c_int32(gray_u8.shape[1]),
            ctypes.c_int32(gray_u8.strides[0]),
            points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(points.shape[0]),
            corners.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(min_features),
            ctypes.c_float(float(min_inlier_ratio)),
            ctypes.c_float(float(support_min_ratio)),
            ctypes.c_float(float(area_min_ratio)),
            ctypes.c_float(float(area_max_ratio)),
            out_corners.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(out_points.shape[0]),
            ctypes.byref(out_points_count),
            ctypes.byref(out_inlier_count),
            out_support_ratio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_area_ratio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_max_shift.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            return None

        tracked_count = int(out_points_count.value)
        if tracked_count <= 0:
            return None

        return {
            "corners": out_corners.astype(np.float32, copy=False),
            "tracked_points": out_points[:tracked_count].reshape(-1, 1, 2).astype(
                np.float32,
                copy=False,
            ),
            "inlier_count": int(out_inlier_count.value),
            "support_ratio": float(out_support_ratio[0]),
            "area_ratio": float(out_area_ratio[0]),
            "max_shift": float(out_max_shift[0]),
        }

    def fit_target_center_mask(
        self,
        mask: np.ndarray,
        scale_x: float,
        scale_y: float,
        expected_radii_px: List[float],
        search_tolerance: float,
        max_radius_error: float,
        check_ring_radius: bool,
        center_prior_weight: float,
        fit_rms_warning_threshold: float,
        rect_width: int,
        rect_height: int,
    ) -> Optional[Tuple[Optional[Tuple[float, float]], Optional[float], Optional[List[float]], str]]:
        self._ensure_initialized()
        if not self._available or self._lib is None:
            return None

        mask_u8 = self._as_u8_image(mask)
        if mask_u8.ndim != 2:
            return None

        radii = self._as_f32_array(np.asarray(expected_radii_px, dtype=np.float32))
        out_center = np.zeros(2, dtype=np.float32)
        out_rms = ctypes.c_float(0.0)
        out_status = ctypes.c_int32(0)
        out_radii = np.zeros(len(expected_radii_px), dtype=np.float32)
        out_radii_count = ctypes.c_int32(0)

        rc = self._lib.td_fit_target_center_mask(
            mask_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(mask_u8.shape[0]),
            ctypes.c_int32(mask_u8.shape[1]),
            ctypes.c_int32(mask_u8.strides[0]),
            ctypes.c_float(float(scale_x)),
            ctypes.c_float(float(scale_y)),
            radii.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(len(expected_radii_px)),
            ctypes.c_float(float(search_tolerance)),
            ctypes.c_float(float(max_radius_error)),
            ctypes.c_int32(1 if check_ring_radius else 0),
            ctypes.c_float(float(center_prior_weight)),
            ctypes.c_float(float(fit_rms_warning_threshold)),
            ctypes.c_int32(rect_width),
            ctypes.c_int32(rect_height),
            out_center.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(out_status),
            ctypes.byref(out_rms),
            out_radii.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(out_radii_count),
        )
        if rc != 0:
            reason = _RING_FIT_ERRORS.get(int(rc), "native fit failed")
            return None, None, None, reason

        radii_count = max(0, int(out_radii_count.value))
        fitted = out_radii[:radii_count].astype(np.float32, copy=False)
        center = (float(out_center[0]), float(out_center[1]))
        rms = float(out_rms.value)
        fit_status = "OK"
        if out_status.value == 1:
            fit_status = "low confidence: RMS {:.1f}px > {:.1f}px".format(
                rms,
                fit_rms_warning_threshold,
            )
        return center, rms, [float(value) for value in fitted], fit_status


_RING_FIT_ERRORS = {
    -1: "empty red mask",
    -2: "insufficient ring candidates",
    -3: "ring centers inconsistent",
    -4: "joint circle fit failed",
    -5: "joint radii outside target range",
}


_NATIVE_BACKEND_SINGLETON: Optional[NativeVisionBackend] = None


def get_native_backend() -> NativeVisionBackend:
    global _NATIVE_BACKEND_SINGLETON
    if _NATIVE_BACKEND_SINGLETON is None:
        _NATIVE_BACKEND_SINGLETON = NativeVisionBackend()
    return _NATIVE_BACKEND_SINGLETON

