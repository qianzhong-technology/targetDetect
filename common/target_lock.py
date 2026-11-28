"""Stable target-lock state with confirmation and loss hysteresis."""

from typing import Optional, Tuple


class TargetLockHysteresis:
    def __init__(
        self,
        lock_confirm_frames: int = 2,
        loss_confirm_frames: int = 3,
        max_point_jump_px: float = 40.0,
        point_smooth_alpha: float = 0.75
    ) -> None:
        self.lock_confirm_frames = max(1, lock_confirm_frames)
        self.loss_confirm_frames = max(1, loss_confirm_frames)
        self.max_point_jump_px = max(1.0, float(max_point_jump_px))
        self.point_smooth_alpha = min(
            1.0,
            max(0.0, float(point_smooth_alpha))
        )
        self.locked = False
        self.lock_hits = 0
        self.loss_hits = 0
        self.lock_point: Optional[Tuple[float, float]] = None
        self.pending_point: Optional[Tuple[float, float]] = None

    def update(
        self,
        measured_ok: bool,
        point: Optional[Tuple[float, float]]
    ) -> Tuple[bool, Optional[Tuple[float, float]]]:
        if measured_ok and point is not None:
            measured_point = (
                float(point[0]),
                float(point[1])
            )

            if self.locked and self.lock_point is not None:
                jump = (
                    (measured_point[0] - self.lock_point[0]) ** 2
                    + (measured_point[1] - self.lock_point[1]) ** 2
                ) ** 0.5
                if jump > self.max_point_jump_px:
                    self.loss_hits += 1
                    self.lock_hits = 0
                    self.pending_point = None
                    if self.loss_hits >= self.loss_confirm_frames:
                        self.locked = False
                        self.lock_point = None
                    if not self.locked:
                        return False, None
                    return True, self.lock_point

            if not self.locked:
                if self.pending_point is not None:
                    pending_jump = (
                        (measured_point[0] - self.pending_point[0]) ** 2
                        + (measured_point[1] - self.pending_point[1]) ** 2
                    ) ** 0.5
                    if pending_jump > self.max_point_jump_px:
                        self.pending_point = measured_point
                        self.lock_hits = 1
                        self.loss_hits = 0
                        return False, None
                else:
                    self.pending_point = measured_point

                self.lock_hits += 1
                self.loss_hits = 0
                if self.lock_hits >= self.lock_confirm_frames:
                    self.locked = True
                    self.lock_point = measured_point
            else:
                self.lock_hits = 0
                self.loss_hits = 0
                if self.lock_point is not None:
                    alpha = self.point_smooth_alpha
                    smoothed = (
                        alpha * measured_point[0]
                        + (1.0 - alpha) * self.lock_point[0],
                        alpha * measured_point[1]
                        + (1.0 - alpha) * self.lock_point[1]
                    )
                    self.lock_point = smoothed
                else:
                    self.lock_point = measured_point

            if self.locked:
                return True, self.lock_point
            return False, None
        else:
            self.loss_hits += 1
            self.lock_hits = 0
            self.pending_point = None
            if (
                self.locked
                and self.loss_hits >= self.loss_confirm_frames
            ):
                self.locked = False
                self.lock_point = None

        if self.locked:
            return True, self.lock_point
        return False, None

