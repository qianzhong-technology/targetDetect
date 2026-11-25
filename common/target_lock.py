"""Stable target-lock state with confirmation and loss hysteresis."""

from typing import Optional, Tuple


class TargetLockHysteresis:
    def __init__(
        self,
        lock_confirm_frames: int = 2,
        loss_confirm_frames: int = 3
    ) -> None:
        self.lock_confirm_frames = max(1, lock_confirm_frames)
        self.loss_confirm_frames = max(1, loss_confirm_frames)
        self.locked = False
        self.lock_hits = 0
        self.loss_hits = 0
        self.lock_point: Optional[Tuple[float, float]] = None

    def update(
        self,
        measured_ok: bool,
        point: Optional[Tuple[float, float]]
    ) -> Tuple[bool, Optional[Tuple[float, float]]]:
        if measured_ok and point is not None:
            self.lock_hits += 1
            self.loss_hits = 0
            if not self.locked:
                if self.lock_hits >= self.lock_confirm_frames:
                    self.locked = True
            if self.locked:
                self.lock_point = (float(point[0]), float(point[1]))
        else:
            self.loss_hits += 1
            self.lock_hits = 0
            if (
                self.locked
                and self.loss_hits >= self.loss_confirm_frames
            ):
                self.locked = False
                self.lock_point = None

        if self.locked:
            return True, self.lock_point
        return False, None

