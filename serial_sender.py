"""MaixCAM 靶心坐标的串口发送模块。"""

import math
import time

from maix import err, pinmap, uart

from console_output import print_status

from config import (
    UART_BAUDRATE,
    UART_HEARTBEAT_INTERVAL,
    UART_PORT,
    UART_RX_PIN,
    UART_SEND_INTERVAL,
    UART_TX_PIN,
)


class TargetSerialSender:
    def __init__(self):
        err.check_raise(
            pinmap.set_pin_function(UART_TX_PIN, "UART1_TX"),
            "set UART1 TX pin failed"
        )
        err.check_raise(
            pinmap.set_pin_function(UART_RX_PIN, "UART1_RX"),
            "set UART1 RX pin failed"
        )
        self.serial = uart.UART(
            UART_PORT,
            UART_BAUDRATE
        )
        self.last_error = None
        self.last_send_time = 0.0
        self.last_successful_send_time = 0.0
        self.last_sent_state = None
        self.next_retry_time = 0.0
        self.failure_count = 0

    def close(self):
        if self.serial.is_open():
            err.check_raise(
                self.serial.close(),
                "close UART1 failed"
            )

    def send_frame(
        self,
        center_camera
    ):
        # 使用相机原始坐标发送：@x,y + CRLF
        # 没有有限有效中心点时，不发送坐标。
        if center_camera is None:
            return False

        raw_x = float(center_camera[0])
        raw_y = float(center_camera[1])
        if not math.isfinite(raw_x) or not math.isfinite(raw_y):
            return False

        x = round(raw_x, 2)
        y = round(raw_y, 2)

        state = (x, y)
        now = time.monotonic()
        state_changed = state != self.last_sent_state
        heartbeat_due = (
            now - self.last_successful_send_time >= UART_HEARTBEAT_INTERVAL
        )
        if not state_changed and not heartbeat_due:
            return False
        if now < self.next_retry_time:
            return False

        encoded = "@{:.2f},{:.2f}\r\n".format(x, y).encode("ascii")

        # 8N1 帧每字节实际占用 10 个位，并预留缓冲余量。
        minimum_interval = max(
            UART_SEND_INTERVAL,
            len(encoded) * 10.0 / UART_BAUDRATE * 1.25
        )
        if now - self.last_send_time < minimum_interval:
            return False

        self.last_send_time = now
        sent = self.serial.write(encoded)
        if sent < 0:
            self._record_failure(sent, now, "UART send failed")
            return False

        if sent != len(encoded):
            self._record_failure(
                "partial",
                now,
                "UART partial send: {} / {}".format(sent, len(encoded))
            )
            return False

        if self.last_error is not None:
            print_status(
                "UART",
                "RECOVERED"
            )
        self.last_error = None
        self.failure_count = 0
        self.next_retry_time = 0.0
        self.last_sent_state = state
        self.last_successful_send_time = time.monotonic()
        return True

    def _record_failure(self, error, now, message):
        if self.last_error != error:
            print_status(
                "UART",
                "SEND_ERROR",
                detail=message
            )
            self.last_error = error

        self.failure_count += 1
        retry_delay = min(
            300.0,
            30.0 * 2.0 ** min(self.failure_count - 1, 4)
        )
        self.next_retry_time = now + retry_delay
