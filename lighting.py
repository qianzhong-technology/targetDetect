"""相机画面亮度检测与补光灯控制。"""

import cv2

from config import (
    FILL_LIGHT_ON_THRESHOLD,
)


def update_fill_light(frame, fill_light_gpio, is_on):
    if is_on:
        return True, None

    height, width = frame.shape[:2]
    roi = frame[
        height // 4:3 * height // 4,
        width // 4:3 * width // 4
    ]
    mean_b, mean_g, mean_r, _ = cv2.mean(roi)
    brightness = 0.114 * mean_b + 0.587 * mean_g + 0.299 * mean_r

    if brightness < FILL_LIGHT_ON_THRESHOLD:
        fill_light_gpio.value(1)
        is_on = True

    return is_on, brightness
