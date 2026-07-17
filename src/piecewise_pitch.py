"""Helpers for a pitch split at a camera standing on the near touchline."""

import cv2
import numpy as np


def transform_piecewise(points, H_left, H_right, split_x, pitch_length):
    """Map image points and choose the half by the projected center seam."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    H_left = np.asarray(H_left, dtype=np.float64)
    H_right = np.asarray(H_right, dtype=np.float64)
    left = cv2.perspectiveTransform(points, H_left).reshape(-1, 2)
    right = cv2.perspectiveTransform(points, H_right).reshape(-1, 2)

    seam_model = np.array([
        [split_x, 0], [split_x, 1],
        [split_x / 2, 1],
    ], dtype=np.float64).reshape(-1, 1, 2)
    seam_image = cv2.perspectiveTransform(
        seam_model, np.linalg.inv(H_left)).reshape(-1, 2)
    seam_start, seam_end, left_reference = seam_image
    direction = seam_end - seam_start

    def side(values):
        offsets = values - seam_start
        return direction[0] * offsets[:, 1] - direction[1] * offsets[:, 0]

    reference_side = side(left_reference.reshape(1, 2))[0]
    image_points = points.reshape(-1, 2)
    use_left = side(image_points) * reference_side >= 0

    # The seam can be numerically unstable far outside the image. Fall back to
    # the valid x interval only in that degenerate case.
    if abs(reference_side) < 1e-8:
        left_penalty = np.maximum(-left[:, 0], 0) + np.maximum(
            left[:, 0] - split_x, 0)
        right_penalty = np.maximum(split_x - right[:, 0], 0) + np.maximum(
            right[:, 0] - pitch_length, 0)
        use_left = left_penalty <= right_penalty
    mapped = np.where(use_left[:, None], left, right)
    return mapped, use_left


def draw_piecewise_overlay(image, H_left, H_right, split_x,
                           pitch_length, pitch_width, thickness=3):
    """Draw the combined outer boundary and the half-field seam."""
    left_model = np.array([
        [0, 0], [0, pitch_width],
        [split_x, pitch_width], [split_x, 0],
    ], dtype=np.float64).reshape(-1, 1, 2)
    right_model = np.array([
        [split_x, 0], [split_x, pitch_width],
        [pitch_length, pitch_width], [pitch_length, 0],
    ], dtype=np.float64).reshape(-1, 1, 2)
    left_px = cv2.perspectiveTransform(
        left_model, np.linalg.inv(np.asarray(H_left))).reshape(-1, 2)
    right_px = cv2.perspectiveTransform(
        right_model, np.linalg.inv(np.asarray(H_right))).reshape(-1, 2)
    boundary = np.array([
        left_px[0], left_px[1], left_px[2],
        right_px[2], right_px[3], left_px[3],
    ], dtype=np.int32)
    result = image.copy()
    cv2.polylines(result, [boundary], True, (0, 255, 255),
                  thickness, cv2.LINE_AA)
    seam = np.array([left_px[2], left_px[3]], dtype=np.int32)
    cv2.line(result, tuple(seam[0]), tuple(seam[1]),
             (255, 220, 0), max(1, thickness - 1), cv2.LINE_AA)
    return result
