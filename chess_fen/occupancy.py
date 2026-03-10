import cv2
import numpy as np

from .fen_utils import _expand_fen_rows


def _estimate_occupied_squares(board_image):
    board_rgb = np.array(board_image.convert("RGB"))
    gray = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    height, width = gray.shape
    if height < 128 or width < 128:
        return 0

    cell_h = height / 8.0
    cell_w = width / 8.0
    tile_stds = []
    for rank_index in range(8):
        for file_index in range(8):
            y0 = int((rank_index + 0.30) * cell_h)
            y1 = int((rank_index + 0.70) * cell_h)
            x0 = int((file_index + 0.30) * cell_w)
            x1 = int((file_index + 0.70) * cell_w)
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            tile_stds.append(float(np.std(patch)))

    if not tile_stds:
        return 0

    tile_stds = np.array(tile_stds)
    threshold = float(np.median(tile_stds) + 0.55 * np.std(tile_stds))
    return int(np.sum(tile_stds > threshold))


def _tile_std_map(board_rgb):
    gray = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    height, width = gray.shape
    if height < 64 or width < 64:
        return np.zeros((8, 8), dtype=np.float32)

    cell_h = height / 8.0
    cell_w = width / 8.0
    std_map = np.zeros((8, 8), dtype=np.float32)

    for rank_index in range(8):
        for file_index in range(8):
            y0 = int((rank_index + 0.25) * cell_h)
            y1 = int((rank_index + 0.75) * cell_h)
            x0 = int((file_index + 0.25) * cell_w)
            x1 = int((file_index + 0.75) * cell_w)
            patch = gray[y0:y1, x0:x1]
            std_map[rank_index, file_index] = float(np.std(patch)) if patch.size else 0.0

    return std_map


def _occupied_mask(board_rgb):
    gray = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    height, width = gray.shape
    if height < 64 or width < 64:
        return np.zeros((8, 8), dtype=bool)

    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)

    cell_h = height / 8.0
    cell_w = width / 8.0
    std_map = np.zeros((8, 8), dtype=np.float32)
    grad_map = np.zeros((8, 8), dtype=np.float32)

    for rank_index in range(8):
        for file_index in range(8):
            y0 = int((rank_index + 0.25) * cell_h)
            y1 = int((rank_index + 0.75) * cell_h)
            x0 = int((file_index + 0.25) * cell_w)
            x1 = int((file_index + 0.75) * cell_w)

            patch = gray[y0:y1, x0:x1]
            grad_patch = gradient_mag[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            std_map[rank_index, file_index] = float(np.std(patch))
            grad_map[rank_index, file_index] = float(np.mean(grad_patch))

    median_std = float(np.median(std_map))
    std_std = float(np.std(std_map))
    median_grad = float(np.median(grad_map))
    std_grad = float(np.std(grad_map))

    occupancy_score = (
        ((std_map - median_std) / (std_std + 1e-6))
        + 0.8 * ((grad_map - median_grad) / (std_grad + 1e-6))
    )
    return occupancy_score > 0.20


def _rotate_mask_k_ccw(mask, k):
    return np.rot90(mask, k % 4)


def _fen_occupancy_mismatch(fen, occ_mask):
    rows = _expand_fen_rows(fen)
    if rows is None:
        return 999, 999

    false_negatives = 0
    false_positives = 0
    for rank_index in range(8):
        for file_index in range(8):
            fen_empty = rows[rank_index][file_index] == "1"
            image_occupied = bool(occ_mask[rank_index, file_index])
            if image_occupied and fen_empty:
                false_negatives += 1
            elif (not image_occupied) and (not fen_empty):
                false_positives += 1

    return false_negatives, false_positives
