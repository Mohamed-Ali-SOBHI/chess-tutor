import os

import cv2
import numpy as np
from PIL import Image

from .constants import DEBUG_IMAGE_ENV_VAR


def _save_debug_crop(image_path, crop):
    debug_flag = os.environ.get(DEBUG_IMAGE_ENV_VAR, "").strip().lower()
    if debug_flag not in {"1", "true", "yes", "on"}:
        return

    temp_dir = os.environ.get("TEMP_DIR", ".")
    os.makedirs(temp_dir, exist_ok=True)
    debug_path = os.path.join(temp_dir, f"cropped_{os.path.basename(image_path)}")
    cv2.imwrite(debug_path, crop)
    print(f"  -> Cropped image saved for debug: {debug_path}")


def _ensure_bgr_image(image):
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _detect_and_crop_board_from_bgr(image_bgr, source_label, fallback_image):
    img = _ensure_bgr_image(image_bgr)
    if img is None:
        print(f"  -> ERROR: Cannot read image with OpenCV: {source_label}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = img.shape[0] * img.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.10 or area > img_area * 0.98:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            print(f"  -> Board detected ({w}x{h})")
            crop = img[y:y + h, x:x + w]
            _save_debug_crop(source_label, crop)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            return Image.fromarray(crop_rgb)

    print("  -> WARNING: No clear board contour found. Using full image.")
    if fallback_image is None:
        return None
    return fallback_image.convert("RGB")


def detect_and_crop_board(image_path):
    image_bgr = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_UNCHANGED)
    fallback_image = None
    if image_bgr is not None:
        try:
            fallback_image = Image.open(image_path).convert("RGB")
        except Exception:
            fallback_image = None

    return _detect_and_crop_board_from_bgr(
        image_bgr,
        source_label=image_path,
        fallback_image=fallback_image,
    )


def detect_and_crop_board_from_image(image, source_label="screen_capture.png"):
    if image is None:
        return None

    source_image = image.convert("RGB")
    image_bgr = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
    return _detect_and_crop_board_from_bgr(
        image_bgr,
        source_label=source_label,
        fallback_image=source_image,
    )


def _order_points(points):
    points = np.asarray(points, dtype=np.float32)
    sum_xy = points.sum(axis=1)
    diff_xy = np.diff(points, axis=1).reshape(-1)
    top_left = points[np.argmin(sum_xy)]
    bottom_right = points[np.argmax(sum_xy)]
    top_right = points[np.argmin(diff_xy)]
    bottom_left = points[np.argmax(diff_xy)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _extract_perspective_square(board_rgb):
    h, w = board_rgb.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < img_area * 0.30 or area > img_area * 0.99:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue

        src = _order_points(approx.reshape(4, 2))
        side = int(
            max(
                np.linalg.norm(src[0] - src[1]),
                np.linalg.norm(src[1] - src[2]),
                np.linalg.norm(src[2] - src[3]),
                np.linalg.norm(src[3] - src[0]),
            )
        )
        if side < 128:
            continue

        dst = np.array(
            [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(board_rgb, matrix, (side, side))

    return None


def _generate_filter_variants(board_rgb):
    variants = [board_rgb]

    gray = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)
    variants.append(cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2RGB))

    blur = cv2.GaussianBlur(board_rgb, (0, 0), 1.0)
    sharpened = cv2.addWeighted(board_rgb, 1.4, blur, -0.4, 0)
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 0.95, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.07, 0, 255)
    variants.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))

    return variants


def _square_center_crop(image_rgb):
    height, width = image_rgb.shape[:2]
    if height == width:
        return image_rgb

    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return image_rgb[y0:y0 + side, x0:x0 + side]


def _checkerboard_alignment_score(board_rgb):
    gray = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    height, width = gray.shape
    if height < 128 or width < 128:
        return -1e9

    cell_h = height / 8.0
    cell_w = width / 8.0
    even_means = []
    odd_means = []

    for rank_index in range(8):
        for file_index in range(8):
            y0 = int((rank_index + 0.30) * cell_h)
            y1 = int((rank_index + 0.70) * cell_h)
            x0 = int((file_index + 0.30) * cell_w)
            x1 = int((file_index + 0.70) * cell_w)

            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            patch_mean = float(np.mean(patch))
            if (rank_index + file_index) % 2 == 0:
                even_means.append(patch_mean)
            else:
                odd_means.append(patch_mean)

    if len(even_means) < 16 or len(odd_means) < 16:
        return -1e9

    contrast = abs(float(np.mean(even_means) - np.mean(odd_means)))
    intra_variance = float(np.std(even_means) + np.std(odd_means))
    return contrast - 0.45 * intra_variance


def _board_signature(board):
    return (
        board.shape[0],
        board.shape[1],
        int(np.mean(board)),
        int(np.std(board)),
        int(board[0, 0, 0]),
        int(board[-1, -1, 0]),
    )


def _refine_alignment_variants(board_rgb, max_variants=3):
    square = _square_center_crop(board_rgb)
    side = square.shape[0]
    if side < 128:
        return [square]

    scored_variants = []
    seen_signatures = set()
    trim_specs = {(0.0, 0.0, 0.0, 0.0)}

    for ratio in (0.01, 0.02, 0.04):
        trim_specs.add((ratio, ratio, ratio, ratio))

    for ratio in (0.04, 0.06, 0.08):
        trim_specs.add((0.0, 0.0, ratio, 0.0))
        trim_specs.add((0.0, 0.0, 0.0, ratio))
        trim_specs.add((0.0, 0.0, ratio, ratio))

    for horizontal_ratio in (0.02, 0.04):
        for vertical_ratio in (0.06, 0.08):
            trim_specs.add((horizontal_ratio, horizontal_ratio, vertical_ratio, 0.0))
            trim_specs.add((horizontal_ratio, horizontal_ratio, 0.0, vertical_ratio))
            trim_specs.add((horizontal_ratio, horizontal_ratio, vertical_ratio, vertical_ratio))

    for left_ratio, right_ratio, top_ratio, bottom_ratio in sorted(trim_specs):
        left_margin = int(round(side * left_ratio))
        right_margin = int(round(side * right_ratio))
        top_margin = int(round(side * top_ratio))
        bottom_margin = int(round(side * bottom_ratio))

        if left_margin + right_margin >= side - 64:
            continue
        if top_margin + bottom_margin >= side - 64:
            continue

        cropped = square[
            top_margin:side - bottom_margin,
            left_margin:side - right_margin,
        ]
        if cropped.shape[0] < 128 or cropped.shape[1] < 128:
            continue

        refined = _square_center_crop(cropped)
        signature = _board_signature(refined)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        total_trim = left_ratio + right_ratio + top_ratio + bottom_ratio
        score = _checkerboard_alignment_score(refined)
        scored_variants.append((score, -total_trim, refined))

    if not scored_variants:
        return [square]

    scored_variants.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [board for _, _, board in scored_variants[:max_variants]]


def _deduplicate_boards(boards):
    unique_boards = []
    seen_signatures = set()

    for board in boards:
        signature = _board_signature(board)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_boards.append(board)

    return unique_boards


def _build_board_candidates(board_image, use_filters=True):
    board_rgb = np.array(board_image.convert("RGB"))
    geometric_candidates = [board_rgb]

    height, width = board_rgb.shape[:2]
    if abs(height - width) > 2:
        side = min(height, width)
        y0 = (height - side) // 2
        x0 = (width - side) // 2
        geometric_candidates.append(board_rgb[y0:y0 + side, x0:x0 + side])

    warped = _extract_perspective_square(board_rgb)
    if warped is not None:
        geometric_candidates.append(warped)

    base_candidates = []
    if use_filters:
        for candidate in geometric_candidates:
            base_candidates.extend(_generate_filter_variants(candidate))
    else:
        base_candidates.extend(geometric_candidates)

    deduped_base_candidates = _deduplicate_boards(base_candidates)
    refined_candidates = list(deduped_base_candidates)
    for candidate in deduped_base_candidates:
        refined_candidates.extend(_refine_alignment_variants(candidate, max_variants=3))

    refined_candidates = _deduplicate_boards(refined_candidates)
    scored = [
        (_checkerboard_alignment_score(candidate), candidate)
        for candidate in refined_candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)

    max_candidates = 18 if use_filters else 10
    return [candidate for _, candidate in scored[:max_candidates]]
