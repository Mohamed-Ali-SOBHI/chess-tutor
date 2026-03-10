from functools import reduce

import cv2
import numpy as np
import torch
from PIL import Image

from chessimg2pos.utils import compressed_fen

CPU_BOARD_BATCH_SIZE = 8
CUDA_BOARD_BATCH_SIZE = 16
TOP_K_TILE_CANDIDATES = 5


def _board_batch_size_for_predictor(predictor):
    device_label = str(getattr(predictor, "device", "cpu")).lower()
    if "cuda" in device_label:
        return CUDA_BOARD_BATCH_SIZE
    return CPU_BOARD_BATCH_SIZE


def _resize_board_to_256(candidate_rgb):
    board_image = Image.fromarray(candidate_rgb, "RGB")
    resized_board = board_image.resize((256, 256), Image.BILINEAR)
    return np.asarray(resized_board, dtype=np.uint8)


def _prepare_single_tile_batch(candidate_rgb, use_grayscale):
    resized_board = _resize_board_to_256(candidate_rgb)

    if use_grayscale:
        board_array = cv2.cvtColor(resized_board, cv2.COLOR_RGB2GRAY)
        tiles = (
            board_array.reshape(8, 32, 8, 32)
            .transpose(0, 2, 1, 3)
            .reshape(64, 1, 32, 32)
        )
    else:
        tiles = (
            resized_board.reshape(8, 32, 8, 32, 3)
            .transpose(0, 2, 4, 1, 3)
            .reshape(64, 3, 32, 32)
        )

    tile_batch = np.ascontiguousarray(tiles, dtype=np.float32)
    tile_batch /= 255.0
    tile_batch -= 0.5
    tile_batch /= 0.5
    return tile_batch


def _prepare_board_tile_batch(candidate_rgbs, use_grayscale):
    per_board_batches = [
        _prepare_single_tile_batch(candidate_rgb, use_grayscale=use_grayscale)
        for candidate_rgb in candidate_rgbs
    ]
    tile_batch = np.concatenate(per_board_batches, axis=0)
    return torch.from_numpy(tile_batch)


def _build_predictions_from_outputs(probabilities, fen_chars):
    max_probs, predicted_indices = torch.max(probabilities, dim=1)
    predictions = []

    for index in range(64):
        row = 7 - (index // 8)
        col = index % 8
        square = chr(97 + col) + str(row + 1)
        fen_char = fen_chars[predicted_indices[index].item()]
        probability = max_probs[index].item()
        predictions.append((square, fen_char, probability))

    return predictions


def _build_cell_candidates_from_outputs(
    probabilities,
    fen_chars,
    top_k=TOP_K_TILE_CANDIDATES,
):
    top_k = max(1, min(int(top_k), probabilities.shape[1]))
    top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)
    cell_candidates = [[[] for _ in range(8)] for _ in range(8)]

    for index in range(64):
        rank_index = index // 8
        file_index = index % 8
        candidates = []
        for candidate_offset in range(top_k):
            label = fen_chars[top_indices[index, candidate_offset].item()]
            probability = top_probs[index, candidate_offset].item()
            candidates.append((label, probability))
        cell_candidates[rank_index][file_index] = candidates

    return cell_candidates


def _predictions_to_fen(predictions, fen_type):
    board_matrix = np.empty((8, 8), dtype=object)
    for square, fen_char, _ in predictions:
        row = 7 - (ord(square[1]) - ord("1"))
        col = ord(square[0]) - ord("a")
        board_matrix[row, col] = fen_char

    fen_notation = "/".join("".join(row) for row in board_matrix)
    if fen_type == "compressed":
        return compressed_fen(fen_notation)
    return fen_notation


def _result_from_probabilities(probabilities, fen_chars, fen_type):
    predictions = _build_predictions_from_outputs(probabilities, fen_chars)
    cell_candidates = _build_cell_candidates_from_outputs(probabilities, fen_chars)
    fen = _predictions_to_fen(predictions, fen_type=fen_type)
    confidence = reduce(lambda x, y: x * y, (item[2] for item in predictions), 1.0)
    return {
        "fen": fen,
        "confidence": confidence,
        "predictions": predictions,
        "cell_candidates": cell_candidates,
    }


def _run_model_for_boards(candidate_rgbs, predictor):
    tile_batch = _prepare_board_tile_batch(candidate_rgbs, predictor.use_grayscale)
    if getattr(predictor, "_use_channels_last", False) and tile_batch.ndim == 4:
        tile_batch = tile_batch.contiguous(memory_format=torch.channels_last)
    tile_batch = tile_batch.to(
        predictor.device,
        non_blocking=getattr(predictor, "_non_blocking_transfers", False),
    )

    with torch.inference_mode():
        outputs = predictor.model(tile_batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    board_count = len(candidate_rgbs)
    return probabilities.reshape(board_count, 64, probabilities.shape[-1])


def predict_chessboard_batch_many(
    candidate_rgbs,
    predictor,
    fen_type="standard",
    board_batch_size=None,
):
    if not candidate_rgbs:
        return []

    board_batch_size = board_batch_size or _board_batch_size_for_predictor(predictor)
    results = []
    for start_index in range(0, len(candidate_rgbs), board_batch_size):
        candidate_chunk = candidate_rgbs[start_index:start_index + board_batch_size]
        probabilities_chunk = _run_model_for_boards(candidate_chunk, predictor)
        for board_index in range(probabilities_chunk.shape[0]):
            results.append(
                _result_from_probabilities(
                    probabilities=probabilities_chunk[board_index],
                    fen_chars=predictor.fen_chars,
                    fen_type=fen_type,
                )
            )

    return results


def predict_chessboard_batch(candidate_rgb, predictor, fen_type="standard"):
    results = predict_chessboard_batch_many(
        [candidate_rgb],
        predictor=predictor,
        fen_type=fen_type,
        board_batch_size=1,
    )
    return results[0]
