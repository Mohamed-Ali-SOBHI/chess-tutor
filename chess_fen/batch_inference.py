from functools import reduce

import numpy as np
import torch
from PIL import Image

from chessimg2pos.utils import compressed_fen

TOP_K_TILE_CANDIDATES = 5


def _resize_board_to_256(candidate_rgb):
    board_image = Image.fromarray(candidate_rgb, "RGB")
    return board_image.resize((256, 256), Image.BILINEAR)


def _prepare_tile_batch(candidate_rgb, use_grayscale):
    resized_board = _resize_board_to_256(candidate_rgb)

    if use_grayscale:
        resized_board = resized_board.convert("L", (0.2989, 0.5870, 0.1140, 0))
        board_array = np.asarray(resized_board, dtype=np.uint8)
        tiles = (
            board_array.reshape(8, 32, 8, 32)
            .transpose(0, 2, 1, 3)
            .reshape(64, 32, 32)
        )
        tile_batch = torch.from_numpy(np.ascontiguousarray(tiles)).unsqueeze(1)
    else:
        board_array = np.asarray(resized_board, dtype=np.uint8)
        tiles = (
            board_array.reshape(8, 32, 8, 32, 3)
            .transpose(0, 2, 4, 1, 3)
            .reshape(64, 3, 32, 32)
        )
        tile_batch = torch.from_numpy(np.ascontiguousarray(tiles))

    tile_batch = tile_batch.float().div_(255.0)
    tile_batch.sub_(0.5).div_(0.5)
    return tile_batch


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


def predict_chessboard_batch(candidate_rgb, predictor, fen_type="standard"):
    tile_batch = _prepare_tile_batch(candidate_rgb, predictor.use_grayscale)
    if getattr(predictor, "_use_channels_last", False) and tile_batch.ndim == 4:
        tile_batch = tile_batch.contiguous(memory_format=torch.channels_last)
    tile_batch = tile_batch.to(
        predictor.device,
        non_blocking=getattr(predictor, "_non_blocking_transfers", False),
    )

    with torch.inference_mode():
        outputs = predictor.model(tile_batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    predictions = _build_predictions_from_outputs(probabilities, predictor.fen_chars)
    cell_candidates = _build_cell_candidates_from_outputs(
        probabilities,
        predictor.fen_chars,
    )
    fen = _predictions_to_fen(predictions, fen_type=fen_type)
    confidence = reduce(lambda x, y: x * y, (item[2] for item in predictions), 1.0)

    return {
        "fen": fen,
        "confidence": confidence,
        "predictions": predictions,
        "cell_candidates": cell_candidates,
    }
