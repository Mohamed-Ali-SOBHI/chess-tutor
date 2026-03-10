from typing import Dict

import numpy as np
from PIL import Image

from .fen_utils import (
    _apply_fen_votes,
    _fen_piece_count,
    _fix_last_rank_pawns,
    _init_votes,
    _is_valid_piece_placement,
    _looks_like_black_view,
    _rotate_fen_k_ccw,
    _score_fen_plausibility,
    _votes_to_fen,
)
from .occupancy import (
    _estimate_occupied_squares,
    _fen_occupancy_mismatch,
    _occupied_mask,
    _rotate_mask_k_ccw,
)
from .vision import _board_signature, _build_board_candidates, _square_center_crop


def _build_prediction_meta(quality_score, avg_confidence, stability_count, fallback_used):
    return {
        "quality_score": int(quality_score),
        "avg_confidence": float(avg_confidence),
        "stability_count": int(stability_count),
        "fallback_used": int(bool(fallback_used)),
    }


def _generate_prediction_crops(board_rgb):
    trim_specs = [
        (0.0, 0.0, 0.0, 0.0),
        (0.01, 0.01, 0.01, 0.01),
        (0.02, 0.02, 0.02, 0.02),
        (0.04, 0.04, 0.04, 0.04),
        (0.0, 0.0, 0.06, 0.0),
        (0.0, 0.0, 0.08, 0.0),
        (0.0, 0.0, 0.0, 0.06),
        (0.02, 0.02, 0.06, 0.0),
    ]

    height, width = board_rgb.shape[:2]
    seen_signatures = set()

    for left_ratio, right_ratio, top_ratio, bottom_ratio in trim_specs:
        left_margin = int(round(width * left_ratio))
        right_margin = int(round(width * right_ratio))
        top_margin = int(round(height * top_ratio))
        bottom_margin = int(round(height * bottom_ratio))

        if left_margin + right_margin >= width - 64:
            continue
        if top_margin + bottom_margin >= height - 64:
            continue

        cropped = board_rgb[
            top_margin:height - bottom_margin,
            left_margin:width - right_margin,
        ]
        if cropped.shape[0] < 128 or cropped.shape[1] < 128:
            continue

        candidate = _square_center_crop(cropped)
        signature = _board_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        yield candidate, left_ratio + right_ratio + top_ratio + bottom_ratio


def _build_candidate_board_specs(board_image, use_filters=True):
    raw_board_rgb = np.array(board_image.convert("RGB"))
    candidate_board_specs = [(raw_board_rgb, 0.0)]
    candidate_board_specs.extend(
        (board, 0.04)
        for board in _build_board_candidates(board_image, use_filters=use_filters)
    )

    deduped_candidates = {}
    for board, transform_penalty in candidate_board_specs:
        signature = _board_signature(board)
        current = deduped_candidates.get(signature)
        if current is None or transform_penalty < current[0]:
            deduped_candidates[signature] = (transform_penalty, board)

    return sorted(deduped_candidates.values(), key=lambda item: item[0])


def _evaluate_fen_candidate(
    candidate,
    predictor,
    rotation_index,
    estimated_occupancy,
    transform_penalty,
    trim_amount,
):
    if candidate.shape[0] < 128 or candidate.shape[1] < 128:
        return None

    try:
        result = predictor.predict_chessboard(
            Image.fromarray(candidate),
            fen_type="compressed",
        )
    except Exception:
        return None

    fen = (result.get("fen") or "").strip()
    predictions = result.get("predictions") or []
    if not fen or not predictions:
        return None

    fen = _rotate_fen_k_ccw(fen, (4 - rotation_index) % 4)
    is_black_view = _looks_like_black_view(fen)
    if is_black_view:
        fen = _rotate_fen_k_ccw(fen, 2)
    fen = _fix_last_rank_pawns(fen)

    avg_confidence = float(np.mean([item[2] for item in predictions]))
    quality_score = _score_fen_plausibility(fen)
    piece_count = _fen_piece_count(fen)
    occupancy_gap = abs(piece_count - estimated_occupancy)
    if not _is_valid_piece_placement(fen, side_to_move="w"):
        return None

    occ_board = _square_center_crop(candidate)
    occ_mask = _occupied_mask(occ_board)
    occ_mask = _rotate_mask_k_ccw(occ_mask, (4 - rotation_index) % 4)
    if is_black_view:
        occ_mask = _rotate_mask_k_ccw(occ_mask, 2)

    false_negatives, false_positives = _fen_occupancy_mismatch(fen, occ_mask)
    if false_negatives >= 6:
        return None

    occupancy_cell_penalty = 0.45 * false_negatives + 0.15 * false_positives
    fallback_rank = (
        avg_confidence,
        quality_score,
        -false_negatives,
        -false_positives,
        -occupancy_gap,
        piece_count,
    )
    aggregate_score = (
        avg_confidence
        + 0.20 * quality_score
        - 0.12 * occupancy_gap
        - occupancy_cell_penalty
        - 1.25 * trim_amount
        - transform_penalty
    )

    return {
        "fen": fen,
        "quality_score": quality_score,
        "avg_confidence": avg_confidence,
        "piece_count": piece_count,
        "occupancy_gap": occupancy_gap,
        "fallback_rank": fallback_rank,
        "aggregate_score": aggregate_score,
    }


def _new_fen_stats_entry():
    return {
        "count": 0.0,
        "sum_confidence": 0.0,
        "sum_quality": 0.0,
        "sum_piece_count": 0.0,
        "best_confidence": 0.0,
        "best_quality": 0.0,
        "sum_score": 0.0,
    }


def _update_fen_stats(fen_stats, evaluation):
    stats = fen_stats.setdefault(evaluation["fen"], _new_fen_stats_entry())
    stats["count"] += 1
    stats["sum_confidence"] += evaluation["avg_confidence"]
    stats["sum_quality"] += evaluation["quality_score"]
    stats["sum_piece_count"] += evaluation["piece_count"]
    stats["best_confidence"] = max(stats["best_confidence"], evaluation["avg_confidence"])
    stats["best_quality"] = max(stats["best_quality"], evaluation["quality_score"])
    stats["sum_score"] += evaluation["aggregate_score"]


def _select_best_single_fen(fen_stats):
    return max(
        fen_stats.items(),
        key=lambda item: (
            (item[1]["sum_score"] / item[1]["count"]),
            (item[1]["sum_confidence"] / item[1]["count"]),
            item[1]["best_confidence"],
            item[1]["best_quality"],
            item[1]["count"],
        ),
    )


def _meta_from_stats(stats, fallback_used):
    average_confidence = stats["sum_confidence"] / stats["count"]
    average_quality = stats["sum_quality"] / stats["count"]
    return _build_prediction_meta(
        quality_score=round(max(stats["best_quality"], average_quality)),
        avg_confidence=average_confidence,
        stability_count=round(stats["count"]),
        fallback_used=fallback_used,
    )


def _choose_final_fen(fen_stats, votes, estimated_occupancy):
    best_single_fen, best_single_stats = _select_best_single_fen(fen_stats)
    best_single_avg_score = best_single_stats["sum_score"] / best_single_stats["count"]

    ensemble_fen = _fix_last_rank_pawns(_votes_to_fen(votes))
    ensemble_piece_count = _fen_piece_count(ensemble_fen)
    ensemble_occupancy_gap = abs(ensemble_piece_count - estimated_occupancy)
    ensemble_stats = fen_stats.get(ensemble_fen)

    ensemble_is_competitive = False
    if ensemble_stats is not None:
        ensemble_avg_score = ensemble_stats["sum_score"] / ensemble_stats["count"]
        ensemble_is_competitive = (
            ensemble_fen == best_single_fen
            or (
                ensemble_stats["count"] >= 2
                and ensemble_avg_score >= best_single_avg_score
            )
        )

    if (
        ensemble_fen
        and _score_fen_plausibility(ensemble_fen) >= 7
        and _is_valid_piece_placement(ensemble_fen, side_to_move="w")
        and ensemble_occupancy_gap <= 8
        and ensemble_is_competitive
    ):
        return ensemble_fen, _meta_from_stats(best_single_stats, fallback_used=False)

    return best_single_fen, _meta_from_stats(best_single_stats, fallback_used=False)


def _predict_best_fen(board_image, predictor, use_filters=True):
    fen_stats: Dict[str, Dict[str, float]] = {}
    estimated_occupancy = _estimate_occupied_squares(board_image)
    votes = _init_votes()
    best_valid_any = None

    for transform_penalty, base_board in _build_candidate_board_specs(
        board_image,
        use_filters=use_filters,
    ):
        for rotation_index in range(4):
            rotated = np.rot90(base_board, rotation_index)
            for candidate, trim_amount in _generate_prediction_crops(rotated):
                evaluation = _evaluate_fen_candidate(
                    candidate=candidate,
                    predictor=predictor,
                    rotation_index=rotation_index,
                    estimated_occupancy=estimated_occupancy,
                    transform_penalty=transform_penalty,
                    trim_amount=trim_amount,
                )
                if evaluation is None:
                    continue

                if (
                    best_valid_any is None
                    or evaluation["fallback_rank"] > best_valid_any["fallback_rank"]
                ):
                    best_valid_any = evaluation

                if estimated_occupancy >= 12 and evaluation["occupancy_gap"] > 10:
                    continue

                _apply_fen_votes(
                    votes,
                    evaluation["fen"],
                    weight=evaluation["aggregate_score"],
                )
                _update_fen_stats(fen_stats, evaluation)

    if not fen_stats:
        if best_valid_any is not None:
            return best_valid_any["fen"], _build_prediction_meta(
                quality_score=best_valid_any["quality_score"],
                avg_confidence=best_valid_any["avg_confidence"],
                stability_count=1,
                fallback_used=True,
            )

        return "", _build_prediction_meta(
            quality_score=0,
            avg_confidence=0.0,
            stability_count=0,
            fallback_used=True,
        )

    return _choose_final_fen(fen_stats, votes, estimated_occupancy)
