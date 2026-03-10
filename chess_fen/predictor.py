from itertools import combinations, product
from typing import Dict

import numpy as np

from .batch_inference import predict_chessboard_batch_many
from .constants import (
    MIN_RELIABLE_CONFIDENCE,
    MIN_RELIABLE_CONFIDENCE_WITH_STRONG_STABILITY,
    MIN_RELIABLE_QUALITY,
    MIN_RELIABLE_STABILITY,
    MIN_RELIABLE_STRONG_STABILITY,
)
from .fen_utils import (
    _apply_fen_votes,
    _compress_fen_row,
    _expand_fen_rows,
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

DEFAULT_TRIM_SPECS = (
    (0.0, 0.0, 0.0, 0.0),
    (0.01, 0.01, 0.01, 0.01),
    (0.02, 0.02, 0.02, 0.02),
    (0.04, 0.04, 0.04, 0.04),
    (0.0, 0.0, 0.06, 0.0),
    (0.0, 0.0, 0.08, 0.0),
    (0.0, 0.0, 0.0, 0.06),
    (0.02, 0.02, 0.06, 0.0),
)
QUICK_TRIM_SPECS = DEFAULT_TRIM_SPECS[:4]
DEEP_TRIM_SPECS = DEFAULT_TRIM_SPECS[:4] + (
    (0.0, 0.0, 0.06, 0.0),
    (0.0, 0.0, 0.0, 0.06),
)
PHASE_SPECS = (
    {
        "name": "quick",
        "label": "Analyse rapide...",
        "complete_label": "Analyse rapide terminee.",
        "progress_start": 0.0,
        "progress_end": 45.0,
        "use_filters": False,
        "max_board_specs": 3,
        "trim_specs": QUICK_TRIM_SPECS,
    },
    {
        "name": "deep",
        "label": "Verification approfondie...",
        "complete_label": "Verification approfondie terminee.",
        "progress_start": 45.0,
        "progress_end": 80.0,
        "use_filters": False,
        "max_board_specs": 6,
        "trim_specs": DEEP_TRIM_SPECS,
    },
    {
        "name": "exhaustive",
        "label": "Analyse exhaustive...",
        "complete_label": "Analyse exhaustive terminee.",
        "progress_start": 80.0,
        "progress_end": 100.0,
        "use_filters": True,
        "max_board_specs": None,
        "trim_specs": DEFAULT_TRIM_SPECS,
    },
)
EXHAUSTIVE_ONLY_PHASE = (
    {
        "name": "exhaustive",
        "label": "Analyse du FEN...",
        "complete_label": "Analyse du FEN terminee.",
        "progress_start": 0.0,
        "progress_end": 100.0,
        "use_filters": True,
        "max_board_specs": None,
        "trim_specs": DEFAULT_TRIM_SPECS,
    },
)
LOCAL_REPAIR_MAX_PRIMARY_SQUARES = 4
LOCAL_REPAIR_MAX_SECONDARY_SQUARES = 2
LOCAL_REPAIR_MAX_CANDIDATES = 30
LOCAL_REPAIR_MAX_EDITS = 2
LOCAL_REPAIR_EDIT_PENALTY = 0.12
LOW_CONFIDENCE_REPAIR_THRESHOLD = 0.92
MIN_LOCAL_REPAIR_PLAUSIBILITY = 7
DEEP_PHASE_STABILITY_SHORT_CIRCUIT = 5
_MISSING = object()


def _build_prediction_meta(quality_score, avg_confidence, stability_count, fallback_used):
    return {
        "quality_score": int(quality_score),
        "avg_confidence": float(avg_confidence),
        "stability_count": int(stability_count),
        "fallback_used": int(bool(fallback_used)),
    }


def _meta_is_reliable(fen, meta):
    fen = (fen or "").strip()
    quality_score = int(meta.get("quality_score") or 0)
    avg_confidence = float(meta.get("avg_confidence") or 0.0)
    stability_count = int(meta.get("stability_count") or 0)
    fallback_used = bool(meta.get("fallback_used"))

    if not fen or fallback_used:
        return False
    if quality_score < MIN_RELIABLE_QUALITY:
        return False
    if stability_count < MIN_RELIABLE_STABILITY:
        return False

    meets_baseline_confidence = avg_confidence >= MIN_RELIABLE_CONFIDENCE
    meets_strong_stability_confidence = (
        stability_count >= MIN_RELIABLE_STRONG_STABILITY
        and avg_confidence >= MIN_RELIABLE_CONFIDENCE_WITH_STRONG_STABILITY
    )
    return meets_baseline_confidence or meets_strong_stability_confidence


def _progress_payload(phase_spec, current, total, message=None):
    total = max(1, int(total or 1))
    current = max(0, min(int(current or 0), total))
    ratio = current / total
    progress_percent = (
        float(phase_spec["progress_start"])
        + (float(phase_spec["progress_end"]) - float(phase_spec["progress_start"])) * ratio
    )
    return {
        "stage": "fen_phase",
        "phase_name": phase_spec["name"],
        "message": message or phase_spec["label"],
        "current": current,
        "total": total,
        "progress_percent": progress_percent,
    }


def _emit_progress(progress_callback, phase_spec, current, total, message=None):
    if progress_callback is None:
        return
    progress_callback(_progress_payload(phase_spec, current, total, message=message))


def _board_batch_materialization_size(predictor):
    device_label = str(getattr(predictor, "device", "cpu")).lower()
    if "cuda" in device_label:
        return 16
    return 8


def _rotate_cell_candidates_k_ccw(cell_candidates, k):
    k = k % 4
    if not k:
        return cell_candidates

    grid = np.array(cell_candidates, dtype=object)
    return np.rot90(grid, k).tolist()


def _rows_to_fen(rows):
    return "/".join(_compress_fen_row("".join(row)) for row in rows)


def _label_probability(candidates, label):
    for candidate_label, probability in candidates:
        if candidate_label == label:
            return float(probability)
    if not candidates:
        return 0.0
    return float(candidates[0][1])


def _square_options_for_repair(current_label, candidates, desired_occupied):
    options = []
    seen_labels = {current_label}

    for candidate_label, probability in candidates:
        if candidate_label in seen_labels:
            continue
        if desired_occupied and candidate_label == "1":
            continue
        if (not desired_occupied) and candidate_label != "1":
            continue

        seen_labels.add(candidate_label)
        options.append((candidate_label, float(probability)))
        if len(options) >= 3:
            break

    return options


def _resolve_last_rank_promotions(fen, cell_candidates):
    rows_raw = _expand_fen_rows(fen)
    if rows_raw is None:
        return fen

    rows = [list(row) for row in rows_raw]
    promotion_map = {
        "P": {"Q", "R", "B", "N"},
        "p": {"q", "r", "b", "n"},
    }
    changed = False

    for rank_index in (0, 7):
        for file_index in range(8):
            piece = rows[rank_index][file_index]
            if piece not in promotion_map:
                continue

            allowed_promotions = promotion_map[piece]
            best_label = None
            best_probability = -1.0
            for candidate_label, probability in cell_candidates[rank_index][file_index]:
                if candidate_label not in allowed_promotions:
                    continue
                if probability > best_probability:
                    best_label = candidate_label
                    best_probability = float(probability)

            if best_label is None:
                continue

            rows[rank_index][file_index] = best_label
            changed = True

    if changed:
        return _rows_to_fen(rows)
    return fen


def _normalize_candidate_fen(fen, cell_candidates):
    fen = _resolve_last_rank_promotions(fen, cell_candidates)
    return _fix_last_rank_pawns(fen)


def _quick_gate_candidate_fen(fen, cell_candidates):
    normalized_fen = _normalize_candidate_fen(fen, cell_candidates)
    quality_score = _score_fen_plausibility(normalized_fen)
    if quality_score < MIN_LOCAL_REPAIR_PLAUSIBILITY:
        return None, quality_score
    if not _is_valid_piece_placement(normalized_fen, side_to_move="w"):
        return None, quality_score
    return normalized_fen, quality_score


def _score_candidate_fen_variant(
    fen,
    avg_confidence,
    estimated_occupancy,
    occ_mask,
    transform_penalty,
    trim_amount,
    edit_count=0,
    cell_candidates=None,
    quality_score=None,
):
    if cell_candidates is not None:
        fen = _normalize_candidate_fen(fen, cell_candidates)
    else:
        fen = _fix_last_rank_pawns(fen)

    quality_score = (
        int(quality_score)
        if quality_score is not None
        else _score_fen_plausibility(fen)
    )
    piece_count = _fen_piece_count(fen)
    occupancy_gap = abs(piece_count - estimated_occupancy)
    if not _is_valid_piece_placement(fen, side_to_move="w"):
        return None

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
        -edit_count,
        piece_count,
    )
    aggregate_score = (
        avg_confidence
        + 0.20 * quality_score
        - 0.12 * occupancy_gap
        - occupancy_cell_penalty
        - 1.25 * trim_amount
        - transform_penalty
        - LOCAL_REPAIR_EDIT_PENALTY * edit_count
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


def _build_local_repair_sources(fen_rows, cell_candidates, occ_mask):
    primary_sources = []
    secondary_sources = []

    for rank_index in range(8):
        for file_index in range(8):
            candidates = cell_candidates[rank_index][file_index]
            if not candidates:
                continue

            current_label = fen_rows[rank_index][file_index]
            current_prob = _label_probability(candidates, current_label)
            fen_occupied = current_label != "1"
            image_occupied = bool(occ_mask[rank_index, file_index])

            if fen_occupied != image_occupied:
                repair_options = _square_options_for_repair(
                    current_label=current_label,
                    candidates=candidates,
                    desired_occupied=image_occupied,
                )
                if repair_options:
                    primary_sources.append(
                        {
                            "coords": (rank_index, file_index),
                            "current_label": current_label,
                            "current_prob": current_prob,
                            "priority": 2 if image_occupied else 1,
                            "options": repair_options,
                        }
                    )

            if fen_occupied and current_prob < LOW_CONFIDENCE_REPAIR_THRESHOLD:
                piece_options = _square_options_for_repair(
                    current_label=current_label,
                    candidates=candidates,
                    desired_occupied=True,
                )
                if piece_options:
                    secondary_sources.append(
                        {
                            "coords": (rank_index, file_index),
                            "current_label": current_label,
                            "current_prob": current_prob,
                            "priority": 0,
                            "options": piece_options,
                        }
                    )

    primary_sources.sort(key=lambda item: (-item["priority"], item["current_prob"]))
    secondary_sources.sort(key=lambda item: item["current_prob"])

    selected = primary_sources[:LOCAL_REPAIR_MAX_PRIMARY_SQUARES]
    selected_coords = {item["coords"] for item in selected}
    for source in secondary_sources:
        if source["coords"] in selected_coords:
            continue
        selected.append(source)
        selected_coords.add(source["coords"])
        if len(selected_coords) >= (
            LOCAL_REPAIR_MAX_PRIMARY_SQUARES + LOCAL_REPAIR_MAX_SECONDARY_SQUARES
        ):
            break

    return selected


def _generate_local_repair_evaluations(
    fen,
    cell_candidates,
    occ_mask,
    estimated_occupancy,
    transform_penalty,
    trim_amount,
):
    fen_rows_raw = _expand_fen_rows(fen)
    if fen_rows_raw is None:
        return []

    fen_rows = [list(row) for row in fen_rows_raw]
    repair_sources = _build_local_repair_sources(fen_rows, cell_candidates, occ_mask)
    if not repair_sources:
        return []

    base_probability_sum = 0.0
    for rank_index in range(8):
        for file_index in range(8):
            base_probability_sum += _label_probability(
                cell_candidates[rank_index][file_index],
                fen_rows[rank_index][file_index],
            )

    pending_repairs = []
    for source in repair_sources:
        for option in source["options"]:
            pending_repairs.append((source, option))

    repair_sets = []
    for source, option in pending_repairs:
        repair_sets.append(((source, option),))

    if len(repair_sources) >= 2 and LOCAL_REPAIR_MAX_EDITS >= 2:
        for source_a, source_b in combinations(repair_sources, 2):
            for option_a, option_b in product(source_a["options"], source_b["options"]):
                repair_sets.append(
                    (
                        (source_a, option_a),
                        (source_b, option_b),
                    )
                )

    repair_sets.sort(
        key=lambda edits: (
            len(edits),
            -sum(item[0]["priority"] for item in edits),
            -sum(item[1][1] for item in edits),
        )
    )

    seen_fens = set()
    evaluations = []
    for edits in repair_sets[:LOCAL_REPAIR_MAX_CANDIDATES]:
        candidate_rows = [row[:] for row in fen_rows]
        probability_sum = base_probability_sum

        for source, option in edits:
            rank_index, file_index = source["coords"]
            new_label, new_probability = option
            candidate_rows[rank_index][file_index] = new_label
            probability_sum += new_probability - source["current_prob"]

        candidate_fen = _rows_to_fen(candidate_rows)
        if candidate_fen in seen_fens:
            continue
        seen_fens.add(candidate_fen)

        evaluation = _score_candidate_fen_variant(
            fen=candidate_fen,
            avg_confidence=probability_sum / 64.0,
            estimated_occupancy=estimated_occupancy,
            occ_mask=occ_mask,
            transform_penalty=transform_penalty,
            trim_amount=trim_amount,
            edit_count=len(edits),
            cell_candidates=cell_candidates,
        )
        if evaluation is not None:
            evaluations.append(evaluation)

    return evaluations


def _candidate_from_trim_spec(board_rgb, trim_spec):
    left_ratio, right_ratio, top_ratio, bottom_ratio = trim_spec
    height, width = board_rgb.shape[:2]

    left_margin = int(round(width * left_ratio))
    right_margin = int(round(width * right_ratio))
    top_margin = int(round(height * top_ratio))
    bottom_margin = int(round(height * bottom_ratio))

    if left_margin + right_margin >= width - 64:
        return None
    if top_margin + bottom_margin >= height - 64:
        return None

    cropped = board_rgb[
        top_margin:height - bottom_margin,
        left_margin:width - right_margin,
    ]
    if cropped.shape[0] < 128 or cropped.shape[1] < 128:
        return None

    candidate = _square_center_crop(cropped)
    if candidate.shape[0] < 128 or candidate.shape[1] < 128:
        return None
    return candidate


def _generate_prediction_crops(board_rgb, trim_specs=None):
    trim_specs = trim_specs or DEFAULT_TRIM_SPECS
    seen_signatures = set()

    for trim_spec in trim_specs:
        candidate = _candidate_from_trim_spec(board_rgb, trim_spec)
        if candidate is None:
            continue

        signature = _board_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        yield candidate, trim_spec, sum(trim_spec)


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


def _build_phase_candidate_descriptors(board_specs, phase_spec):
    descriptors = []
    trim_specs = phase_spec["trim_specs"]

    for board_index, (transform_penalty, base_board) in enumerate(board_specs):
        for rotation_index in range(4):
            rotated = np.rot90(base_board, rotation_index)
            for _, trim_spec, trim_amount in _generate_prediction_crops(
                rotated,
                trim_specs=trim_specs,
            ):
                descriptors.append(
                    {
                        "cache_key": (
                            bool(phase_spec["use_filters"]),
                            board_index,
                            rotation_index,
                            tuple(trim_spec),
                        ),
                        "base_board": base_board,
                        "rotation_index": rotation_index,
                        "transform_penalty": transform_penalty,
                        "trim_spec": tuple(trim_spec),
                        "trim_amount": trim_amount,
                    }
                )

    return descriptors


def _materialize_descriptor_candidate(descriptor):
    rotated = np.rot90(descriptor["base_board"], descriptor["rotation_index"])
    return _candidate_from_trim_spec(rotated, descriptor["trim_spec"])


def _normalize_prediction_result(fen, cell_candidates, rotation_index):
    fen = _rotate_fen_k_ccw(fen, (4 - rotation_index) % 4)
    cell_candidates = _rotate_cell_candidates_k_ccw(
        cell_candidates,
        (4 - rotation_index) % 4,
    )
    is_black_view = _looks_like_black_view(fen)
    if is_black_view:
        fen = _rotate_fen_k_ccw(fen, 2)
        cell_candidates = _rotate_cell_candidates_k_ccw(cell_candidates, 2)

    return fen, cell_candidates, is_black_view


def _build_occupancy_mask(candidate, rotation_index, is_black_view):
    occ_board = _square_center_crop(candidate)
    occ_mask = _occupied_mask(occ_board)
    occ_mask = _rotate_mask_k_ccw(occ_mask, (4 - rotation_index) % 4)
    if is_black_view:
        occ_mask = _rotate_mask_k_ccw(occ_mask, 2)
    return occ_mask


def _evaluate_prediction_result(
    candidate,
    result,
    descriptor,
    estimated_occupancy,
):
    fen = (result.get("fen") or "").strip()
    predictions = result.get("predictions") or []
    cell_candidates = result.get("cell_candidates") or []
    if candidate is None or not fen or not predictions or not cell_candidates:
        return None

    fen, cell_candidates, is_black_view = _normalize_prediction_result(
        fen=fen,
        cell_candidates=cell_candidates,
        rotation_index=descriptor["rotation_index"],
    )
    normalized_fen, quality_score = _quick_gate_candidate_fen(fen, cell_candidates)
    if not normalized_fen:
        return None

    occ_mask = _build_occupancy_mask(
        candidate=candidate,
        rotation_index=descriptor["rotation_index"],
        is_black_view=is_black_view,
    )
    avg_confidence = float(np.mean([item[2] for item in predictions]))
    evaluations = []

    base_evaluation = _score_candidate_fen_variant(
        fen=normalized_fen,
        avg_confidence=avg_confidence,
        estimated_occupancy=estimated_occupancy,
        occ_mask=occ_mask,
        transform_penalty=descriptor["transform_penalty"],
        trim_amount=descriptor["trim_amount"],
        cell_candidates=cell_candidates,
        quality_score=quality_score,
    )
    if base_evaluation is not None:
        evaluations.append(base_evaluation)

    evaluations.extend(
        _generate_local_repair_evaluations(
            fen=normalized_fen,
            cell_candidates=cell_candidates,
            occ_mask=occ_mask,
            estimated_occupancy=estimated_occupancy,
            transform_penalty=descriptor["transform_penalty"],
            trim_amount=descriptor["trim_amount"],
        )
    )
    if not evaluations:
        return None

    return max(
        evaluations,
        key=lambda item: (
            item["aggregate_score"],
            item["avg_confidence"],
            item["quality_score"],
            item["fallback_rank"],
        ),
    )


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
        and _score_fen_plausibility(ensemble_fen) >= MIN_LOCAL_REPAIR_PLAUSIBILITY
        and _is_valid_piece_placement(ensemble_fen, side_to_move="w")
        and ensemble_occupancy_gap <= 8
        and ensemble_is_competitive
    ):
        return ensemble_fen, _meta_from_stats(best_single_stats, fallback_used=False)

    return best_single_fen, _meta_from_stats(best_single_stats, fallback_used=False)


def _apply_evaluation(evaluation, estimated_occupancy, fen_stats, votes, best_valid_any):
    if evaluation is None:
        return best_valid_any

    if best_valid_any is None or evaluation["fallback_rank"] > best_valid_any["fallback_rank"]:
        best_valid_any = evaluation

    if estimated_occupancy >= 12 and evaluation["occupancy_gap"] > 10:
        return best_valid_any

    _apply_fen_votes(
        votes,
        evaluation["fen"],
        weight=evaluation["aggregate_score"],
    )
    _update_fen_stats(fen_stats, evaluation)
    return best_valid_any


def _finalize_phase_result(fen_stats, votes, estimated_occupancy, best_valid_any, phase_name):
    if not fen_stats:
        if best_valid_any is not None:
            meta = _build_prediction_meta(
                quality_score=best_valid_any["quality_score"],
                avg_confidence=best_valid_any["avg_confidence"],
                stability_count=1,
                fallback_used=True,
            )
            meta["phase_name"] = phase_name
            return best_valid_any["fen"], meta

        meta = _build_prediction_meta(
            quality_score=0,
            avg_confidence=0.0,
            stability_count=0,
            fallback_used=True,
        )
        meta["phase_name"] = phase_name
        return "", meta

    fen, meta = _choose_final_fen(fen_stats, votes, estimated_occupancy)
    meta["phase_name"] = phase_name
    return fen, meta


def _run_prediction_phase(
    board_image,
    predictor,
    estimated_occupancy,
    phase_spec,
    board_specs_cache,
    evaluation_cache,
    progress_callback=None,
):
    board_specs = board_specs_cache.get(phase_spec["use_filters"])
    if board_specs is None:
        board_specs = _build_candidate_board_specs(
            board_image,
            use_filters=phase_spec["use_filters"],
        )
        board_specs_cache[phase_spec["use_filters"]] = board_specs

    max_board_specs = phase_spec["max_board_specs"]
    if max_board_specs is not None:
        board_specs = board_specs[:max_board_specs]

    descriptors = _build_phase_candidate_descriptors(board_specs, phase_spec)
    total_descriptors = len(descriptors)
    fen_stats: Dict[str, Dict[str, float]] = {}
    votes = _init_votes()
    best_valid_any = None
    processed = 0
    pending_descriptors = []

    _emit_progress(
        progress_callback,
        phase_spec,
        current=0,
        total=total_descriptors,
        message=phase_spec["label"],
    )

    def _maybe_return_reliable_result():
        if not fen_stats:
            return None

        fen, meta = _finalize_phase_result(
            fen_stats=fen_stats,
            votes=votes,
            estimated_occupancy=estimated_occupancy,
            best_valid_any=best_valid_any,
            phase_name=phase_spec["name"],
        )
        if _meta_is_reliable(fen, meta):
            _emit_progress(
                progress_callback,
                phase_spec,
                current=processed,
                total=total_descriptors,
                message=phase_spec["complete_label"],
            )
            return fen, meta
        return None

    def _flush_pending():
        nonlocal best_valid_any, processed, pending_descriptors

        if not pending_descriptors:
            return None

        candidates = [_materialize_descriptor_candidate(descriptor) for descriptor in pending_descriptors]
        results = predict_chessboard_batch_many(
            candidates,
            predictor=predictor,
            fen_type="compressed",
        )
        for descriptor, candidate, result in zip(pending_descriptors, candidates, results):
            evaluation = _evaluate_prediction_result(
                candidate=candidate,
                result=result,
                descriptor=descriptor,
                estimated_occupancy=estimated_occupancy,
            )
            evaluation_cache[descriptor["cache_key"]] = evaluation
            best_valid_any = _apply_evaluation(
                evaluation=evaluation,
                estimated_occupancy=estimated_occupancy,
                fen_stats=fen_stats,
                votes=votes,
                best_valid_any=best_valid_any,
            )
            processed += 1
            reliable_result = _maybe_return_reliable_result()
            if reliable_result is not None:
                pending_descriptors = []
                return reliable_result
            if processed == total_descriptors or processed % 4 == 0:
                _emit_progress(
                    progress_callback,
                    phase_spec,
                    current=processed,
                    total=total_descriptors,
                    message=phase_spec["label"],
                )

        pending_descriptors = []
        return None

    materialization_chunk_size = _board_batch_materialization_size(predictor)
    for descriptor in descriptors:
        cached_evaluation = evaluation_cache.get(descriptor["cache_key"], _MISSING)
        if cached_evaluation is not _MISSING:
            best_valid_any = _apply_evaluation(
                evaluation=cached_evaluation,
                estimated_occupancy=estimated_occupancy,
                fen_stats=fen_stats,
                votes=votes,
                best_valid_any=best_valid_any,
            )
            processed += 1
            reliable_result = _maybe_return_reliable_result()
            if reliable_result is not None:
                return reliable_result
            if processed == total_descriptors or processed % 4 == 0:
                _emit_progress(
                    progress_callback,
                    phase_spec,
                    current=processed,
                    total=total_descriptors,
                    message=phase_spec["label"],
                )
            continue

        pending_descriptors.append(descriptor)
        if len(pending_descriptors) >= materialization_chunk_size:
            reliable_result = _flush_pending()
            if reliable_result is not None:
                return reliable_result

    reliable_result = _flush_pending()
    if reliable_result is not None:
        return reliable_result
    _emit_progress(
        progress_callback,
        phase_spec,
        current=total_descriptors,
        total=total_descriptors,
        message=phase_spec["complete_label"],
    )
    return _finalize_phase_result(
        fen_stats=fen_stats,
        votes=votes,
        estimated_occupancy=estimated_occupancy,
        best_valid_any=best_valid_any,
        phase_name=phase_spec["name"],
    )


def _predict_best_fen(
    board_image,
    predictor,
    use_filters=True,
    progress_callback=None,
    strategy="auto_precise",
):
    estimated_occupancy = _estimate_occupied_squares(board_image)
    board_specs_cache = {}
    evaluation_cache = {}

    if strategy == "exhaustive":
        phase_specs = EXHAUSTIVE_ONLY_PHASE
        if not use_filters:
            phase_specs = (
                dict(EXHAUSTIVE_ONLY_PHASE[0], use_filters=False),
            )
    else:
        phase_specs = PHASE_SPECS
        if not use_filters:
            phase_specs = tuple(
                dict(phase_spec, use_filters=False)
                for phase_spec in phase_specs
            )

    last_fen = ""
    last_meta = _build_prediction_meta(
        quality_score=0,
        avg_confidence=0.0,
        stability_count=0,
        fallback_used=True,
    )
    previous_phase_fen = ""

    for phase_spec in phase_specs:
        fen, meta = _run_prediction_phase(
            board_image=board_image,
            predictor=predictor,
            estimated_occupancy=estimated_occupancy,
            phase_spec=phase_spec,
            board_specs_cache=board_specs_cache,
            evaluation_cache=evaluation_cache,
            progress_callback=progress_callback,
        )
        last_fen = fen
        last_meta = meta

        if _meta_is_reliable(fen, meta):
            return fen, meta

        if phase_spec["name"] == "quick":
            previous_phase_fen = fen
            continue

        if (
            phase_spec["name"] == "deep"
            and fen
            and fen == previous_phase_fen
            and int(meta.get("stability_count") or 0) >= DEEP_PHASE_STABILITY_SHORT_CIRCUIT
        ):
            return fen, meta

        if (
            phase_spec["name"] == "deep"
            and fen
            and fen == previous_phase_fen
            and int(meta.get("quality_score") or 0) >= MIN_RELIABLE_QUALITY
            and float(meta.get("avg_confidence") or 0.0)
            < MIN_RELIABLE_CONFIDENCE_WITH_STRONG_STABILITY
        ):
            return fen, meta

    return last_fen, last_meta
