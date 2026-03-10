from chess_fen.constants import (
    DEBUG_IMAGE_ENV_VAR,
    DEFAULT_ENGINE_ELO,
    EXCLUDED_IMAGE_DIRS,
    IGNORED_IMAGE_PREFIXES,
    IMAGE_EXTENSIONS,
)
from chess_fen.capture import capture_full_screen
from chess_fen.engine import (
    _configure_engine_strength,
    _resolve_stockfish_path,
    get_best_move_from_fen,
)
from chess_fen.fen_utils import (
    VoteMap,
    _apply_fen_votes,
    _basic_fen_sanity,
    _compress_fen_row,
    _expand_fen_rows,
    _fen_piece_count,
    _fen_to_grid,
    _fix_last_rank_pawns,
    _grid_to_fen,
    _init_votes,
    _is_valid_piece_placement,
    _looks_like_black_view,
    _rotate_fen_180,
    _rotate_fen_k_ccw,
    _rotate_grid_ccw,
    _score_fen_plausibility,
    _to_full_fen,
    _votes_to_fen,
)
from chess_fen.occupancy import (
    _estimate_occupied_squares,
    _fen_occupancy_mismatch,
    _occupied_mask,
    _rotate_mask_k_ccw,
    _tile_std_map,
)
from chess_fen.predictor import (
    _build_candidate_board_specs,
    _build_prediction_meta,
    _choose_final_fen,
    _evaluate_fen_candidate,
    _generate_prediction_crops,
    _meta_from_stats,
    _new_fen_stats_entry,
    _predict_best_fen,
    _select_best_single_fen,
    _update_fen_stats,
)
from chess_fen.service import (
    analyze_board_image,
    analyze_image,
    _convert_single_image_to_fen,
    _find_manual_override,
    _format_prediction_status,
    _list_image_paths,
    _load_board_image,
    _load_manual_overrides,
    _load_predictor,
    _normalize_path_key,
    analyze_image_and_suggest_move,
    analyze_screen_image,
    analyze_screen_image_and_suggest_move,
    convert_images_to_fen,
    save_manual_fen_override,
)
from chess_fen.ui import launch_desktop_app
from chess_fen.vision import (
    _board_signature,
    _build_board_candidates,
    _checkerboard_alignment_score,
    _deduplicate_boards,
    _extract_perspective_square,
    _generate_filter_variants,
    _order_points,
    _refine_alignment_variants,
    _save_debug_crop,
    _square_center_crop,
    detect_and_crop_board,
    detect_and_crop_board_from_image,
)


if __name__ == "__main__":
    convert_images_to_fen()
