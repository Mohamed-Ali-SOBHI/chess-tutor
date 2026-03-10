import csv
import os

from PIL import Image

from chessimg2pos import ChessPositionPredictor
from chessimg2pos.constants import DEFAULT_CLASSIFIER
from chessimg2pos.model_loader import download_pretrained_model

from .constants import (
    DEFAULT_ENGINE_ELO,
    EXCLUDED_IMAGE_DIRS,
    IGNORED_IMAGE_PREFIXES,
    IMAGE_EXTENSIONS,
)
from .engine import get_best_move_from_fen
from .fen_utils import _to_full_fen
from .predictor import _predict_best_fen
from .vision import detect_and_crop_board, detect_and_crop_board_from_image


def _load_predictor():
    model_path = download_pretrained_model()
    return ChessPositionPredictor(
        model_path=model_path,
        classifier=DEFAULT_CLASSIFIER,
        verbose=False,
    )


def _load_board_image(image_path):
    board_image = detect_and_crop_board(image_path)
    if board_image is not None:
        return board_image, None

    try:
        return Image.open(image_path).convert("RGB"), "  -> WARNING: fallback to original image"
    except Exception as open_error:
        return None, f"  -> ERROR: cannot open image: {open_error}"


def _build_prediction_result(source_label, board_image, fen, meta):
    return {
        "source_label": source_label,
        "fen": fen,
        "quality_score": meta["quality_score"],
        "avg_confidence": meta["avg_confidence"],
        "stability_count": int(meta.get("stability_count", 0)),
        "fallback_used": bool(meta.get("fallback_used", 0)),
        "board_image": board_image,
    }


def _predict_board_image(board_image, predictor, use_filters, source_label):
    if board_image is None:
        raise RuntimeError("Impossible de charger l'image du plateau")

    fen, meta = _predict_best_fen(board_image, predictor, use_filters=use_filters)
    return _build_prediction_result(
        source_label=source_label,
        board_image=board_image,
        fen=fen,
        meta=meta,
    )


def analyze_board_image(
    board_image,
    predictor=None,
    use_filters=True,
    source_label="board_image",
):
    predictor = predictor or _load_predictor()
    normalized_image = board_image.convert("RGB")
    return _predict_board_image(
        board_image=normalized_image,
        predictor=predictor,
        use_filters=use_filters,
        source_label=source_label,
    )


def analyze_image(
    image_path,
    predictor=None,
    use_filters=True,
):
    predictor = predictor or _load_predictor()
    board_image, board_image_message = _load_board_image(image_path)
    if board_image is None:
        raise RuntimeError(board_image_message or "Impossible de charger l'image")

    result = _predict_board_image(
        board_image=board_image,
        predictor=predictor,
        use_filters=use_filters,
        source_label=image_path,
    )
    result["image_path"] = image_path
    result["board_image_message"] = board_image_message
    return result


def analyze_screen_image(
    screen_image,
    predictor=None,
    use_filters=True,
    source_label="screen_capture.png",
):
    predictor = predictor or _load_predictor()
    board_image = detect_and_crop_board_from_image(
        screen_image,
        source_label=source_label,
    )
    if board_image is None:
        raise RuntimeError("Impossible de detecter un plateau dans la capture d'ecran")

    return _predict_board_image(
        board_image=board_image,
        predictor=predictor,
        use_filters=use_filters,
        source_label=source_label,
    )


def _attach_move_recommendation(
    prediction_result,
    stockfish_path,
    side_to_move,
    think_time,
    engine_elo,
):
    result = dict(prediction_result)
    fen = result["fen"]

    if not fen:
        result.update(
            {
                "best_move_uci": "",
                "best_move_san": "",
                "score_cp": None,
                "mate_in": None,
                "engine_elo": engine_elo,
                "stockfish_path": stockfish_path,
            }
        )
        return result

    move_info = get_best_move_from_fen(
        fen,
        stockfish_path=stockfish_path,
        side_to_move=side_to_move,
        think_time=think_time,
        engine_elo=engine_elo,
    )

    result.update(
        {
            "best_move_uci": move_info["best_move_uci"],
            "best_move_san": move_info["best_move_san"],
            "score_cp": move_info["score_cp"],
            "mate_in": move_info["mate_in"],
            "engine_elo": move_info["engine_elo"],
            "stockfish_path": move_info["stockfish_path"],
        }
    )
    return result


def analyze_image_and_suggest_move(
    image_path,
    stockfish_path=None,
    side_to_move="w",
    think_time=0.20,
    use_filters=True,
    engine_elo=DEFAULT_ENGINE_ELO,
    predictor=None,
):
    prediction_result = analyze_image(
        image_path,
        predictor=predictor,
        use_filters=use_filters,
    )
    return _attach_move_recommendation(
        prediction_result=prediction_result,
        stockfish_path=stockfish_path,
        side_to_move=side_to_move,
        think_time=think_time,
        engine_elo=engine_elo,
    )


def analyze_screen_image_and_suggest_move(
    screen_image,
    stockfish_path=None,
    side_to_move="w",
    think_time=0.20,
    use_filters=True,
    engine_elo=DEFAULT_ENGINE_ELO,
    predictor=None,
    source_label="screen_capture.png",
):
    prediction_result = analyze_screen_image(
        screen_image,
        predictor=predictor,
        use_filters=use_filters,
        source_label=source_label,
    )
    return _attach_move_recommendation(
        prediction_result=prediction_result,
        stockfish_path=stockfish_path,
        side_to_move=side_to_move,
        think_time=think_time,
        engine_elo=engine_elo,
    )


def _list_image_paths(images_dir):
    image_paths = []
    for root, dirs, files in os.walk(images_dir):
        dirs[:] = [directory for directory in dirs if directory not in EXCLUDED_IMAGE_DIRS]
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            normalized_name = filename.lower()
            if normalized_name.startswith(IGNORED_IMAGE_PREFIXES):
                continue
            image_paths.append(os.path.abspath(os.path.join(root, filename)))

    image_paths.sort()
    return image_paths


def _normalize_path_key(path):
    return os.path.normcase(os.path.abspath(path))


def _load_manual_overrides(manual_overrides_csv):
    overrides_by_path = {}
    overrides_by_name = {}

    if not manual_overrides_csv or not os.path.isfile(manual_overrides_csv):
        return overrides_by_path, overrides_by_name

    with open(manual_overrides_csv, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            fen = (row.get("fen") or "").strip()
            if not image_path or not fen:
                continue

            overrides_by_path[_normalize_path_key(image_path)] = fen
            overrides_by_name[os.path.basename(image_path).lower()] = fen

    return overrides_by_path, overrides_by_name


def _find_manual_override(image_path, overrides_by_path, overrides_by_name):
    by_path = overrides_by_path.get(_normalize_path_key(image_path))
    if by_path:
        return by_path
    return overrides_by_name.get(os.path.basename(image_path).lower())


def save_manual_fen_override(
    image_path,
    fen,
    manual_overrides_csv="fen_overrides.csv",
):
    if not image_path:
        raise ValueError("image_path est requis")
    if not fen:
        raise ValueError("fen est requis")

    normalized_path = _normalize_path_key(image_path)
    rows = []
    found = False

    if os.path.isfile(manual_overrides_csv):
        with open(manual_overrides_csv, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                current_path = (row.get("image_path") or "").strip()
                current_fen = (row.get("fen") or "").strip()
                if not current_path:
                    continue

                if _normalize_path_key(current_path) == normalized_path:
                    rows.append({"image_path": os.path.abspath(image_path), "fen": fen})
                    found = True
                else:
                    rows.append({"image_path": current_path, "fen": current_fen})

    if not found:
        rows.append({"image_path": os.path.abspath(image_path), "fen": fen})

    with open(manual_overrides_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "fen"])
        writer.writeheader()
        writer.writerows(rows)


def _format_prediction_status(fen, meta, min_confidence, min_quality_score):
    quality_score = meta["quality_score"]
    avg_confidence = meta["avg_confidence"]
    stability_count = int(meta.get("stability_count", 0))
    fallback_used = bool(meta.get("fallback_used", 0))

    if not fen:
        return "  -> WARNING: no FEN produced"
    if fallback_used:
        return (
            "  -> FEN (fallback): "
            f"{fen} "
            f"(quality={quality_score}, conf={avg_confidence:.3f}, "
            f"stability={stability_count})"
        )
    if quality_score < min_quality_score or avg_confidence < min_confidence:
        return (
            "  -> FEN (low confidence): "
            f"{fen} "
            f"(quality={quality_score}, conf={avg_confidence:.3f}, "
            f"stability={stability_count})"
        )
    return (
        f"  -> FEN: {fen} "
        f"(quality={quality_score}, conf={avg_confidence:.3f}, "
        f"stability={stability_count})"
    )


def _convert_single_image_to_fen(image_path, predictor, use_filters):
    board_image, board_image_message = _load_board_image(image_path)
    if board_image is None:
        return "", None, board_image_message

    fen, meta = _predict_best_fen(board_image, predictor, use_filters=use_filters)
    return fen, meta, board_image_message


def convert_images_to_fen(
    images_dir=".",
    output_csv="fen_results.csv",
    min_confidence=0.95,
    min_quality_score=8,
    min_stability_count=4,
    strict_mode=False,
    use_filters=True,
    manual_overrides_csv="fen_overrides.csv",
    output_full_fen=False,
    side_to_move="w",
):
    if not os.path.isdir(images_dir):
        raise ValueError(f"Le dossier d'images n'existe pas : {images_dir}")

    image_paths = _list_image_paths(images_dir)
    if not image_paths:
        print(f"No image found in folder: {images_dir}")
        return

    print("Loading chessimg2pos model...")
    predictor = _load_predictor()
    overrides_by_path, overrides_by_name = _load_manual_overrides(manual_overrides_csv)

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "fen"])

        for image_path in image_paths:
            print(f"\nProcessing: {image_path}")
            try:
                manual_fen = _find_manual_override(image_path, overrides_by_path, overrides_by_name)
                if manual_fen:
                    writer.writerow([image_path, manual_fen])
                    print("  -> FEN override from manual file")
                    continue

                fen, meta, board_image_message = _convert_single_image_to_fen(
                    image_path,
                    predictor,
                    use_filters=use_filters,
                )
                if board_image_message:
                    print(board_image_message)
                if meta is None:
                    writer.writerow([image_path, ""])
                    continue

                quality_score = meta["quality_score"]
                avg_confidence = meta["avg_confidence"]
                stability_count = int(meta.get("stability_count", 0))

                if strict_mode and (
                    not fen
                    or quality_score < min_quality_score
                    or avg_confidence < min_confidence
                    or stability_count < min_stability_count
                ):
                    print(
                        "  -> Rejected in strict mode "
                        f"(quality={quality_score}, conf={avg_confidence:.3f}, "
                        f"stability={stability_count})"
                    )
                    fen = ""
                else:
                    print(
                        _format_prediction_status(
                            fen=fen,
                            meta=meta,
                            min_confidence=min_confidence,
                            min_quality_score=min_quality_score,
                        )
                    )

                if fen and output_full_fen:
                    fen = _to_full_fen(fen, side_to_move=side_to_move)

                writer.writerow([image_path, fen])
            except Exception as error:
                writer.writerow([image_path, ""])
                print(f"  -> ERROR: {error}")

    print(f"\nDone. Results saved to: {output_csv}")
