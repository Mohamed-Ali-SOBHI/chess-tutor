import os
import shutil

try:
    import chess
    import chess.engine
except ImportError:
    chess = None

from .constants import DEFAULT_ENGINE_ELO
from .fen_utils import _to_full_fen


def _resolve_stockfish_path(stockfish_path=None):
    candidate_paths = []
    if stockfish_path:
        candidate_paths.append(stockfish_path)

    for env_var in ("STOCKFISH_PATH", "STOCKFISH_EXECUTABLE"):
        env_value = os.environ.get(env_var)
        if env_value:
            candidate_paths.append(env_value)

    which_stockfish = shutil.which("stockfish") or shutil.which("stockfish.exe")
    if which_stockfish:
        candidate_paths.append(which_stockfish)

    for candidate_path in candidate_paths:
        normalized_path = os.path.abspath(candidate_path)
        if os.path.isfile(normalized_path):
            return normalized_path

    raise ValueError(
        "Chemin Stockfish invalide. "
        "Passez `stockfish_path`, ou definissez `STOCKFISH_PATH`."
    )


def _configure_engine_strength(engine, engine_elo):
    if engine_elo is None:
        return None

    configured_options = {}
    engine_options = getattr(engine, "options", {})
    if "UCI_LimitStrength" in engine_options:
        configured_options["UCI_LimitStrength"] = True
    if "UCI_Elo" in engine_options:
        configured_options["UCI_Elo"] = int(engine_elo)

    if configured_options:
        engine.configure(configured_options)
        return int(engine_elo)

    return None


def get_best_move_from_fen(
    fen,
    stockfish_path=None,
    side_to_move="w",
    think_time=0.20,
    engine_elo=DEFAULT_ENGINE_ELO,
):
    if chess is None:
        raise ImportError(
            "python-chess n'est pas installe. Executez: pip install python-chess"
        )

    resolved_stockfish_path = _resolve_stockfish_path(stockfish_path)
    full_fen = _to_full_fen(fen, side_to_move=side_to_move)
    board = chess.Board(full_fen)

    with chess.engine.SimpleEngine.popen_uci(resolved_stockfish_path) as engine:
        applied_engine_elo = _configure_engine_strength(engine, engine_elo)
        play_result = engine.play(board, chess.engine.Limit(time=think_time))
        analysis = engine.analyse(board, chess.engine.Limit(time=think_time))

    move = play_result.move
    if move is None:
        raise RuntimeError("Stockfish did not return a best move")

    score_obj = analysis.get("score")
    score_cp = None
    score_mate = None
    if score_obj is not None:
        pov_score = score_obj.pov(board.turn)
        score_cp = pov_score.score(mate_score=100000)
        score_mate = pov_score.mate()

    return {
        "fen": fen,
        "full_fen": full_fen,
        "best_move_uci": move.uci(),
        "best_move_san": board.san(move),
        "score_cp": score_cp,
        "mate_in": score_mate,
        "engine_elo": applied_engine_elo,
        "stockfish_path": resolved_stockfish_path,
    }
