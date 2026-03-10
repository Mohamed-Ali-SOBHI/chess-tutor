import atexit
import threading

try:
    import chess
    import chess.engine
except ImportError:
    chess = None

from .constants import DEFAULT_ENGINE_ELO, DEFAULT_ENGINE_THINK_TIME
from .fen_utils import _to_full_fen
from .stockfish_manager import (
    iter_existing_stockfish_paths,
    iter_installable_stockfish_paths,
)

_ENGINE_CACHE = {}
_ENGINE_CACHE_LOCK = threading.Lock()


class _CachedEngineSession:
    def __init__(self, engine, applied_engine_elo):
        self.engine = engine
        self.applied_engine_elo = applied_engine_elo
        self.lock = threading.Lock()


def _resolve_stockfish_path(stockfish_path=None):
    for candidate_path in iter_existing_stockfish_paths(stockfish_path=stockfish_path):
        return candidate_path

    raise ValueError(
        "Stockfish introuvable. "
        "Le moteur sera installe automatiquement au premier calcul si aucun chemin n'est fourni."
    )


def _iter_stockfish_launch_candidates(
    stockfish_path=None,
    auto_install=True,
    progress_callback=None,
):
    yielded_paths = set()

    for candidate_path in iter_existing_stockfish_paths(stockfish_path=stockfish_path):
        if candidate_path in yielded_paths:
            continue
        yielded_paths.add(candidate_path)
        yield candidate_path

    if stockfish_path or not auto_install:
        return

    for candidate_path in iter_installable_stockfish_paths(
        progress_callback=progress_callback,
    ):
        if candidate_path in yielded_paths:
            continue
        yielded_paths.add(candidate_path)
        yield candidate_path


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


def _engine_cache_key(stockfish_path, engine_elo):
    return stockfish_path, int(engine_elo) if engine_elo is not None else None


def _discard_cached_engine(stockfish_path, engine_elo):
    entry = None
    with _ENGINE_CACHE_LOCK:
        entry = _ENGINE_CACHE.pop(_engine_cache_key(stockfish_path, engine_elo), None)

    if entry is None:
        return

    try:
        entry.engine.quit()
    except Exception:
        pass


def close_cached_engines():
    with _ENGINE_CACHE_LOCK:
        entries = list(_ENGINE_CACHE.values())
        _ENGINE_CACHE.clear()

    for entry in entries:
        try:
            entry.engine.quit()
        except Exception:
            pass


def _get_cached_engine_session(stockfish_path, engine_elo):
    cache_key = _engine_cache_key(stockfish_path, engine_elo)
    with _ENGINE_CACHE_LOCK:
        cached_entry = _ENGINE_CACHE.get(cache_key)
        if cached_entry is not None:
            return cached_entry

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    applied_engine_elo = _configure_engine_strength(engine, engine_elo)
    entry = _CachedEngineSession(engine=engine, applied_engine_elo=applied_engine_elo)
    with _ENGINE_CACHE_LOCK:
        previous_entry = _ENGINE_CACHE.get(cache_key)
        if previous_entry is not None:
            try:
                engine.quit()
            except Exception:
                pass
            return previous_entry

        _ENGINE_CACHE[cache_key] = entry
    return entry


def _play_with_engine(board, stockfish_path, think_time, engine_elo):
    last_error = None
    for _ in range(2):
        entry = _get_cached_engine_session(stockfish_path, engine_elo)
        with entry.lock:
            try:
                play_result = entry.engine.play(board, chess.engine.Limit(time=think_time))
                analysis = entry.engine.analyse(board, chess.engine.Limit(time=think_time))
                return play_result, analysis, entry.applied_engine_elo
            except (
                OSError,
                chess.engine.EngineError,
                chess.engine.EngineTerminatedError,
            ) as error:
                last_error = error
        _discard_cached_engine(stockfish_path, engine_elo)

    raise last_error


def get_best_move_from_fen(
    fen,
    stockfish_path=None,
    side_to_move="w",
    think_time=DEFAULT_ENGINE_THINK_TIME,
    engine_elo=DEFAULT_ENGINE_ELO,
    auto_install=True,
    progress_callback=None,
):
    if chess is None:
        raise ImportError(
            "python-chess n'est pas installe. Executez: pip install python-chess"
        )

    full_fen = _to_full_fen(fen, side_to_move=side_to_move)
    board = chess.Board(full_fen)

    last_engine_error = None
    resolved_stockfish_path = None

    for candidate_path in _iter_stockfish_launch_candidates(
        stockfish_path=stockfish_path,
        auto_install=auto_install,
        progress_callback=progress_callback,
    ):
        resolved_stockfish_path = candidate_path
        try:
            play_result, analysis, applied_engine_elo = _play_with_engine(
                board=board,
                stockfish_path=candidate_path,
                think_time=think_time,
                engine_elo=engine_elo,
            )
            break
        except (OSError, chess.engine.EngineError, chess.engine.EngineTerminatedError) as error:
            last_engine_error = error
    else:
        if stockfish_path:
            raise ValueError(f"Chemin Stockfish invalide ou inutilisable: {stockfish_path}")
        if last_engine_error is not None:
            raise RuntimeError(
                f"Impossible de lancer Stockfish automatiquement: {last_engine_error}"
            ) from last_engine_error
        raise RuntimeError("Aucun binaire Stockfish disponible")

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
        "side_to_move": side_to_move,
        "best_move_uci": move.uci(),
        "best_move_san": board.san(move),
        "score_cp": score_cp,
        "mate_in": score_mate,
        "engine_elo": applied_engine_elo,
        "stockfish_path": resolved_stockfish_path,
    }


atexit.register(close_cached_engines)
