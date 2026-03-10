from collections import defaultdict
from typing import DefaultDict, List

try:
    import chess
except ImportError:
    chess = None


def _expand_fen_rows(fen):
    rows = fen.strip().split("/")
    if len(rows) != 8:
        return None

    expanded_rows = []
    for row in rows:
        expanded = []
        for char in row:
            if char.isdigit():
                expanded.extend("1" * int(char))
            elif char.isalpha():
                expanded.append(char)
            else:
                return None
        if len(expanded) != 8:
            return None
        expanded_rows.append("".join(expanded))
    return expanded_rows


def _compress_fen_row(expanded_row):
    compressed = []
    empty_count = 0
    for char in expanded_row:
        if char == "1":
            empty_count += 1
            continue
        if empty_count:
            compressed.append(str(empty_count))
            empty_count = 0
        compressed.append(char)

    if empty_count:
        compressed.append(str(empty_count))

    return "".join(compressed)


def _rotate_fen_180(fen):
    expanded_rows = _expand_fen_rows(fen)
    if expanded_rows is None:
        return fen

    rotated_rows = [row[::-1] for row in expanded_rows[::-1]]
    return "/".join(_compress_fen_row(row) for row in rotated_rows)


def _fen_to_grid(fen):
    rows = _expand_fen_rows(fen)
    if rows is None:
        return None
    return [list(row) for row in rows]


def _grid_to_fen(grid):
    rows = ["".join(row) for row in grid]
    return "/".join(_compress_fen_row(row) for row in rows)


def _basic_fen_sanity(fen):
    rows = _expand_fen_rows(fen)
    if rows is None:
        return False

    board = "".join(rows)
    if board.count("K") != 1 or board.count("k") != 1:
        return False
    if board.count("P") > 8 or board.count("p") > 8:
        return False
    if "P" in rows[0] or "P" in rows[7] or "p" in rows[0] or "p" in rows[7]:
        return False

    white_pieces = sum(1 for piece in board if piece.isalpha() and piece.isupper())
    black_pieces = sum(1 for piece in board if piece.isalpha() and piece.islower())
    if white_pieces > 16 or black_pieces > 16:
        return False

    return True


def _rotate_grid_ccw(grid):
    return [[grid[file_index][7 - rank_index] for file_index in range(8)] for rank_index in range(8)]


def _rotate_fen_k_ccw(fen, k):
    k = k % 4
    grid = _fen_to_grid(fen)
    if grid is None:
        return fen

    for _ in range(k):
        grid = _rotate_grid_ccw(grid)
    return _grid_to_fen(grid)


def _to_full_fen(fen, side_to_move="w"):
    parts = fen.strip().split()
    if len(parts) == 1:
        return f"{parts[0]} {side_to_move} - - 0 1"

    board = parts[0]
    turn = parts[1] if len(parts) > 1 else side_to_move
    castling = parts[2] if len(parts) > 2 else "-"
    en_passant = parts[3] if len(parts) > 3 else "-"
    halfmove = parts[4] if len(parts) > 4 else "0"
    fullmove = parts[5] if len(parts) > 5 else "1"
    return f"{board} {turn} {castling} {en_passant} {halfmove} {fullmove}"


def _is_valid_piece_placement(fen, side_to_move="w"):
    if not _basic_fen_sanity(fen):
        return False
    if chess is None:
        return True

    try:
        board = chess.Board(_to_full_fen(fen, side_to_move=side_to_move))
    except Exception:
        return False

    if hasattr(board, "is_valid") and not board.is_valid():
        return False

    return True


def _fix_last_rank_pawns(fen):
    grid = _fen_to_grid(fen)
    if grid is None:
        return fen

    flat = [piece for row in grid for piece in row]
    black_bishops = flat.count("b")
    white_bishops = flat.count("B")

    def _promo_for_black():
        return "b" if black_bishops < 2 else "q"

    def _promo_for_white():
        return "B" if white_bishops < 2 else "Q"

    changed = False
    for file_index in range(8):
        top_piece = grid[0][file_index]
        bottom_piece = grid[7][file_index]

        if top_piece == "p":
            promoted = _promo_for_black()
            grid[0][file_index] = promoted
            if promoted == "b":
                black_bishops += 1
            changed = True
        elif top_piece == "P":
            promoted = _promo_for_white()
            grid[0][file_index] = promoted
            if promoted == "B":
                white_bishops += 1
            changed = True

        if bottom_piece == "p":
            promoted = _promo_for_black()
            grid[7][file_index] = promoted
            if promoted == "b":
                black_bishops += 1
            changed = True
        elif bottom_piece == "P":
            promoted = _promo_for_white()
            grid[7][file_index] = promoted
            if promoted == "B":
                white_bishops += 1
            changed = True

    return _grid_to_fen(grid) if changed else fen


VoteMap = List[DefaultDict[str, float]]


def _init_votes():
    return [defaultdict(float) for _ in range(64)]


def _apply_fen_votes(votes, fen, weight):
    rows = _expand_fen_rows(fen)
    if rows is None:
        return

    for rank_index in range(8):
        for file_index in range(8):
            label = rows[rank_index][file_index]
            votes[rank_index * 8 + file_index][label] += weight


def _votes_to_fen(votes):
    rows = []
    for rank_index in range(8):
        row_chars = []
        for file_index in range(8):
            cell_votes = votes[rank_index * 8 + file_index]
            if not cell_votes:
                row_chars.append("1")
                continue
            best_label = max(cell_votes.items(), key=lambda item: item[1])[0]
            row_chars.append(best_label)

        rows.append(_compress_fen_row("".join(row_chars)))

    return "/".join(rows)


def _looks_like_black_view(fen):
    expanded_rows = _expand_fen_rows(fen)
    if expanded_rows is None:
        return False

    top_rows = "".join(expanded_rows[:2])
    bottom_rows = "".join(expanded_rows[-2:])
    white_top = sum(char.isupper() for char in top_rows)
    white_bottom = sum(char.isupper() for char in bottom_rows)
    black_top = sum(char.islower() for char in top_rows)
    black_bottom = sum(char.islower() for char in bottom_rows)
    return (white_top - white_bottom >= 2) and (black_bottom - black_top >= 2)


def _score_fen_plausibility(fen):
    expanded_rows = _expand_fen_rows(fen)
    if expanded_rows is None:
        return 0

    board = "".join(expanded_rows)
    score = 0

    def _promotion_excess_score(pawn_char, queen_char, rook_char, bishop_char, knight_char):
        pawns = board.count(pawn_char)
        missing_pawns = max(0, 8 - pawns)
        promotions_used = (
            max(0, board.count(queen_char) - 1)
            + max(0, board.count(rook_char) - 2)
            + max(0, board.count(bishop_char) - 2)
            + max(0, board.count(knight_char) - 2)
        )

        if promotions_used == 0:
            return 1
        if promotions_used <= missing_pawns:
            return 0
        return -2

    if board.count("K") == 1 and board.count("k") == 1:
        score += 4

    piece_count = sum(char.isalpha() for char in board)
    if piece_count <= 32:
        score += 1
    if board.count("P") <= 8 and board.count("p") <= 8:
        score += 1
    if any(char in "RNBQP" for char in board) and any(char in "rnbqp" for char in board):
        score += 1
    if all(char not in "Pp" for char in (expanded_rows[0] + expanded_rows[-1])):
        score += 1

    score += _promotion_excess_score("P", "Q", "R", "B", "N")
    score += _promotion_excess_score("p", "q", "r", "b", "n")
    return score


def _fen_piece_count(fen):
    expanded_rows = _expand_fen_rows(fen)
    if expanded_rows is None:
        return 0
    return sum(char.isalpha() for row in expanded_rows for char in row)
