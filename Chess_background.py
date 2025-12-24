import chess
import torch

# ──────────────────────────────────────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

PLANES_PER_SQUARE = 73
ACTION_SIZE = 64 * PLANES_PER_SQUARE  # 4672

# Queen-like directions: N, NE, E, SE, S, SW, W, NW
SLIDE_DIRS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
]
KNIGHT_DIRS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]
UNDERPROMOTION_PIECES = [chess.ROOK, chess.BISHOP, chess.KNIGHT]
UNDERPROMOTION_DF = [0, -1, 1]  # forward, diag-left, diag-right (by df)

PIECE_TO_ID = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _on_board(file_idx: int, rank_idx: int) -> bool:
    return 0 <= file_idx < 8 and 0 <= rank_idx < 8


def board_to_piece_ids(board: chess.Board, device_override=None) -> torch.Tensor:
    """
    Returns a length-64 tensor of piece IDs (0 = empty, 1..12 = pieces).
    """
    dev = device_override or device
    ids = torch.zeros((64,), dtype=torch.long, device=dev)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        ids[sq] = PIECE_TO_ID[(piece.color, piece.piece_type)] + 1
    return ids


def halfmove_to_bucket(halfmove_clock: int) -> int:
    if halfmove_clock <= 5:
        return halfmove_clock
    if halfmove_clock <= 9:
        return 6
    if halfmove_clock <= 19:
        return 7
    if halfmove_clock <= 39:
        return 8
    if halfmove_clock <= 59:
        return 9
    if halfmove_clock <= 79:
        return 10
    if halfmove_clock <= 99:
        return 11
    return 12


def build_global_ids(board: chess.Board, repetition_count: int, device_override=None):
    """
    Returns categorical IDs for global state:
      side_id (0/1),
      castle_bits (4,),
      ep_id (0..8),
      hmc_bucket (0..12),
      rep_bucket (0..3).
    """
    dev = device_override or device
    side_id = torch.tensor(
        0 if board.turn == chess.WHITE else 1,
        dtype=torch.long,
        device=dev,
    )
    castle_bits = torch.tensor(
        [
            int(board.has_kingside_castling_rights(chess.WHITE)),
            int(board.has_queenside_castling_rights(chess.WHITE)),
            int(board.has_kingside_castling_rights(chess.BLACK)),
            int(board.has_queenside_castling_rights(chess.BLACK)),
        ],
        dtype=torch.long,
        device=dev,
    )

    ep_id = 0
    if board.ep_square is not None:
        ep_id = chess.square_file(board.ep_square) + 1
    ep_id = torch.tensor(ep_id, dtype=torch.long, device=dev)

    hmc_bucket = torch.tensor(
        halfmove_to_bucket(min(board.halfmove_clock, 100)),
        dtype=torch.long,
        device=dev,
    )
    rep_bucket = torch.tensor(
        min(repetition_count, 3),
        dtype=torch.long,
        device=dev,
    )
    return side_id, castle_bits, ep_id, hmc_bucket, rep_bucket


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    from_sq = index // PLANES_PER_SQUARE
    plane = index % PLANES_PER_SQUARE
    fr = chess.square_rank(from_sq)
    ff = chess.square_file(from_sq)

    if plane < 56:
        dir_idx = plane // 7
        dist = (plane % 7) + 1
        df, dr = SLIDE_DIRS[dir_idx]
        to_f = ff + df * dist
        to_r = fr + dr * dist
        if not _on_board(to_f, to_r):
            return chess.Move.null()
        move = chess.Move(from_sq, chess.square(to_f, to_r))
        return _maybe_add_queen_promo(move, board)

    if plane < 64:
        df, dr = KNIGHT_DIRS[plane - 56]
        to_f = ff + df
        to_r = fr + dr
        if not _on_board(to_f, to_r):
            return chess.Move.null()
        move = chess.Move(from_sq, chess.square(to_f, to_r))
        return _maybe_add_queen_promo(move, board)

    if board is None:
        raise ValueError("board is required for underpromotion decoding")

    up_idx = plane - 64
    dir_idx = up_idx // 3
    piece_idx = up_idx % 3
    df = UNDERPROMOTION_DF[dir_idx]
    dr = 1 if board.turn == chess.WHITE else -1
    to_f = ff + df
    to_r = fr + dr
    if not _on_board(to_f, to_r):
        return chess.Move.null()
    promotion = UNDERPROMOTION_PIECES[piece_idx]
    return chess.Move(from_sq, chess.square(to_f, to_r), promotion=promotion)


def _maybe_add_queen_promo(move: chess.Move, board: chess.Board) -> chess.Move:
    if board is None:
        return move
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return move
    to_rank = chess.square_rank(move.to_square)
    if (piece.color == chess.WHITE and to_rank == 7) or \
       (piece.color == chess.BLACK and to_rank == 0):
        return chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    return move


def move_to_index(move: chess.Move) -> int | None:
    from_sq = move.from_square
    to_sq = move.to_square
    fr = chess.square_rank(from_sq)
    ff = chess.square_file(from_sq)
    tr = chess.square_rank(to_sq)
    tf = chess.square_file(to_sq)
    df = tf - ff
    dr = tr - fr

    if move.promotion and move.promotion != chess.QUEEN:
        if df not in (-1, 0, 1) or abs(dr) != 1:
            return None
        try:
            dir_idx = UNDERPROMOTION_DF.index(df)
            piece_idx = UNDERPROMOTION_PIECES.index(move.promotion)
        except ValueError:
            return None
        plane = 64 + dir_idx * 3 + piece_idx
        return from_sq * PLANES_PER_SQUARE + plane

    if (df, dr) in KNIGHT_DIRS:
        plane = 56 + KNIGHT_DIRS.index((df, dr))
        return from_sq * PLANES_PER_SQUARE + plane

    if df == 0 or dr == 0 or abs(df) == abs(dr):
        dir = (_sign(df), _sign(dr))
        dist = max(abs(df), abs(dr))
        if dist < 1 or dist > 7:
            return None
        try:
            dir_idx = SLIDE_DIRS.index(dir)
        except ValueError:
            return None
        plane = dir_idx * 7 + (dist - 1)
        return from_sq * PLANES_PER_SQUARE + plane

    return None


# ──────────────────────────────────────────────────────────────────────────
class ChessGame:
    """
    Markov chess environment (no hidden memory tokens, no state history).
    """
    def __init__(self, game_id: int = 0):
        self.game_id = game_id
        self.reset()

    # ────────────────────────────────────────────────────────────────────
    def reset(self):
        self.board = chess.Board()
        self.game_over = False
        self.game_over_reason = ""
        self.last_move_was_legal = True
        self.z_white = None
        self.winner_color = None

        self.current_agent_color = self.board.turn
        self.color_str = "WHITE" if self.current_agent_color == chess.WHITE else "BLACK"
        self._reset_repetition_counts()

    def _position_key(self):
        return self.board._transposition_key()

    def _reset_repetition_counts(self):
        key = self._position_key()
        self.repetition_counts = {key: 1}

    def _update_repetition_counts(self, irreversible: bool):
        key = self._position_key()
        if irreversible:
            self.repetition_counts = {key: 1}
        else:
            self.repetition_counts[key] = self.repetition_counts.get(key, 0) + 1

    def _current_repetition_count(self) -> int:
        key = self._position_key()
        return self.repetition_counts.get(key, 1)

    # ────────────────────────────────────────────────────────────────────
    def step(self, action: int):
        reward, done, illegal = self.play_move(action)
        next_state = self.update()
        return next_state, reward, done, {"illegal": illegal}

    # ────────────────────────────────────────────────────────────────────
    def update(self, device_override=None):
        repetition_count = self._current_repetition_count()
        piece_ids = board_to_piece_ids(self.board, device_override=device_override)
        global_ids = build_global_ids(
            self.board, repetition_count, device_override=device_override
        )
        return piece_ids, global_ids

    # ────────────────────────────────────────────────────────────────────
    def play_move(self, action: int):
        if self.game_over:
            z = self.z_white if self.z_white is not None else 0.0
            return z, True, False

        self.last_move_was_legal = True
        self.z_white = None
        self.winner_color = None
        moved_color = self.current_agent_color

        move = index_to_move(action, self.board)
        if move not in self.board.legal_moves:
            self.last_move_was_legal = False
            self.game_over = True
            self.game_over_reason = "illegal_move"
            self.z_white = -1.0 if moved_color == chess.WHITE else 1.0
            self.winner_color = chess.WHITE if self.z_white > 0 else chess.BLACK
            return self.z_white, True, True

        irreversible = self.board.is_irreversible(move)
        self.board.push(move)
        self._update_repetition_counts(irreversible)

        if self.board.can_claim_threefold_repetition():
            self.game_over = True
            self.game_over_reason = "threefold_repetition"
            self.z_white = 0.0
            self.winner_color = None
            return self.z_white, True, False

        if self.board.can_claim_fifty_moves():
            self.game_over = True
            self.game_over_reason = "fifty_move"
            self.z_white = 0.0
            self.winner_color = None
            return self.z_white, True, False

        outcome = self.board.outcome(claim_draw=False)
        if outcome is not None:
            self.game_over = True
            self.game_over_reason = outcome.termination.name.lower()
            if outcome.winner is None:
                self.z_white = 0.0
                self.winner_color = None
                return self.z_white, True, False
            self.z_white = 1.0 if outcome.winner == chess.WHITE else -1.0
            self.winner_color = chess.WHITE if self.z_white > 0 else chess.BLACK
            return self.z_white, True, False

        self.current_agent_color = self.board.turn
        self.color_str = "WHITE" if self.current_agent_color == chess.WHITE else "BLACK"
        return 0.0, False, False

    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def process_moves(games, actions, auto_reset: bool = True):
        rewards, dones, illegal_moves = [], [], []
        z_whites, winner_colors, reasons, color_strs = [], [], [], []
        valid_moves_count = 0

        for g, act in zip(games, actions):
            r, d, ill = g.play_move(act)
            rewards.append(r)
            dones.append(d)
            illegal_moves.append(ill)
            z_whites.append(g.z_white)
            winner_colors.append(g.winner_color)
            reasons.append(g.game_over_reason)
            color_strs.append(g.color_str)
            if g.last_move_was_legal:
                valid_moves_count += 1
            if d and auto_reset:
                g.reset()

        return (
            games, rewards, dones, z_whites, winner_colors,
            reasons, color_strs, valid_moves_count, illegal_moves
        )


def states_board_and_masks(games, device='mps'):
    """
    Returns:
      piece_ids_tensor: [batch, 64]
      global_state: (side, castle_bits, ep, hmc, rep)
      boards: list[chess.Board]
      masks_tensor: [batch, 4672]
    """
    piece_ids_list = []
    side_list = []
    castle_list = []
    ep_list = []
    hmc_list = []
    rep_list = []
    for g in games:
        piece_ids, global_ids = g.update(device_override=device)
        side_id, castle_bits, ep_id, hmc_bucket, rep_bucket = global_ids
        piece_ids_list.append(piece_ids)
        side_list.append(side_id)
        castle_list.append(castle_bits)
        ep_list.append(ep_id)
        hmc_list.append(hmc_bucket)
        rep_list.append(rep_bucket)
    piece_ids_tensor = torch.stack(piece_ids_list).to(device)
    global_state = (
        torch.stack(side_list).to(device),
        torch.stack(castle_list).to(device),
        torch.stack(ep_list).to(device),
        torch.stack(hmc_list).to(device),
        torch.stack(rep_list).to(device),
    )
    boards = [g.board for g in games]
    masks_list = []
    for board in boards:
        mask = torch.zeros(ACTION_SIZE, dtype=torch.bool, device=device)
        for mv in board.legal_moves:
            idx = move_to_index(mv)
            if idx is not None:
                mask[idx] = True
        masks_list.append(mask)
    masks_tensor = torch.stack(masks_list)
    return piece_ids_tensor, global_state, boards, masks_tensor
