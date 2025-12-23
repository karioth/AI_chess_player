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

PIECE_TO_PLANE = {
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


def board_to_matrix(board: chess.Board) -> torch.Tensor:
    """
    Returns an 8x8x16 tensor:
      - planes 0..11: piece one-hot
      - planes 12..15: unused (zeros)
    """
    mat = torch.zeros((8, 8, 16), dtype=torch.float32, device=device)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        plane = PIECE_TO_PLANE[(piece.color, piece.piece_type)]
        r = chess.square_rank(sq)
        f = chess.square_file(sq)
        mat[r, f, plane] = 1.0
    return mat


def add_state_vector(board_matrix: torch.Tensor, board: chess.Board) -> torch.Tensor:
    """
    Prepends a 1x16 global vector:
      [turn, WK, WQ, BK, BQ, ep_present, ep_file(8), halfmove_norm, fullmove_norm]
    """
    global_vec = torch.zeros((1, 16), dtype=torch.float32, device=device)
    global_vec[0, 0] = 1.0 if board.turn == chess.WHITE else 0.0
    global_vec[0, 1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    global_vec[0, 2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    global_vec[0, 3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    global_vec[0, 4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    if board.ep_square is not None:
        global_vec[0, 5] = 1.0
        ep_file = chess.square_file(board.ep_square)
        global_vec[0, 6 + ep_file] = 1.0

    global_vec[0, 14] = min(board.halfmove_clock, 150) / 150.0
    global_vec[0, 15] = min(board.fullmove_number, 200) / 200.0

    flat = board_matrix.view(64, 16)
    return torch.cat([global_vec, flat], dim=0)


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
        self.agent_won = None

        self.current_agent_color = self.board.turn
        self.color_str = "WHITE" if self.current_agent_color == chess.WHITE else "BLACK"

    # ────────────────────────────────────────────────────────────────────
    def step(self, action: int):
        reward, done, illegal = self.play_move(action)
        next_state = self.update()
        return next_state, reward, done, {"illegal": illegal}

    # ────────────────────────────────────────────────────────────────────
    def update(self):
        mat = board_to_matrix(self.board)
        return add_state_vector(mat, self.board)

    # ────────────────────────────────────────────────────────────────────
    def play_move(self, action: int):
        if self.game_over:
            self.reset()
            return 0.0, True, False

        self.last_move_was_legal = True
        moved_color = self.current_agent_color

        move = index_to_move(action, self.board)
        if move not in self.board.legal_moves:
            self.last_move_was_legal = False
            self.game_over = True
            self.agent_won = False
            self.game_over_reason = "illegal_move"
            return -1.0, True, True

        self.board.push(move)
        outcome = self.board.outcome(claim_draw=False)
        if outcome is not None:
            self.game_over = True
            self.game_over_reason = outcome.termination.name.lower()
            if outcome.winner is None:
                self.agent_won = None
                return 0.0, True, False
            if outcome.winner == moved_color:
                self.agent_won = True
                return 1.0, True, False
            self.agent_won = False
            return -1.0, True, False

        self.current_agent_color = self.board.turn
        self.color_str = "WHITE" if self.current_agent_color == chess.WHITE else "BLACK"
        return 0.0, False, False

    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def process_moves(games, actions):
        rewards, dones, illegal_moves = [], [], []
        agent_wons, reasons, color_strs = [], [], []
        valid_moves_count = 0

        for g, act in zip(games, actions):
            r, d, ill = g.play_move(act)
            rewards.append(r)
            dones.append(d)
            illegal_moves.append(ill)
            agent_wons.append(g.agent_won)
            reasons.append(g.game_over_reason)
            color_strs.append(g.color_str)
            if g.last_move_was_legal:
                valid_moves_count += 1
            if d:
                g.reset()

        return (
            games, rewards, dones, agent_wons,
            reasons, color_strs, valid_moves_count, illegal_moves
        )


def states_board_and_masks(games, device='mps'):
    """
    Returns:
      states_tensor: [batch, 65, 16]
      boards: list[chess.Board]
      masks_tensor: [batch, 4672]
    """
    states_tensor = torch.stack([g.update() for g in games]).float().to(device)
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
    return states_tensor, boards, masks_tensor
