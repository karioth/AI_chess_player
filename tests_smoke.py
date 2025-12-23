import random

import chess
import torch

from Chess_background import (
    ACTION_SIZE,
    ChessGame,
    index_to_move,
    move_to_index,
    states_board_and_masks,
)
from Model import ChessModel

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def assert_legal_mask(board: chess.Board, mask: torch.Tensor) -> None:
    for idx in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        mv = index_to_move(idx, board)
        assert mv in board.legal_moves, (idx, mv.uci())


def test_model_shapes():
    model = ChessModel().to(DEVICE)
    states, _, _ = states_board_and_masks([ChessGame()], device=DEVICE)
    logits, value = model(states)
    assert logits.shape == (1, ACTION_SIZE), logits.shape
    assert value.shape == (1,), value.shape


def test_legal_mask_initial():
    game = ChessGame()
    _, boards, masks = states_board_and_masks([game], device=DEVICE)
    assert masks.shape == (1, ACTION_SIZE)
    assert_legal_mask(boards[0], masks[0])


def test_underpromotion_and_queen():
    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.E7, chess.Piece(chess.PAWN, chess.WHITE))
    board.turn = chess.WHITE

    q = chess.Move.from_uci("e7e8q")
    n = chess.Move.from_uci("e7e8n")
    idx_q = move_to_index(q)
    idx_n = move_to_index(n)
    assert idx_q is not None
    assert idx_n is not None
    mv_q = index_to_move(idx_q, board)
    mv_n = index_to_move(idx_n, board)
    assert mv_q.promotion == chess.QUEEN, mv_q
    assert mv_n.promotion == chess.KNIGHT, mv_n


def test_en_passant():
    board = chess.Board("rnbqkbnr/pppppppp/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 2")
    game = ChessGame()
    game.board = board
    _, boards, masks = states_board_and_masks([game], device=DEVICE)
    ep_move = chess.Move.from_uci("e5f6")
    idx = move_to_index(ep_move)
    assert idx is not None
    assert masks[0, idx].item() is True
    assert ep_move in boards[0].legal_moves


def test_castling():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    game = ChessGame()
    game.board = board
    _, boards, masks = states_board_and_masks([game], device=DEVICE)
    wk = chess.Move.from_uci("e1g1")
    wq = chess.Move.from_uci("e1c1")
    assert masks[0, move_to_index(wk)].item() is True
    assert masks[0, move_to_index(wq)].item() is True
    assert wk in boards[0].legal_moves
    assert wq in boards[0].legal_moves

    board.turn = chess.BLACK
    _, boards, masks = states_board_and_masks([game], device=DEVICE)
    bk = chess.Move.from_uci("e8g8")
    bq = chess.Move.from_uci("e8c8")
    assert masks[0, move_to_index(bk)].item() is True
    assert masks[0, move_to_index(bq)].item() is True
    assert bk in boards[0].legal_moves
    assert bq in boards[0].legal_moves


def test_random_rollout(plies: int = 200):
    game = ChessGame()
    for _ in range(plies):
        _, boards, masks = states_board_and_masks([game], device=DEVICE)
        mask = masks[0]
        legal_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        assert legal_indices, "no legal moves found"
        idx = random.choice(legal_indices)
        mv = index_to_move(idx, boards[0])
        assert mv in boards[0].legal_moves
        (games, rewards, dones, *_ ) = ChessGame.process_moves([game], [idx])
        game = games[0]
        if dones[0]:
            game = ChessGame()


if __name__ == "__main__":
    test_model_shapes()
    test_legal_mask_initial()
    test_underpromotion_and_queen()
    test_en_passant()
    test_castling()
    test_random_rollout()
    print("All tests passed.")
