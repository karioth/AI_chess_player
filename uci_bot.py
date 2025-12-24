#!/usr/bin/env python3
import argparse
import sys

import chess
import torch

from Chess_background import (
    ACTION_SIZE,
    board_to_piece_ids,
    build_global_ids,
    index_to_move,
    move_to_index,
)
from Model import ChessModel


def parse_args():
    parser = argparse.ArgumentParser(description="UCI shim for the chess model.")
    parser.add_argument("--ckpt", default="chess_model_transformer_weights_exp2.pth")
    parser.add_argument("--name", default="ChessBot")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--policy", choices=["argmax", "sample"], default="argmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_position(tokens, board: chess.Board):
    if not tokens:
        return board
    idx = 0
    if tokens[idx] == "startpos":
        board.set_fen(chess.STARTING_FEN)
        idx += 1
    elif tokens[idx] == "fen":
        fen = " ".join(tokens[idx + 1:idx + 7])
        board.set_fen(fen)
        idx += 7
    if idx < len(tokens) and tokens[idx] == "moves":
        for mv in tokens[idx + 1:]:
            board.push_uci(mv)
    return board


def repetition_count(board: chess.Board) -> int:
    if board.is_repetition(3):
        return 3
    if board.is_repetition(2):
        return 2
    return 1


def build_inputs(board: chess.Board, device: torch.device):
    piece_ids = board_to_piece_ids(board, device_override=device).unsqueeze(0)
    rep_count = repetition_count(board)
    side, castle_bits, ep, hmc, rep = build_global_ids(
        board, rep_count, device_override=device
    )
    global_state = (
        side.unsqueeze(0),
        castle_bits.unsqueeze(0),
        ep.unsqueeze(0),
        hmc.unsqueeze(0),
        rep.unsqueeze(0),
    )
    mask = torch.zeros((1, ACTION_SIZE), dtype=torch.bool, device=device)
    for mv in board.legal_moves:
        idx = move_to_index(mv)
        if idx is not None:
            mask[0, idx] = True
    return piece_ids, global_state, mask


def choose_move(model, board: chess.Board, device, policy, temperature):
    piece_ids, global_state, mask = build_inputs(board, device)
    with torch.no_grad():
        logits, _ = model(piece_ids, global_state, mask)
    logits = logits[0]

    if policy == "sample":
        temp = max(temperature, 1e-3)
        probs = torch.softmax(logits / temp, dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
    else:
        action = torch.argmax(logits).item()

    mv = index_to_move(action, board)
    if mv not in board.legal_moves:
        mv = next(iter(board.legal_moves))
    return mv


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    )

    model = ChessModel().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model.eval()

    board = chess.Board()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        cmd = parts[0]

        if cmd == "uci":
            print(f"id name {args.name}")
            print("id author codex")
            print("uciok")
        elif cmd == "isready":
            print("readyok")
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "position":
            board = parse_position(parts[1:], board)
        elif cmd == "go":
            move = choose_move(model, board, device, args.policy, args.temperature)
            print(f"bestmove {move.uci()}")
        elif cmd == "quit":
            break
        else:
            pass

        sys.stdout.flush()


if __name__ == "__main__":
    main()
