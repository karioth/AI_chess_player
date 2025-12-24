import argparse
import csv
import math
import os
import random
import shutil

import chess
import chess.engine
import torch

from Chess_background import ChessGame, move_to_index, states_board_and_masks
from Model import ChessModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a chess model.")
    parser.add_argument("--ckpt", default="chess_model_transformer_weights_exp2.pth")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--opponent", choices=["random", "engine", "checkpoint"], default="random")
    parser.add_argument("--opponent-ckpt", default=None)
    parser.add_argument("--engine-path", default=None)
    parser.add_argument("--engine-time", type=float, default=0.05)
    parser.add_argument("--engine-depth", type=int, default=None)
    parser.add_argument("--model-color", choices=["white", "black", "both"], default="both")
    parser.add_argument("--policy", choices=["argmax", "sample"], default="argmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-csv", default=None)
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=None,
        help="Override device selection (default: auto)",
    )
    return parser.parse_args()


def elo_from_score(score: float) -> float:
    eps = 1e-6
    score = min(max(score, eps), 1 - eps)
    return -400.0 * math.log10(1 / score - 1)


def select_random_action(mask: torch.Tensor) -> int:
    legal_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    return random.choice(legal_indices)


def select_model_action(model, piece_ids, global_state, mask, policy, temperature):
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
    if not mask[0, action]:
        action = select_random_action(mask[0])
    return action


def select_engine_action(engine, board, mask, limit):
    result = engine.play(board, limit)
    idx = move_to_index(result.move)
    if idx is None or not mask[0, idx]:
        idx = select_random_action(mask[0])
    return idx


def outcome_from_move(game: ChessGame, moved_color: bool, model_color: bool):
    if game.agent_won is None:
        return "draw"
    if game.agent_won and moved_color == model_color:
        return "win"
    return "loss"


def resolve_engine_path(engine_path: str | None) -> str | None:
    if engine_path:
        return engine_path
    return shutil.which("stockfish")


def run_eval(model: ChessModel,
             run_device: torch.device,
             games: int = 20,
             opponent: str = "random",
             opponent_model: ChessModel | None = None,
             engine_path: str | None = None,
             engine_time: float = 0.05,
             engine_depth: int | None = None,
             model_color: str = "both",
             policy: str = "argmax",
             temperature: float = 1.0,
             seed: int = 0,
             log_csv: str | None = None):
    random.seed(seed)
    torch.manual_seed(seed)

    engine = None
    limit = None
    if opponent == "engine":
        resolved = resolve_engine_path(engine_path)
        if not resolved:
            raise SystemExit("Stockfish not found. Install it or pass --engine-path.")
        engine = chess.engine.SimpleEngine.popen_uci(resolved)
        if engine_depth is not None:
            limit = chess.engine.Limit(depth=engine_depth)
        else:
            limit = chess.engine.Limit(time=engine_time)
    elif opponent == "checkpoint":
        if opponent_model is None:
            raise SystemExit("--opponent-ckpt is required when --opponent=checkpoint")
        opponent_model.to(run_device).eval()

    wins = draws = losses = 0
    total_plies = 0

    log_writer = None
    log_file = None
    if log_csv:
        log_file = open(log_csv, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["game", "model_color", "result", "plies"])

    try:
        for i in range(games):
            game = ChessGame()
            if model_color == "both":
                model_side = chess.WHITE if i % 2 == 0 else chess.BLACK
            elif model_color == "white":
                model_side = chess.WHITE
            else:
                model_side = chess.BLACK

            plies = 0
            while True:
                piece_ids, global_state, _, mask = states_board_and_masks([game], device=run_device)
                moved_color = game.current_agent_color
                if moved_color == model_side:
                    action = select_model_action(
                        model, piece_ids, global_state, mask,
                        policy, temperature
                    )
                else:
                    if opponent == "engine":
                        action = select_engine_action(engine, game.board, mask, limit)
                    elif opponent == "checkpoint":
                        action = select_model_action(
                            opponent_model, piece_ids, global_state, mask,
                            policy, temperature
                        )
                    else:
                        action = select_random_action(mask[0])

                _, done, illegal = game.play_move(action)
                if illegal:
                    done = True
                    game.agent_won = False if moved_color == model_side else True

                plies += 1
                if done:
                    result = outcome_from_move(game, moved_color, model_side)
                    if result == "win":
                        wins += 1
                    elif result == "loss":
                        losses += 1
                    else:
                        draws += 1
                    total_plies += plies
                    if log_writer:
                        color_str = "white" if model_side == chess.WHITE else "black"
                        log_writer.writerow([i, color_str, result, plies])
                    break
    finally:
        if engine is not None:
            engine.close()
        if log_file is not None:
            log_file.close()

    games_played = wins + draws + losses
    score = (wins + 0.5 * draws) / max(games_played, 1)
    elo = elo_from_score(score)
    avg_len = total_plies / max(games_played, 1)
    return {
        "games": games_played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "elo_diff": elo,
        "avg_plies": avg_len,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    )

    model = ChessModel().to(run_device)
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=run_device), strict=False)
    model.eval()

    opponent_model = None
    if args.opponent == "checkpoint":
        if not args.opponent_ckpt:
            raise SystemExit("--opponent-ckpt is required when --opponent=checkpoint")
        opponent_model = ChessModel().to(run_device)
        opponent_model.load_state_dict(
            torch.load(args.opponent_ckpt, map_location=run_device), strict=False
        )
        opponent_model.eval()

    results = run_eval(
        model=model,
        run_device=run_device,
        games=args.games,
        opponent=args.opponent,
        opponent_model=opponent_model,
        engine_path=args.engine_path,
        engine_time=args.engine_time,
        engine_depth=args.engine_depth,
        model_color=args.model_color,
        policy=args.policy,
        temperature=args.temperature,
        seed=args.seed,
        log_csv=args.log_csv,
    )

    print(f"Games: {results['games']} | W/D/L: {results['wins']}/{results['draws']}/{results['losses']}")
    print(f"Score: {results['score']:.3f} | Elo diff vs opponent: {results['elo_diff']:+.1f}")
    print(f"Avg plies: {results['avg_plies']:.1f}")


if __name__ == "__main__":
    main()
