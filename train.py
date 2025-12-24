import argparse
import csv
import os

import torch
import torch.optim as optim

from Model import ChessModel
from eval import run_eval
from learning import train, device as default_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train the chess agent with GRPO.")
    parser.add_argument("--ckpt", default="chess_model_transformer_weights_exp2.pth")
    parser.add_argument("--log", default="grpo_stepwise_log.csv")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-int", type=int, default=50)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-plies", type=int, default=4096)
    parser.add_argument("--max-plies", type=int, default=0,
                        help="Optional max plies per game (0 disables).")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Run evaluation every N epochs (0 disables).")
    parser.add_argument("--eval-on-checkpoint", action="store_true",
                        help="Run evaluation every time a checkpoint is saved.")
    parser.add_argument("--no-eval-on-checkpoint", action="store_true",
                        help="Disable checkpoint-triggered evaluation.")
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--eval-opponent", choices=["random", "engine", "checkpoint"], default="checkpoint")
    parser.add_argument("--eval-opponent-ckpt", default=None,
                        help="Checkpoint path for eval-opponent=checkpoint.")
    parser.add_argument("--eval-engine-path", default=None)
    parser.add_argument("--eval-engine-time", type=float, default=0.05)
    parser.add_argument("--eval-engine-depth", type=int, default=None)
    parser.add_argument("--eval-model-color", choices=["white", "black", "both"], default="both")
    parser.add_argument("--eval-policy", choices=["argmax", "sample"], default="argmax")
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--eval-log-csv", default=None,
                        help="Per-game log CSV path (supports {epoch}).")
    parser.add_argument("--eval-summary-csv", default=None,
                        help="Append summary rows per eval (supports {epoch}).")
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=None,
        help="Override device selection (default: auto)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load checkpoint if it exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_eval_on_checkpoint:
        args.eval_on_checkpoint = False
    elif args.eval_every == 0 and not args.eval_on_checkpoint:
        args.eval_on_checkpoint = True
    run_device = default_device if args.device is None else torch.device(args.device)
    print("Using device:", run_device)
    model = ChessModel().to(run_device)
    if args.resume and os.path.exists(args.ckpt):
        try:
            model.load_state_dict(torch.load(args.ckpt, map_location=run_device), strict=False)
            print("Model loaded.")
        except Exception as exc:
            print("Load error:", exc)

    def eval_callback(current_model, tag):
        log_csv = None
        if args.eval_log_csv:
            log_csv = args.eval_log_csv.format(epoch=tag)
        opponent_model = None
        opponent_ckpt_path = None
        if args.eval_opponent == "checkpoint":
            opponent_ckpt_path = args.eval_opponent_ckpt or f"{args.ckpt}.prev"
            if not os.path.exists(opponent_ckpt_path):
                torch.save(current_model.state_dict(), opponent_ckpt_path)
                print(f"[Eval] baseline created at {opponent_ckpt_path}, skipping eval.")
                return
            opponent_model = ChessModel().to(run_device)
            opponent_model.load_state_dict(
                torch.load(opponent_ckpt_path, map_location=run_device), strict=False
            )
            opponent_model.eval()

        results = run_eval(
            model=current_model,
            run_device=run_device,
            games=args.eval_games,
            opponent=args.eval_opponent,
            opponent_model=opponent_model,
            engine_path=args.eval_engine_path,
            engine_time=args.eval_engine_time,
            engine_depth=args.eval_engine_depth,
            model_color=args.eval_model_color,
            policy=args.eval_policy,
            temperature=args.eval_temperature,
            seed=args.eval_seed + int(tag),
            log_csv=log_csv,
        )
        print(
            f"[Eval] {tag} | W/D/L {results['wins']}/{results['draws']}/"
            f"{results['losses']} | Score {results['score']:.3f} | "
            f"Elo {results['elo_diff']:+.1f} | Plies {results['avg_plies']:.1f}"
        )
        if opponent_ckpt_path:
            torch.save(current_model.state_dict(), opponent_ckpt_path)
        if args.eval_summary_csv:
            summary_path = args.eval_summary_csv.format(epoch=tag)
            write_header = not os.path.exists(summary_path)
            with open(summary_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "epoch", "games", "wins", "draws", "losses",
                        "score", "elo_diff", "avg_plies"
                    ])
                writer.writerow([
                    tag, results["games"], results["wins"], results["draws"],
                    results["losses"], f"{results['score']:.6f}",
                    f"{results['elo_diff']:.2f}", f"{results['avg_plies']:.2f}"
                ])

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    train(
        model,
        opt,
        args.ckpt,
        args.log,
        epochs=args.epochs,
        batch=args.batch,
        save_int=args.save_int,
        device_override=run_device,
        ppo_epochs=args.ppo_epochs,
        minibatch_plies=args.minibatch_plies,
        max_plies=args.max_plies if args.max_plies > 0 else None,
        eval_every=args.eval_every if args.eval_every > 0 else None,
        eval_fn=eval_callback if (args.eval_every > 0 or args.eval_on_checkpoint) else None,
    )


if __name__ == "__main__":
    main()
