import argparse
import os

import torch
import torch.optim as optim

from Model import ChessModel
from arena import run_arena
from checkpoints import load_checkpoint, save_checkpoint, save_history
from learning import train, device as default_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the chess agent with GRPO + champion gating."
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Deprecated alias for --candidate-ckpt.",
    )
    parser.add_argument(
        "--candidate-ckpt",
        default="checkpoints/candidate.pth",
        help="Latest candidate checkpoint path.",
    )
    parser.add_argument(
        "--champion-ckpt",
        default="checkpoints/champion.pth",
        help="Champion checkpoint path.",
    )
    parser.add_argument(
        "--history-dir",
        default="checkpoints/history",
        help="Optional history directory for periodic candidate snapshots.",
    )
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save candidate snapshots to --history-dir on checkpoint.",
    )
    parser.add_argument("--log", default="grpo_stepwise_log.csv")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-int", type=int, default=50)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-plies", type=int, default=4096)
    parser.add_argument("--max-plies", type=int, default=0,
                        help="Optional max plies per game (0 disables).")
    parser.add_argument("--eval-interval", type=int, default=50,
                        help="Run arena evaluation every N epochs (0 disables).")
    parser.add_argument("--arena-games", type=int, default=80)
    parser.add_argument("--promote-threshold", type=float, default=0.55)
    parser.add_argument("--arena-policy", choices=["argmax", "sample"], default="argmax")
    parser.add_argument("--arena-temperature", type=float, default=1.0)
    parser.add_argument("--arena-seed", type=int, default=0)
    parser.add_argument("--rollout-policy", choices=["latest", "champion"], default="latest",
                        help="Policy used for rollout generation.")
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
    candidate_ckpt = args.candidate_ckpt
    if args.ckpt:
        candidate_ckpt = args.ckpt
    run_device = default_device if args.device is None else torch.device(args.device)
    print("Using device:", run_device)
    model = ChessModel().to(run_device)
    if args.resume and os.path.exists(candidate_ckpt):
        try:
            model.load_state_dict(torch.load(candidate_ckpt, map_location=run_device), strict=False)
            print("Model loaded.")
        except Exception as exc:
            print("Load error:", exc)

    champion_model = ChessModel().to(run_device)
    champion_loaded = load_checkpoint(champion_model, args.champion_ckpt, run_device)
    if not champion_loaded:
        if load_checkpoint(champion_model, candidate_ckpt, run_device):
            print(f"Champion initialized from {candidate_ckpt}.")
        else:
            champion_model.load_state_dict(model.state_dict())
            print("Champion initialized from current model.")
        save_checkpoint(champion_model, args.champion_ckpt)

    def save_candidate(current_model, step):
        save_checkpoint(current_model, candidate_ckpt)
        if args.save_history:
            save_history(current_model, args.history_dir, step, prefix="candidate")

    def arena_callback(current_model, tag):
        results = run_arena(
            candidate_model=current_model,
            champion_model=champion_model,
            games=args.arena_games,
            run_device=run_device,
            policy=args.arena_policy,
            temperature=args.arena_temperature,
            seed=args.arena_seed + int(tag),
        )
        promoted = 0
        if results["score"] >= args.promote_threshold:
            champion_model.load_state_dict(current_model.state_dict())
            save_checkpoint(champion_model, args.champion_ckpt)
            promoted = 1
            print(
                f"[Arena] promoted at {tag} | score {results['score']:.3f} "
                f"(W/D/L {results['wins']}/{results['draws']}/{results['losses']})"
            )
        else:
            print(
                f"[Arena] kept champion at {tag} | score {results['score']:.3f} "
                f"(W/D/L {results['wins']}/{results['draws']}/{results['losses']})"
            )
        results["promoted"] = promoted
        return results

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    rollout_model = champion_model if args.rollout_policy == "champion" else None
    train(
        model,
        opt,
        candidate_ckpt,
        args.log,
        epochs=args.epochs,
        batch=args.batch,
        save_int=args.save_int,
        device_override=run_device,
        ppo_epochs=args.ppo_epochs,
        minibatch_plies=args.minibatch_plies,
        max_plies=args.max_plies if args.max_plies > 0 else None,
        rollout_model=rollout_model,
        eval_every=args.eval_interval if args.eval_interval > 0 else None,
        eval_fn=arena_callback if args.eval_interval > 0 else None,
        save_fn=save_candidate,
    )


if __name__ == "__main__":
    main()
