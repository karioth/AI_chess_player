import argparse
import os

import torch
import torch.optim as optim

from Model import ChessModel
from learning import train, device as default_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train the chess agent with PPO.")
    parser.add_argument("--ckpt", default="chess_model_transformer_weights_exp2.pth")
    parser.add_argument("--log", default="grpo_stepwise_log.csv")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-int", type=int, default=50)
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
    run_device = default_device if args.device is None else torch.device(args.device)

    model = ChessModel().to(run_device)
    if args.resume and os.path.exists(args.ckpt):
        try:
            model.load_state_dict(torch.load(args.ckpt, map_location=run_device), strict=False)
            print("Model loaded.")
        except Exception as exc:
            print("Load error:", exc)

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
    )


if __name__ == "__main__":
    main()
