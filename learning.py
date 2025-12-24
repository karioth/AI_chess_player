# train_grpo.py
import os
import csv
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from Model import ChessModel
from Chess_background import ChessGame, states_board_and_masks

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ───────── Hyperparameters ─────────
CLIP_EPS = 0.2
ENT_COEF = 0.02
SAVE_INT = 50
BATCH = 64
PPO_EPOCHS = 2
MINIBATCH_PLIES = 4096
MAX_PLIES = None  # set to an int to truncate long games


def count_parameters(model: torch.nn.Module):
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {tot:,},  trainable: {trn:,}")


def compute_group_advantages(z_white: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = z_white.mean()
    std = z_white.std(unbiased=False)
    if std < eps:
        return torch.zeros_like(z_white)
    return (z_white - mean) / (std + eps)


def collect_rollouts(model_old: ChessModel,
                     num_games: int,
                     run_device: torch.device,
                     max_plies: int | None = None):
    envs = [ChessGame(i) for i in range(num_games)]
    done = [False] * num_games
    game_len = torch.zeros(num_games, dtype=torch.long, device=run_device)
    z_white = torch.zeros(num_games, dtype=torch.float32, device=run_device)

    piece_ids_list = []
    side_list = []
    castle_list = []
    ep_list = []
    hmc_list = []
    rep_list = []
    mask_list = []
    action_list = []
    logp_list = []
    game_id_list = []

    while not all(done):
        alive_indices = [i for i, d in enumerate(done) if not d]
        if not alive_indices:
            break
        alive_envs = [envs[i] for i in alive_indices]
        piece_ids, global_state, _, masks = states_board_and_masks(
            alive_envs, device=run_device
        )

        if not masks.any(dim=1).all():
            bad = torch.where(~masks.any(dim=1))[0].tolist()
            raise RuntimeError(f"No legal actions for env indices: {bad}")

        with torch.no_grad():
            logits = model_old(piece_ids, global_state, masks)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        logp = dist.log_prob(actions)

        piece_ids_list.append(piece_ids)
        side, castle, ep, hmc, rep = global_state
        side_list.append(side)
        castle_list.append(castle)
        ep_list.append(ep)
        hmc_list.append(hmc)
        rep_list.append(rep)
        mask_list.append(masks)
        action_list.append(actions)
        logp_list.append(logp)
        game_id_list.append(torch.tensor(alive_indices, device=run_device))

        for j, env_idx in enumerate(alive_indices):
            reward, done_step, _ = envs[env_idx].play_move(actions[j].item())
            game_len[env_idx] += 1
            if done_step:
                done[env_idx] = True
                z_white[env_idx] = reward

        if max_plies is not None and game_len.max().item() >= max_plies:
            for env_idx in alive_indices:
                if not done[env_idx]:
                    done[env_idx] = True
                    z_white[env_idx] = 0.0

    piece_ids_tensor = torch.cat(piece_ids_list, dim=0)
    side_tensor = torch.cat(side_list, dim=0)
    castle_tensor = torch.cat(castle_list, dim=0)
    ep_tensor = torch.cat(ep_list, dim=0)
    hmc_tensor = torch.cat(hmc_list, dim=0)
    rep_tensor = torch.cat(rep_list, dim=0)
    masks_tensor = torch.cat(mask_list, dim=0)
    actions_tensor = torch.cat(action_list, dim=0)
    logp_tensor = torch.cat(logp_list, dim=0)
    game_id_tensor = torch.cat(game_id_list, dim=0)

    global_state = (side_tensor, castle_tensor, ep_tensor, hmc_tensor, rep_tensor)
    return {
        "piece_ids": piece_ids_tensor,
        "global_state": global_state,
        "mask": masks_tensor,
        "action": actions_tensor,
        "logp_old": logp_tensor,
        "side": side_tensor,
        "game_id": game_id_tensor,
        "game_len": game_len,
        "z_white": z_white,
    }


def ppo_grpo_update(model: ChessModel,
                    optimizer: optim.Optimizer,
                    rollout: dict,
                    adv: torch.Tensor,
                    weights: torch.Tensor,
                    clip_eps: float,
                    ent_coef: float,
                    ppo_epochs: int,
                    minibatch_plies: int):
    piece_ids = rollout["piece_ids"]
    global_state = rollout["global_state"]
    mask = rollout["mask"]
    actions = rollout["action"]
    logp_old = rollout["logp_old"]

    num_plies = actions.size(0)
    last_loss = None
    for _ in range(ppo_epochs):
        indices = torch.randperm(num_plies, device=actions.device)
        for start in range(0, num_plies, minibatch_plies):
            idx = indices[start:start + minibatch_plies]
            piece_mb = piece_ids[idx]
            global_mb = tuple(t[idx] for t in global_state)
            mask_mb = mask[idx]
            action_mb = actions[idx]
            logp_old_mb = logp_old[idx]
            adv_mb = adv[idx]
            weight_mb = weights[idx]

            logits = model(piece_mb, global_mb, mask_mb)
            logp = F.log_softmax(logits, dim=-1).gather(
                1, action_mb.unsqueeze(1)
            ).squeeze(1)

            ratio = torch.exp(logp - logp_old_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb
            pg_loss = -torch.min(surr1, surr2)

            dist = Categorical(logits=logits)
            ent = dist.entropy()

            loss = (pg_loss * weight_mb).mean() - ent_coef * ent.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss.detach()
    return last_loss


def train(model: ChessModel,
          optimizer: optim.Optimizer,
          ckpt_path: str,
          log_csv: str,
          epochs: int = 1000,
          batch: int = BATCH,
          save_int: int = SAVE_INT,
          device_override: torch.device | None = None,
          ppo_epochs: int = PPO_EPOCHS,
          minibatch_plies: int = MINIBATCH_PLIES,
          max_plies: int | None = MAX_PLIES,
          eval_every: int | None = None,
          eval_fn=None):

    run_device = device_override or device
    model.to(run_device).train()
    count_parameters(model)

    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['Ep', 'Loss', 'MeanZ', 'StdZ', 'Plies', 'Games']
        )

    model_old = ChessModel().to(run_device)
    for ep in range(1, epochs + 1):
        model_old.load_state_dict(model.state_dict())
        model_old.eval()

        rollout = collect_rollouts(
            model_old, batch, run_device, max_plies=max_plies
        )

        z_white = rollout["z_white"]
        a_game = compute_group_advantages(z_white)
        side = rollout["side"]
        ones = torch.ones_like(side, dtype=torch.float32)
        sign = torch.where(side == 0, ones, -ones).to(run_device)
        adv = sign * a_game[rollout["game_id"]]

        game_len = rollout["game_len"].clamp_min(1).float()
        weights = (1.0 / game_len)[rollout["game_id"]]

        loss = ppo_grpo_update(
            model,
            optimizer,
            rollout,
            adv,
            weights,
            clip_eps=CLIP_EPS,
            ent_coef=ENT_COEF,
            ppo_epochs=ppo_epochs,
            minibatch_plies=minibatch_plies,
        )

        mean_z = z_white.mean().item()
        std_z = z_white.std(unbiased=False).item()
        plies = rollout["action"].size(0)

        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow(
                [ep, f"{loss.item():+.6f}" if loss is not None else "nan",
                 f"{mean_z:+.3f}", f"{std_z:+.3f}", plies, batch]
            )

        print(
            f"[Ep{ep}] loss={loss.item() if loss is not None else 0.0:+.4f} "
            f"z̄={mean_z:+.3f} σ={std_z:+.3f} plies={plies}"
        )

        if save_int and ep % save_int == 0:
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Checkpoint] saved at epoch {ep}")
            if eval_fn:
                model.eval()
                eval_fn(model, ep)
                model.train()

        if eval_every and eval_fn and ep % eval_every == 0:
            model.eval()
            eval_fn(model, ep)
            model.train()

    torch.save(model.state_dict(), ckpt_path)
    print("Training finished.")


if __name__ == '__main__':
    CKPT = 'chess_model_transformer_weights_exp2.pth'
    LOG = 'grpo_stepwise_log.csv'
    LR = 1e-4

    model = ChessModel().to(device)
    if os.path.exists(CKPT):
        try:
            model.load_state_dict(torch.load(CKPT, map_location=device),
                                  strict=False)
            print("Model loaded.")
        except Exception as e:
            print("Load error:", e)
    else:
        print("No checkpoint found — training from scratch.")

    opt = optim.AdamW(model.parameters(), lr=LR)
    train(model, opt, CKPT, LOG, batch=BATCH, save_int=SAVE_INT)
