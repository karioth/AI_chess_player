import random

import chess
import torch

from Chess_background import ChessGame, states_board_and_masks


def _select_action(logits, mask, policy, temperature):
    if policy == "sample":
        temp = max(temperature, 1e-3)
        probs = torch.softmax(logits / temp, dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
    else:
        action = torch.argmax(logits).item()
    if not mask[action]:
        legal = torch.nonzero(mask, as_tuple=False).flatten()
        if legal.numel() == 0:
            raise RuntimeError("No legal actions available in arena eval.")
        action = legal[0].item()
    return action


def run_arena(candidate_model,
              champion_model,
              games: int,
              run_device: torch.device,
              policy: str = "argmax",
              temperature: float = 1.0,
              seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)

    candidate_model.eval()
    champion_model.eval()

    wins = draws = losses = 0
    total_plies = 0

    for i in range(games):
        if i % 2 == 0:
            candidate_color = chess.WHITE
        else:
            candidate_color = chess.BLACK

        env = ChessGame()
        plies = 0

        while True:
            piece_ids, global_state, _, masks = states_board_and_masks([env], device=run_device)
            mask = masks[0]
            if not mask.any():
                raise RuntimeError("No legal actions available in arena eval.")
            moved_color = env.current_agent_color
            if moved_color == candidate_color:
                with torch.no_grad():
                    logits = candidate_model(piece_ids, global_state, masks)[0]
            else:
                with torch.no_grad():
                    logits = champion_model(piece_ids, global_state, masks)[0]

            action = _select_action(logits, mask, policy, temperature)
            _, done, _ = env.play_move(action)
            plies += 1
            if done:
                z_white = env.z_white
                if z_white is None or z_white == 0.0:
                    draws += 1
                else:
                    if candidate_color == chess.WHITE:
                        candidate_score = z_white
                    else:
                        candidate_score = -z_white
                    if candidate_score > 0:
                        wins += 1
                    else:
                        losses += 1
                total_plies += plies
                break

    games_played = wins + draws + losses
    score = (wins + 0.5 * draws) / max(games_played, 1)
    avg_plies = total_plies / max(games_played, 1)
    return {
        "games": games_played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "avg_plies": avg_plies,
    }
