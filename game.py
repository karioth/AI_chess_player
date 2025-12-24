import pygame
import chess
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import os

from Model import ChessModel
from Chess_background import ChessGame, states_board_and_masks, index_to_move

pygame.init()

WIDTH, HEIGHT = 800, 800
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Шахматы')
SQ_SIZE = WIDTH // 8

def load_images():
    pieces = ['wp', 'bp', 'wr', 'br', 'wn', 'bn', 'wb', 'bb', 'wq', 'bq', 'wk', 'bk']
    images = {}
    for piece in pieces:
        img = pygame.image.load(os.path.join("images", f"{piece}.png"))
        images[piece] = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
    return images

IMAGES = load_images()

def draw_board(screen, board):
    colors = [pygame.Color(255, 255, 255), pygame.Color(50, 50, 50)]
    for r in range(8):
        for c in range(8):
            color = colors[(r + c) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
            piece = board.piece_at(chess.square(c, 7 - r))
            if piece:
                key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().lower()
                screen.blit(IMAGES[key], pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_end_game_message(screen, game, z_white, mv, reason):
    font = pygame.font.SysFont("Arial", 30, bold=True)
    overlay = pygame.Surface((WIDTH, 120), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, HEIGHT // 2 - 60))
    if z_white is None or z_white == 0.0:
        text1 = font.render("Draw", True, pygame.Color(255, 255, 255))
    elif z_white > 0:
        text1 = font.render("White Won!", True, pygame.Color(255, 255, 255))
    else:
        text1 = font.render("Black Won!", True, pygame.Color(255, 255, 255))
    text2 = font.render(f"Reason: {reason} by move {mv}", True, pygame.Color(255, 200, 200))
    screen.blit(text1, text1.get_rect(center=(WIDTH//2, HEIGHT//2 - 20)))
    screen.blit(text2, text2.get_rect(center=(WIDTH//2, HEIGHT//2 + 20)))

def reset_game():
    return ChessGame(game_id=1)

def main():
    device = torch.device('cpu')

    model = ChessModel().to(device)
    ckpt = 'chess_model_transformer_weights_exp2.pth'   # ← обновлённый путь
    if os.path.exists(ckpt):
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
            print("Model loaded successfully.")
        except Exception as e:
            print("Error loading model:", e)
    else:
        print("No model checkpoint found.")
    model.eval()

    game = ChessGame(game_id=1)
    clock = pygame.time.Clock()
    running = True
    z_white = None
    mv = None
    reason = None
    color_str = ''

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if game.game_over:
            draw_end_game_message(WINDOW, game, z_white, mv, reason)
            pygame.display.flip()
            waiting = True
            while waiting and running:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_r:
                            game = reset_game()
                            waiting = False
                        elif ev.key == pygame.K_d:
                            running = False
                            waiting = False
                clock.tick(1)
            continue

        draw_board(WINDOW, game.board)
        pygame.display.flip()
        clock.tick(2)

        # 1) get state & mask
        piece_ids, global_vec, _, masks_tensor = states_board_and_masks([game], device)

        # 2) forward in no_grad
        with torch.no_grad():
            logits = model(piece_ids, global_vec, masks_tensor)

        # 3) sample action
        probs = F.softmax(logits, dim=1)
        dist = Categorical(probs)
        action = dist.sample().item()
        mv = index_to_move(action, game.board)

        # 4) step environment
        (games, rewards, dones, z_whites, winner_colors,
         reasons, color_strs, valid_count, _) = ChessGame.process_moves([game], [action])

        game       = games[0]
        z_white    = z_whites[0]
        reason     = reasons[0]
        color_str  = color_strs[0]

        if dones[0]:
            game.game_over = True
            continue

        print(f"Move: {mv} Color: {color_str} Reward: {rewards[0]}")

    pygame.quit()

if __name__ == '__main__':
    main()
