# ♟️ AI\_chess\_player

A GRPO (terminal-only PPO-clip) Transformer agent trained to play chess from scratch using reinforcement learning.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/SargisVardanian/AI_chess_player.git
cd AI_chess_player

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install python-chess

# 3. Run a self-play game
python game.py
```

Upon first execution, `game.py` will automatically download the pretrained model weights from GitHub Releases.

---

## Training (CLI)

Use the `train.py` wrapper to override hyperparameters without editing code:

```bash
python train.py --epochs 10 --batch 8 --save-int 200 --lr 1e-4
```

Common options:

```bash
python train.py --ckpt my_ckpt.pth --log my_log.csv --device mps --resume
```

Periodic eval during training:

```bash
python train.py --epochs 20 --eval-every 2 --eval-games 6
```

Engine eval during training (requires Stockfish in PATH or `--eval-engine-path`):

```bash
python train.py --eval-every 5 --eval-opponent engine --eval-engine-path /path/to/stockfish
```

Checkpoint-vs-previous eval during training (default):

```bash
python train.py --eval-every 2 --eval-opponent checkpoint
```

By default, training runs eval on every checkpoint (set `--no-eval-on-checkpoint`
if you want to disable it).

---

## Evaluation (CLI)

Evaluate against a random baseline:

```bash
python eval.py --ckpt chess_model_transformer_weights_exp2.pth --games 20
```

Evaluate against Stockfish (if installed):

```bash
python eval.py --opponent engine --engine-path /path/to/stockfish --games 10
```

---

## UCI + fastchess + BayesElo (engine-style eval)

Expose the model as a UCI engine:

```bash
python uci_bot.py --ckpt chess_model_transformer_weights_exp2.pth
```

Run an engine-vs-engine match (requires `fastchess` in PATH):

```bash
bash scripts/run_fastchess.sh new_ckpt.pth old_ckpt.pth out.pgn
```

Convert PGN → Elo with BayesElo:

```text
ResultSet>readpgn out.pgn
ResultSet>elo
ResultSet-EloRating>mm
ResultSet-EloRating>exactdist
ResultSet-EloRating>ratings
```

The file `scripts/bayeselo_commands.txt` contains the same sequence.

---

## Pretrained Weights

| Filename                                   | Size   | SHA-256 Digest |
| ------------------------------------------ | ------ | -------------- |
| `chess_model_transformer_weights_exp2.pth` | 157 MB | `ed20e8b5…`    |

The weights are fetched at runtime from:

```
https://github.com/SargisVardanian/AI_chess_player/releases/download/v1.0-weights/chess_model_transformer_weights_exp2.pth
```

---

## Repository Structure

```
AI_chess_player/
├─ game.py                   # Single-game simulator + auto-weight download
├─ train.py                  # Training CLI wrapper
├─ learning.py               # GRPO training loop (critic-free)
├─ eval.py                   # Eval runner (random/engine/checkpoint)
├─ uci_bot.py                # UCI shim for engine-style eval
├─ Chess_background.py       # Environment + rules/encoding
├─ Model.py                  # Transformer-only policy network
├─ images/                   # Chess piece sprite assets
├─ .github/
│   └─ workflows/
│       └─ run_game.yml      # “play-demo” GitHub Actions workflow
├─ .gitattributes            # Git LFS configuration
├─ .gitignore
└─ README.md
```

---

## Technical Overview

### GRPO Training (`learning.py`)

Training uses a terminal-only, critic-free GRPO loop:

* roll out full self-play games with a frozen policy
* compute group-normalized terminal outcomes from White’s perspective
* assign per-ply advantages by flipping sign for Black-to-move
* perform PPO-clip updates over the stored plies (no value head)

Each epoch rolls out a batch of games and then runs multiple PPO epochs
over the collected plies. Checkpoints are saved every `save_int` epochs.

---

### Model Architecture (`Model.py`)

The agent uses a Transformer-only encoder with embedding-based inputs:

1. **Square Tokens (64)**
   Each square gets a token built by summing:
   * a learnable **position embedding** (one per square), and
   * a learnable **piece embedding** (empty + 12 pieces).

2. **CLS Token (Global Rule State)**
   The CLS token is a **sum of categorical embeddings**:
   side-to-move, four castling bits, en-passant file, halfmove bucket,
   and repetition bucket.

3. **Transformer Encoder**
   Six layers ($d_{\text{model}} = 512,\; h = 16$) process the 65 tokens
   (`[CLS] + 64 squares`).

4. **Heads**
   * **Policy:** MLP on the CLS token → 4672 logits.
   * **Value:** MLP on the CLS token → scalar $V(s)$.

---

## Interactive Execution on GitHub

### Codespaces

Launch an interactive development environment in seconds:

[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?repo=SargisVardanian/AI_chess_player)

### GitHub Actions Demo

1. Go to **Actions → play-demo**.
2. Click **Run workflow**.
3. Inspect the logs for a step-by-step record of the agent’s moves.

The workflow `.github/workflows/run_game.yml` sets up Python 3.10, installs dependencies, and executes `python game.py`.

---

## References

\[1] Schulman, J. et al. “Proximal Policy Optimization Algorithms.” arXiv:1707.06347 (2017).
